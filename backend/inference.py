import asyncio
import json
import logging
import threading
import time
from typing import AsyncGenerator

from fastapi import Request

import backend.llm as llm_mod
from backend.config import MAX_TOKENS, REPEAT_PENALTY, TEMPERATURE, TOP_P

log = logging.getLogger("medgemma")


async def generate(
    prompt: str, lock: asyncio.Lock, patient_name: str, request: Request
) -> AsyncGenerator:
    log.debug("Inference queued for %r — prompt length: %d chars", patient_name, len(prompt))
    yield {"data": json.dumps({"status": "thinking"})}
    async with lock:
        llm_mod._stats["requests_active"] += 1
        log.info("Inference started for %r", patient_name)
        t0 = time.monotonic()
        token_count = 0

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()

        def run_inference():
            nonlocal token_count
            try:
                log.debug("llm() call entered for %r", patient_name)
                for chunk in llm_mod.llm(  # type: ignore[misc]
                    prompt,
                    stream=True,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repeat_penalty=REPEAT_PENALTY,
                ):
                    if stop_event.is_set():
                        log.debug(
                            "Inference cancelled for %r after %d tokens",
                            patient_name, token_count,
                        )
                        break
                    token = chunk.get("choices", [{}])[0].get("text", "")
                    if token:
                        token_count += 1
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                log.exception("Inference error for %r: %s", patient_name, e)
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

        t = threading.Thread(target=run_inference, daemon=True)
        t.start()
        try:
            while True:
                if await request.is_disconnected():
                    log.info("Client disconnected for %r — stopping inference", patient_name)
                    break
                try:
                    kind, value = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if kind == "token":
                    yield {"data": json.dumps({"token": value})}
                elif kind == "done":
                    llm_mod._stats["requests_served"] += 1
                    elapsed = time.monotonic() - t0
                    log.info(
                        "Inference done for %r — %d tokens in %.1fs (%.1f tok/s)",
                        patient_name, token_count, elapsed,
                        token_count / elapsed if elapsed > 0 else 0,
                    )
                    yield {"data": json.dumps({"token": "", "done": True})}
                    break
                elif kind == "error":
                    log.error("Inference stream error for %r: %s", patient_name, value)
                    yield {"data": json.dumps({"error": value, "done": True})}
                    break
        finally:
            stop_event.set()
            await asyncio.to_thread(t.join)
            llm_mod._stats["requests_active"] -= 1
            llm_mod._stats["last_patient"] = patient_name
            llm_mod._stats["last_tokens"] = token_count
            llm_mod._stats["last_elapsed_s"] = round(time.monotonic() - t0, 1)
