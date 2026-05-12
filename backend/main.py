import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config import (
    MAX_TOKENS,
    MODEL_PATH,
    N_CTX,
    N_GPU_LAYERS,
    N_THREADS,
    REPEAT_PENALTY,
    TEMPERATURE,
    TOP_P,
)
from prompt import build_prompt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("medgemma")

# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment,misc]

llm: "Llama | None" = None
_load_error: str | None = None

# Runtime stats — updated during inference, read by /api/stats
_stats: dict = {
    "requests_served": 0,
    "requests_active": 0,
    "last_patient": None,
    "last_tokens": 0,
    "last_elapsed_s": 0.0,
    "model_load_time_s": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, _load_error
    app.state.llm_lock = asyncio.Lock()  # Created inside running event loop

    if Llama is None:
        _load_error = "llama_cpp not installed — run: pip install llama-cpp-python"
        log.error(_load_error)
        yield
        return

    model_path = Path(MODEL_PATH)
    log.info("Model path resolved to: %s", model_path.resolve())

    if not model_path.exists():
        _load_error = f"Model file not found: {model_path.resolve()}"
        log.error(_load_error)
        yield
        return

    log.info(
        "Loading MedGemma (size=%.1f GB, n_ctx=%d, n_threads=%d, n_gpu_layers=%d)...",
        model_path.stat().st_size / 1e9,
        N_CTX,
        N_THREADS,
        N_GPU_LAYERS,
    )
    t0 = time.monotonic()
    try:
        llm = await asyncio.to_thread(
            Llama,
            model_path=str(model_path),
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )
        elapsed = time.monotonic() - t0
        _stats["model_load_time_s"] = round(elapsed, 1)
        log.info("MedGemma ready — loaded in %.1fs", elapsed)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        log.exception("Model load failed after %.1fs", time.monotonic() - t0)
    yield


app = FastAPI(title="MedGemma Clinical CDSS API", lifespan=lifespan)

# CORS — allow all origins for development (no credentials needed for SSE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ContributionItem(BaseModel):
    feature: str
    delta: float
    direction: Literal["risk", "protective"]


class InterpretRequest(BaseModel):
    patient_name: str
    age: int
    sex: Literal["M", "F"]
    bacteriologic_status: str
    microscopy_result: str
    anatomical_site: Literal["P", "EP"]
    registration_group: str
    source_of_patient: str
    type: str
    days_to_treatment: int
    failure_probability: float
    contributions: list[ContributionItem]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


async def _generate(prompt: str, lock: asyncio.Lock, patient_name: str, request: Request) -> AsyncGenerator:
    log.debug(
        "Inference queued for %r — prompt length: %d chars", patient_name, len(prompt)
    )
    # Yield before acquiring the lock so the SSE byte stream opens immediately
    # (sse_starlette needs a first yield to finalise response headers).
    # requests_active is incremented inside the lock so it only reflects
    # requests that are actually running inference, not ones waiting to acquire.
    yield {"data": json.dumps({"status": "thinking"})}
    async with lock:
        _stats["requests_active"] += 1
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
                for chunk in llm(  # type: ignore[misc]
                    prompt,
                    stream=True,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repeat_penalty=REPEAT_PENALTY,
                ):
                    if stop_event.is_set():
                        log.debug("Inference cancelled for %r after %d tokens", patient_name, token_count)
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
                    continue  # re-poll is_disconnected during long prefill phase
                if kind == "token":
                    yield {"data": json.dumps({"token": value})}
                elif kind == "done":
                    _stats["requests_served"] += 1
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
            _stats["requests_active"] -= 1
            _stats["last_patient"] = patient_name
            _stats["last_tokens"] = token_count
            _stats["last_elapsed_s"] = round(time.monotonic() - t0, 1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health():
    if _load_error:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "error": _load_error,
                "model": "medgemma-1.5-4b-it-IQ4_XS",
                "n_ctx": N_CTX,
            },
        )
    if llm is None:
        return {"status": "loading", "model": "medgemma-1.5-4b-it-IQ4_XS", "n_ctx": N_CTX}
    return {"status": "ready", "model": "medgemma-1.5-4b-it-IQ4_XS", "n_ctx": N_CTX}


@app.get("/api/stats")
async def stats_endpoint(request: Request):
    import sys, resource as _resource
    lock = request.app.state.llm_lock
    try:
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        # macOS: bytes; Linux: kilobytes
        ram_mb = rss // (1024 * 1024) if sys.platform == "darwin" else rss // 1024
    except Exception:
        ram_mb = 0
    return {
        "status": "error" if _load_error else ("loading" if llm is None else "ready"),
        "error": _load_error,
        "inference_active": lock.locked(),
        "ram_mb": ram_mb,
        **_stats,
    }


@app.post("/api/interpret")
async def interpret(req: InterpretRequest, request: Request):
    log.info(
        "POST /api/interpret — patient=%r failure_prob=%.1f%% contributions=%d",
        req.patient_name,
        req.failure_probability * 100,
        len(req.contributions),
    )

    if llm is None:
        error_msg = _load_error or "Model not loaded"
        log.warning("Rejecting request — model unavailable: %s", error_msg)

        async def err():
            yield {"data": json.dumps({"error": error_msg, "done": True})}

        return EventSourceResponse(err())

    lock = request.app.state.llm_lock
    prompt = build_prompt(req)
    log.debug("Prompt built (%d chars)", len(prompt))
    return EventSourceResponse(_generate(prompt, lock, req.patient_name, request))
