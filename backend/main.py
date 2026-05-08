import asyncio
import json
import threading
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
# Model lifecycle
# ---------------------------------------------------------------------------

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment,misc]

llm: "Llama | None" = None
_load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, _load_error
    app.state.llm_lock = asyncio.Lock()  # Created inside running event loop
    if Llama is not None:
        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            _load_error = f"Model file not found: {model_path}"
            print(f"[ERROR] {_load_error}")
        else:
            try:
                llm = await asyncio.to_thread(
                    Llama,
                    model_path=MODEL_PATH,
                    n_ctx=N_CTX,
                    n_threads=N_THREADS,
                    n_gpu_layers=N_GPU_LAYERS,
                    verbose=False,
                )
                print(f"[INFO] MedGemma loaded successfully from {MODEL_PATH}")
            except Exception as exc:
                _load_error = str(exc)
                print(f"[ERROR] Model failed to load: {exc}")
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


async def _generate(prompt: str, lock: asyncio.Lock) -> AsyncGenerator:
    async with lock:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()

        def run_inference():
            try:
                for chunk in llm(  # type: ignore[misc]
                    prompt,
                    stream=True,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repeat_penalty=REPEAT_PENALTY,
                ):
                    if stop_event.is_set():
                        break
                    token = chunk.get("choices", [{}])[0].get("text", "")
                    if token:  # Skip empty tokens
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

        t = threading.Thread(target=run_inference, daemon=True)
        t.start()
        try:
            while True:
                kind, value = await queue.get()
                if kind == "token":
                    yield {"data": json.dumps({"token": value})}
                elif kind == "done":
                    yield {"data": json.dumps({"token": "", "done": True})}
                    break
                elif kind == "error":
                    yield {"data": json.dumps({"error": value, "done": True})}
                    break
        finally:
            stop_event.set()
            await asyncio.to_thread(t.join)  # Wait for thread to finish before releasing lock


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


@app.post("/api/interpret")
async def interpret(req: InterpretRequest, request: Request):
    if llm is None:
        error_msg = _load_error or "Model not loaded"

        async def err():
            yield {"data": json.dumps({"error": error_msg, "done": True})}

        return EventSourceResponse(err())
    lock = request.app.state.llm_lock
    prompt = build_prompt(req)
    return EventSourceResponse(_generate(prompt, lock))
