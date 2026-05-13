import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

try:
    import llama_cpp
    from llama_cpp import Llama
except ImportError:
    llama_cpp = None  # type: ignore
    Llama = None  # type: ignore[assignment,misc]

from backend.config import (
    APPLE_SILICON,
    FLASH_ATTN,
    MODEL_PATH,
    N_BATCH,
    N_CTX,
    N_GPU_LAYERS,
    N_THREADS,
    N_THREADS_BATCH,
    TYPE_K,
    TYPE_V,
)

log = logging.getLogger("medgemma")

llm: "Llama | None" = None
_load_error: str | None = None
_stats: dict = {
    "backend": "metal" if APPLE_SILICON else "cuda",
    "requests_served": 0,
    "requests_active": 0,
    "last_patient": None,
    "last_tokens": 0,
    "last_elapsed_s": 0.0,
    "model_load_time_s": 0.0,
}

_BACKEND_LABEL = "Metal (Apple Silicon)" if APPLE_SILICON else "CUDA"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, _load_error
    app.state.llm_lock = asyncio.Lock()

    if sys.platform == "darwin" and not APPLE_SILICON:
        log.warning("macOS detected but APPLE_SILICON=False — running under Rosetta or Intel Mac; Metal offload disabled")

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
        "Loading MedGemma via %s (size=%.1f GB, n_ctx=%d, n_threads=%d, n_gpu_layers=%d, n_batch=%d, flash_attn=%s)...",
        _BACKEND_LABEL,
        model_path.stat().st_size / 1e9,
        N_CTX,
        N_THREADS,
        N_GPU_LAYERS,
        N_BATCH,
        FLASH_ATTN,
    )
    t0 = time.monotonic()
    try:
        _llm_kwargs: dict = dict(
            model_path=str(model_path),
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            n_threads=N_THREADS,
            n_threads_batch=N_THREADS_BATCH,
            n_gpu_layers=N_GPU_LAYERS,
            flash_attn=FLASH_ATTN,
            verbose=False,
        )
        if TYPE_K is not None:
            _llm_kwargs["type_k"] = TYPE_K
        if TYPE_V is not None:
            _llm_kwargs["type_v"] = TYPE_V

        llm = await asyncio.to_thread(Llama, **_llm_kwargs)
        elapsed = time.monotonic() - t0
        _stats["model_load_time_s"] = round(elapsed, 1)
        log.info("MedGemma ready — loaded in %.1fs", elapsed)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        log.exception("Model load failed after %.1fs", time.monotonic() - t0)
    yield
