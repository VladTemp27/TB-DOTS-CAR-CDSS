import json
import logging
import sys

from fastapi import APIRouter, Request
from fastapi.exceptions import HTTPException
from sse_starlette.sse import EventSourceResponse

import backend.llm as llm_mod
from backend.config import N_CTX
from backend.inference import generate
from backend.prompt import build_prompt
from backend.schemas import InterpretRequest

log = logging.getLogger("medgemma")
router = APIRouter()


@router.get("/api/health")
def health():
    _backend = llm_mod._stats["backend"]
    if llm_mod._load_error:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "error",
                "error": llm_mod._load_error,
                "model": "medgemma-1.5-4b-it-IQ4_XS",
                "n_ctx": N_CTX,
                "backend": _backend,
            },
        )
    if llm_mod.llm is None:
        return {"status": "loading", "model": "medgemma-1.5-4b-it-IQ4_XS", "n_ctx": N_CTX, "backend": _backend}
    return {"status": "ready", "model": "medgemma-1.5-4b-it-IQ4_XS", "n_ctx": N_CTX, "backend": _backend}


@router.get("/api/stats")
async def stats_endpoint(request: Request):
    import resource as _resource
    lock = request.app.state.llm_lock
    try:
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        ram_mb = rss // (1024 * 1024) if sys.platform == "darwin" else rss // 1024
    except Exception:
        ram_mb = 0
    return {
        "status": "error" if llm_mod._load_error else ("loading" if llm_mod.llm is None else "ready"),
        "error": llm_mod._load_error,
        "inference_active": lock.locked(),
        "ram_mb": ram_mb,
        **llm_mod._stats,
    }


@router.post("/api/interpret")
async def interpret(req: InterpretRequest, request: Request):
    log.info(
        "POST /api/interpret — patient=%r failure_prob=%.1f%% contributions=%d",
        req.patient_name,
        req.failure_probability * 100,
        len(req.contributions),
    )

    if llm_mod.llm is None:
        error_msg = llm_mod._load_error or "Model not loaded"
        log.warning("Rejecting request — model unavailable: %s", error_msg)

        async def err():
            yield {"data": json.dumps({"error": error_msg, "done": True})}

        return EventSourceResponse(err())

    lock = request.app.state.llm_lock
    prompt = build_prompt(req)
    log.debug("Prompt built (%d chars)", len(prompt))
    return EventSourceResponse(generate(prompt, lock, req.patient_name, request))
