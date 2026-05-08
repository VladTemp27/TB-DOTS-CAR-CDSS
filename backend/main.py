import json
from typing import Literal

from fastapi import FastAPI
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

app = FastAPI(title="MedGemma Clinical CDSS API")

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment,misc]

llm: "Llama | None" = None


@app.on_event("startup")
async def load_model() -> None:
    global llm
    if Llama is None:
        return
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
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
    sex: str
    bacteriologic_status: str
    microscopy_result: str
    anatomical_site: str
    registration_group: str
    source_of_patient: str
    type: str
    days_to_treatment: int
    failure_probability: float
    contributions: list[ContributionItem]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ready" if llm is not None else "loading",
        "model": "medgemma-1.5-4b-it-IQ4_XS",
        "n_ctx": N_CTX,
    }


@app.post("/api/interpret")
async def interpret(req: InterpretRequest) -> EventSourceResponse:
    async def generate():
        if llm is None:
            yield {"data": json.dumps({"error": "Model not loaded"})}
            return

        prompt = build_prompt(req)

        for chunk in llm(
            prompt,
            stream=True,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repeat_penalty=REPEAT_PENALTY,
        ):
            token = chunk["choices"][0]["text"]
            yield {"data": json.dumps({"token": token})}

        yield {"data": json.dumps({"token": "", "done": True})}

    return EventSourceResponse(generate())
