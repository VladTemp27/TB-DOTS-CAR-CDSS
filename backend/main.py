import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.llm import lifespan
from backend.routers.ai import router as ai_router
from backend.routers.patients import router as patients_router
from backend.routers.xrays import router as xrays_router

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="MedGemma Clinical CDSS API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router)
app.include_router(patients_router)
app.include_router(xrays_router)
