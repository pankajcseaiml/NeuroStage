"""
FastAPI application entry point.

Startup sequence:
  1. Create DB tables (if not exist)
  2. Load inference model into memory once
  3. Mount all routers

CORS is configured for local development. In production, restrict to your actual domain.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.eeg import router as eeg_router
from backend.api.experiments import router as exp_router
from backend.auth.router import router as auth_router
from backend.core.config import settings
from backend.db.database import Base, engine
from backend.services.inference import load_inference_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("neurostage.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    log.info("Starting NeuroStage API...")

    # Initialize database tables
    Base.metadata.create_all(bind=engine)
    log.info("Database tables initialized.")

    # Load ML model once at startup
    load_inference_model(settings.model_checkpoint, settings.model_config_path)

    yield

    log.info("Shutting down NeuroStage API.")


app = FastAPI(
    title="NeuroStage API",
    description=(
        "API for the NeuroStage lightweight CNN-Transformer EEG sleep stage classification system. "
        "This is a research prototype — not a clinically certified diagnostic system."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
cors_origins = list(settings.cors_origins)
for default_origin in ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"]:
    if default_origin not in cors_origins:
        cors_origins.append(default_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router)
app.include_router(eeg_router)
app.include_router(exp_router)


@app.get("/health")
def health():
    """Liveness/readiness probe."""
    from backend.services.inference import is_model_loaded
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "version": app.version,
    }


@app.get("/")
def root():
    return {"message": "NeuroStage API — see /docs for the OpenAPI interface."}
