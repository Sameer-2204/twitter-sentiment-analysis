"""
main.py — FastAPI application entry point.

Registers CORS middleware, request logging middleware, includes all
routers with cache-control headers, and runs a startup preloading
sequence for ML models and dataset.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import dashboard, eda, models, predict
from app.services.data_service import data_service
from app.services.eda_service import eda_service
from app.services.model_service import model_service
from app.services.predictor import predictor

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────
settings = get_settings()

# ── FastAPI instance ──────────────────────────────────────────
app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description=(
        "REST API for Twitter sentiment analysis with 5 ML/DL models. "
        "Provides dashboard statistics, EDA analytics, model comparison "
        "metrics, and live/batch sentiment prediction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "API health and readiness checks"},
        {"name": "Dashboard", "description": "Dashboard statistics and trends"},
        {"name": "EDA", "description": "Exploratory data analysis endpoints"},
        {"name": "Models", "description": "Model comparison, confusion matrix, training curves"},
        {"name": "Prediction", "description": "Single, multi-model, and batch sentiment prediction"},
    ],
)

# ── CORS middleware ───────────────────────────────────────────
_origins = [
    *settings.ALLOWED_ORIGINS,
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging middleware ────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with method, path, status, and duration."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        "%s %s → %d (%.3fs)",
        request.method,
        request.url.path,
        response.status_code,
        duration,
    )
    return response


# ── Cache-control middleware ──────────────────────────────────
_CACHE_RULES = {
    "/api/dashboard": "public, max-age=300",         # 5 min
    "/api/eda": "public, max-age=600",               # 10 min
    "/api/models": "public, max-age=3600",           # 1 hour
}


@app.middleware("http")
async def add_cache_headers(request: Request, call_next):
    """Attach Cache-Control headers for GET endpoints based on prefix."""
    response = await call_next(request)

    # Only cache GET requests
    if request.method != "GET":
        return response

    path = request.url.path
    for prefix, cache_value in _CACHE_RULES.items():
        if path.startswith(prefix):
            response.headers["Cache-Control"] = cache_value
            return response

    # Predictions and unknown routes: no caching
    if path.startswith("/api/predict"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

    return response


# ── Route registration ────────────────────────────────────────
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(eda.router, prefix="/api/eda", tags=["EDA"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])


# ── Startup event ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load dataset, pre-compute statistics, and load ML models."""
    start = time.time()
    logger.info("Starting up Twitter Sentiment Analysis API...")
    logger.info("LIGHTWEIGHT_MODE=%s | LAZY_LOADING=%s",
                settings.LIGHTWEIGHT_MODE, settings.LAZY_LOADING)

    # Step 1: Load dataset
    logger.info("Step 1/4 — Loading dataset...")
    try:
        data_service.load_data()
        logger.info("✓ Dataset loaded (%d rows).", len(data_service.df))
    except Exception as exc:
        logger.error("✗ Dataset loading failed: %s", exc)

    # Step 2: Pre-compute EDA statistics
    logger.info("Step 2/4 — Pre-computing EDA statistics...")
    try:
        if data_service.loaded:
            eda_service.precompute()
            logger.info("✓ EDA statistics pre-computed.")
    except Exception as exc:
        logger.error("✗ EDA pre-computation failed: %s", exc)

    # Step 3: Load model comparison data
    logger.info("Step 3/4 — Loading model comparison data...")
    try:
        model_service.load_comparison_data()
        logger.info("✓ Model comparison data loaded.")
    except Exception as exc:
        logger.error("✗ Model comparison loading failed: %s", exc)

    # Step 4: Load ML models
    logger.info("Step 4/4 — Loading ML models...")
    try:
        predictor.load_all_models()
        logger.info("✓ ML models loaded.")
    except Exception as exc:
        logger.error("✗ Model loading failed: %s", exc)

    total_time = time.time() - start
    logger.info("Startup complete in %.2f s", total_time)
    logger.info("Models loaded: %s", predictor.get_available_models())
    logger.info("Data loaded: %s", data_service.loaded)


# ── Shutdown event ────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown_event():
    """Log graceful shutdown."""
    logger.info("Shutting down Twitter Sentiment Analysis API...")


# ── Health check ──────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
def health_check():
    """Return service health and readiness status."""
    return {
        "status": "ok",
        "data_loaded": data_service.loaded,
        "models_loaded": predictor.loaded,
        "available_models": predictor.get_available_models(),
        "lightweight_mode": settings.LIGHTWEIGHT_MODE,
        "lazy_loading": settings.LAZY_LOADING,
        "version": settings.VERSION,
    }


# ── CORS test endpoint ───────────────────────────────────────
@app.get("/api/cors-test", tags=["Health"])
def cors_test(request: Request):
    """Debug endpoint to verify CORS is configured correctly."""
    return {
        "message": "CORS is working",
        "origin": request.headers.get("origin", "no origin header"),
        "allowed_origins": _origins,
    }


# ── Root ──────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Root endpoint — basic API info."""
    return {
        "message": "Twitter Sentiment Analysis API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }
