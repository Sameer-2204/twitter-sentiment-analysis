"""
main.py — FastAPI application entry point (local development).

Registers CORS middleware, request-ID logging, cache-control headers,
and runs a startup preloading sequence for ML models and dataset.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import configure_logging, get_settings
from app.routes import dashboard, eda, models, predict
from app.services.data_service import data_service
from app.services.eda_service import eda_service
from app.services.model_service import model_service
from app.services.predictor import predictor

# ── Settings & logging ────────────────────────────────────────
settings = get_settings()
configure_logging(settings)
logger = logging.getLogger(__name__)

# ── Startup timestamp (for uptime tracking) ───────────────────
_startup_time: float = 0.0

# ── FastAPI instance ──────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "REST API for Twitter sentiment analysis with 5 ML/DL models. "
        "Provides dashboard statistics, EDA analytics, model comparison "
        "metrics, and live/batch sentiment prediction."
    ),
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "API health, readiness, and info"},
        {"name": "Dashboard", "description": "Dashboard statistics and trends"},
        {"name": "EDA", "description": "Exploratory data analysis endpoints"},
        {"name": "Models", "description": "Model comparison, confusion matrix, training curves"},
        {"name": "Prediction", "description": "Single, multi-model, and batch sentiment prediction"},
    ],
)

# ── CORS middleware ───────────────────────────────────────────
_raw_origins = settings.allowed_origins_list

if "*" in _raw_origins:
    # Allow all origins — useful during local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,   # credentials can't be used with "*"
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )
else:
    # Specific origins — add common local dev ports as fallback
    _origins = list(set(_raw_origins + [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
    ]))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"],
    )


# ── Request ID + logging middleware ───────────────────────────
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Inject a unique request ID, log method/path/status/duration,
    and catch unhandled exceptions cleanly."""
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    start = time.time()

    try:
        response = await call_next(request)
    except Exception as exc:
        duration = time.time() - start
        logger.error(
            "[%s] %s %s → 500 (%.3fs) UNHANDLED: %s",
            request_id, request.method, request.url.path, duration, exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
        )

    duration = time.time() - start

    # Attach tracing headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration:.3f}s"

    # Skip logging for noisy health-check polls
    if request.url.path != "/api/health":
        logger.info(
            "[%s] %s %s → %d (%.3fs)",
            request_id, request.method, request.url.path,
            response.status_code, duration,
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
    if request.method != "GET":
        return response

    path = request.url.path
    for prefix, cache_value in _CACHE_RULES.items():
        if path.startswith(prefix):
            response.headers["Cache-Control"] = cache_value
            return response

    if path.startswith("/api/predict"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

    return response


# ── Route registration ────────────────────────────────────────
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(eda.router, prefix="/api/eda", tags=["EDA"])
app.include_router(models.router, prefix="/api/models", tags=["Models"])
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])


# ── Helpers ───────────────────────────────────────────────────

def _timed(label: str, fn, *args, **kwargs):
    """Run *fn* and print a timed status line. Returns (result, elapsed)."""
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info("   ✅ %s (%.1fs)", label, elapsed)
        return result, elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("   ❌ %s failed (%.1fs): %s", label, elapsed, exc)
        return None, elapsed


# ── Startup event ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Load dataset, pre-compute statistics, and load ML models."""
    global _startup_time
    _startup_time = time.time()
    start = _startup_time

    # ── Banner ────────────────────────────────────────────────
    print()
    logger.info("=" * 56)
    logger.info("🐦 TWITTER SENTIMENT ANALYZER — STARTING UP")
    logger.info("=" * 56)
    logger.info("Environment: LOCAL DEVELOPMENT")
    logger.info("Debug Mode:  %s", "ON" if settings.DEBUG else "OFF")
    logger.info("Log Level:   %s", settings.LOG_LEVEL)
    logger.info("Python:      %s", sys.version.split()[0])
    logger.info("Platform:    %s %s", platform.system(), platform.machine())
    logger.info("-" * 56)

    # ── Paths ─────────────────────────────────────────────────
    logger.info("📁 Paths:")
    logger.info("   Models:  %s %s", settings.MODELS_DIR,
                "✅" if settings.MODELS_DIR.exists() else "⚠️ MISSING")
    logger.info("   Data:    %s %s", settings.DATA_DIR,
                "✅" if settings.DATA_DIR.exists() else "⚠️ MISSING")
    logger.info("   Reports: %s %s", settings.REPORTS_DIR,
                "✅" if settings.REPORTS_DIR.exists() else "⚠️ MISSING")
    logger.info("-" * 56)

    # ── Load services ─────────────────────────────────────────
    logger.info("⏳ Loading services...")

    _timed("Data service loaded", data_service.load_data)

    if data_service.loaded:
        logger.info("   📊 Dataset: %d rows", len(data_service.df))
        _timed("EDA service loaded", eda_service.precompute)

    _timed("Model comparison loaded", model_service.load_comparison_data)

    # ── Load ML models ────────────────────────────────────────
    logger.info("   ⏳ Loading ML models...")

    model_load_start = time.time()
    try:
        predictor.load_all_models()
    except Exception as exc:
        logger.error("   ❌ Model loading failed: %s", exc)

    model_load_elapsed = time.time() - model_load_start
    available = predictor.get_available_models()
    failed = list(predictor.load_errors.keys())
    logger.info("   ✅ All models loaded (%.1fs) — %d available, %d failed",
                model_load_elapsed, len(available), len(failed))

    # ── Enrich dataset with real sentiment predictions ─────
    if data_service.loaded:
        logger.info("   ⏳ Predicting real sentiments for dataset (VADER)...")
        _timed("Sentiment enrichment", data_service.enrich_with_predictions)

        # Re-precompute EDA stats with real sentiments
        eda_service._cache.clear()
        _timed("EDA re-precomputed", eda_service.precompute)

    if failed:
        logger.warning("   ⚠️  Failed models: %s", failed)

    # ── Summary ───────────────────────────────────────────────
    total_time = time.time() - start
    logger.info("-" * 56)
    logger.info("🚀 API Ready! (%.1fs total)", total_time)
    logger.info("   Local: http://localhost:%d", settings.API_PORT)
    logger.info("   Docs:  http://localhost:%d/docs", settings.API_PORT)
    logger.info("=" * 56)
    print()


# ── Shutdown event ────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down %s...", settings.APP_NAME)


# ── Health check ──────────────────────────────────────────────
@app.get("/api/health", tags=["Health"])
def health_check():
    """Health check with model status and uptime."""
    available = predictor.get_available_models()
    models_ready = len(available) >= len(settings.MODEL_NAMES)

    models_status = {
        name: name in available for name in settings.MODEL_NAMES
    }

    ready = data_service.loaded and models_ready

    # Uptime
    uptime_seconds = time.time() - _startup_time if _startup_time else 0

    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "data_loaded": data_service.loaded,
        "models_loaded": models_status,
        "models_available": available,
        "load_errors": predictor.load_errors,
        "load_times": predictor.load_times,
        "uptime_seconds": round(uptime_seconds, 1),
        "version": settings.VERSION,
    }


# ── API info endpoint ────────────────────────────────────────
@app.get("/api/info", tags=["Health"])
def api_info():
    """Return API metadata and available models."""
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "available_models": predictor.get_available_models(),
        "all_model_names": settings.MODEL_NAMES,
        "total_tweets": len(data_service.df) if data_service.loaded else 0,
        "data_loaded": data_service.loaded,
        "docs_url": "/docs",
        "endpoints": {
            "health": "/api/health",
            "info": "/api/info",
            "predict": "/api/predict",
            "predict_all": "/api/predict/all",
            "predict_batch": "/api/predict/batch",
            "models_status": "/api/predict/models",
            "dashboard": "/api/dashboard",
            "eda": "/api/eda",
        },
    }


# ── CORS test endpoint ───────────────────────────────────────
@app.get("/api/cors-test", tags=["Health"])
def cors_test(request: Request):
    """Debug endpoint to verify CORS is configured correctly."""
    return {
        "message": "CORS is working",
        "origin": request.headers.get("origin", "no origin header"),
        "allowed_origins": settings.ALLOWED_ORIGINS,
    }


# ── Root ──────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Root endpoint — basic API info."""
    return {
        "message": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/api/health",
    }
