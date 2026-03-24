"""
routes/predict.py — Endpoints for live sentiment prediction.

Mounted at ``/api/predict`` in main.py.
Rate-limited to prevent abuse on paid inference endpoints.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from typing import List

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.middleware.rate_limiter import rate_limiter
from app.schemas.prediction import (
    AllModelsRequest,
    AllModelsResponse,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.predictor import predictor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

# ── Allowed model names ───────────────────────────────────────
_ALLOWED_MODELS = {
    "logistic_regression", "lstm", "bilstm", "cnn", "distilbert",
}


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────


@router.post(
    "/",
    response_model=PredictionResponse,
    summary="Predict sentiment for a single text",
    description=(
        "Runs the specified model on the given text and returns the "
        "predicted sentiment label, confidence score, and per-class "
        "probability distribution."
    ),
)
def predict_single(request: PredictionRequest, req: Request):
    """Predict sentiment for a single text using the chosen model."""
    rate_limiter.check_rate_limit(req)

    # Validate model name
    if request.model_name not in _ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model '{request.model_name}'. "
                f"Allowed: {sorted(_ALLOWED_MODELS)}"
            ),
        )

    start = time.time()
    try:
        result = predictor.predict(request.text, request.model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.time() - start
    logger.info(
        "POST /predict | model=%s | text_len=%d | label=%s | conf=%.1f%% | %.3fs",
        request.model_name,
        len(request.text),
        result.label,
        result.confidence,
        elapsed,
    )
    return result


@router.post(
    "/all",
    response_model=AllModelsResponse,
    summary="Get predictions from all 5 models for comparison",
    description=(
        "Runs every loaded model on the same text and returns "
        "individual predictions, the consensus label, and how many "
        "models agree."
    ),
)
def predict_all_models(request: AllModelsRequest, req: Request):
    """Run every loaded model on the same text and return consensus."""
    rate_limiter.check_rate_limit(req)
    try:
        return predictor.predict_all_models(request.text)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("All-models prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Batch predict sentiment for uploaded CSV file",
    description=(
        "Accepts a CSV file with a 'text' column (case-insensitive: "
        "text, Text, TEXT, tweet, Tweet). Returns predictions for every "
        "row plus an aggregate summary. Max file size: 5 MB, max 5000 rows."
    ),
)
async def predict_batch(
    req: Request,
    file: UploadFile = File(
        ..., description="CSV file with a 'text' column"
    ),
    model_name: str = Query(
        "distilbert",
        description="Model to use for prediction",
    ),
):
    rate_limiter.check_rate_limit(req)
    """Predict sentiment for every row in an uploaded CSV file."""
    # Validate model name
    if model_name not in _ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model '{model_name}'. "
                f"Allowed: {sorted(_ALLOWED_MODELS)}"
            ),
        )

    # Validate content type
    allowed_types = (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    )
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted.",
        )

    # Read file
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum 5 MB allowed.",
        )

    # Parse CSV — find a text column (case-insensitive)
    try:
        decoded = contents.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded))
        fieldnames = reader.fieldnames or []

        # Case-insensitive search for text column
        text_col = None
        for name in fieldnames:
            if name.strip().lower() in ("text", "tweet", "content", "message"):
                text_col = name
                break

        if text_col is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "CSV must contain a 'text' column. "
                    f"Found columns: {fieldnames}"
                ),
            )

        texts: List[str] = [
            row[text_col]
            for row in reader
            if row.get(text_col, "").strip()
        ]
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File is not valid UTF-8 encoded.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"CSV parsing error: {exc}",
        )

    if not texts:
        raise HTTPException(
            status_code=400,
            detail="CSV contains no valid text rows.",
        )

    if len(texts) > 5000:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum 5000 rows allowed, found {len(texts)}.",
        )

    logger.info(
        "Batch prediction: %d rows with model '%s'",
        len(texts),
        model_name,
    )

    try:
        return predictor.predict_batch(texts, model_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/sample-csv",
    summary="Download sample CSV file",
    description=(
        "Returns a sample CSV file with 5 example tweets that can be "
        "used to test the batch prediction endpoint."
    ),
)
def download_sample_csv():
    """Return a sample CSV file with example tweets."""
    sample_rows = [
        "text",
        "I absolutely love this new product! Best purchase I've made all year.",
        "The stock market crashed today, wiping out billions in value.",
        "Apple just announced their quarterly earnings report for Q3.",
        "This is the worst customer service experience I've ever had.",
        "The weather is nice today, going for a walk in the park.",
    ]
    csv_content = "\n".join(sample_rows) + "\n"

    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="sample_tweets.csv"',
        },
    )
