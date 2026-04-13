FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── System dependencies ──────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── NLTK data (needed by EDA service) ────────────────────────
RUN python -c "import nltk; \
    nltk.download('stopwords', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('wordnet', quiet=True)"

# ── Copy application code ────────────────────────────────────
COPY backend/app/ app/

# ── Copy data (baked into image) ─────────────────────────────
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# ── Runtime environment ──────────────────────────────────────
ENV PORT=8080 \
    MODELS_DIR=/app/models \
    DATA_DIR=/app/data \
    REPORTS_DIR=/app/reports \
    ALLOWED_ORIGINS="http://localhost:5173" \
    DEBUG=false \
    LOG_LEVEL=INFO

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-8080}/api/health || exit 1

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
