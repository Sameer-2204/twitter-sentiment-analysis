FROM python:3.10-slim

WORKDIR /app

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (cached layer) ───────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download NLTK data ───────────────────────────────────────
RUN python -c "import nltk; nltk.download('stopwords'); \
    nltk.download('punkt'); nltk.download('punkt_tab'); \
    nltk.download('wordnet')"

# ── Copy application code ────────────────────────────────────
COPY app/ app/

# ── Copy data and model artefacts ────────────────────────────
# These can also be mounted as volumes in production
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# ── Expose port ──────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Start server ─────────────────────────────────────────────
# workers=1 because ML models are loaded in-process memory.
# Multiple workers would duplicate RAM usage.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
