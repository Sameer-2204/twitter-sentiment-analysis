FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    PORT=7860 \
    DEPLOYMENT_TARGET=hf \
    LIGHTWEIGHT_MODE=false \
    LAZY_LOADING=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gcc \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && success=0; \
    for i in 1 2 3 4 5; do \
      if pip install --no-cache-dir --retries 20 --timeout 180 --resume-retries 20 -r requirements.txt; then \
        success=1; \
        break; \
      fi; \
      echo "pip install failed (attempt ${i}/5), retrying in 20s..."; \
      sleep 20; \
    done; \
    if [ "$success" -ne 1 ]; then \
      echo "pip install failed after 5 attempts"; \
      exit 1; \
    fi

RUN python -c "import nltk; \
    nltk.download('stopwords', quiet=True); \
    nltk.download('punkt', quiet=True); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('wordnet', quiet=True)"

COPY backend/app/ app/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD ["sh", "-c", "curl -f http://127.0.0.1:${PORT:-7860}/api/health || exit 1"]

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
