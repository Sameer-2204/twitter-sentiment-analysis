# Twitter Sentiment Analysis API

REST API for Twitter sentiment analysis using 5 ML/DL models. Powers the React dashboard with real-time prediction, EDA analytics, and model comparison.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI + Uvicorn |
| Traditional ML | scikit-learn (Logistic Regression + TF-IDF) |
| Deep Learning | TensorFlow (LSTM, BiLSTM, CNN) |
| Transformer | PyTorch + HuggingFace Transformers (DistilBERT) |
| Data | pandas, NumPy |
| Config | Pydantic Settings |
| Deployment | Docker, Render |

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env with your settings

# 4. Ensure data and models exist
# Place train_data.csv in data/
# Place all model files in models/

# 5. Run development server
uvicorn app.main:app --reload --port 8000
```

## API Documentation

Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Endpoints

### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service health & readiness |
| GET | `/api/cors-test` | CORS configuration debug |

### Dashboard
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dashboard/stats` | Aggregate statistics |
| GET | `/api/dashboard/recent-tweets` | Paginated tweet list |
| GET | `/api/dashboard/sentiment-trend` | Sentiment trend over batches |

### EDA
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/eda/class-distribution` | Sentiment class counts |
| GET | `/api/eda/word-frequency` | Top words by frequency |
| GET | `/api/eda/bigrams` | Top bigrams |
| GET | `/api/eda/trigrams` | Top trigrams |
| GET | `/api/eda/tweet-lengths` | Length distribution stats |
| GET | `/api/eda/wordcloud-data` | Word→count map for wordcloud |
| GET | `/api/eda/hashtags` | Top hashtags |
| GET | `/api/eda/mentions` | Top @mentions |

### Models
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models/comparison` | All 5 models compared |
| GET | `/api/models/confusion-matrix/{name}` | Confusion matrix |
| GET | `/api/models/training-history/{name}` | Training curves (DL only) |
| GET | `/api/models/available` | Currently loaded models |

### Prediction
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/predict/` | Single text prediction |
| POST | `/api/predict/all` | All-models comparison |
| POST | `/api/predict/batch` | CSV batch prediction |
| GET | `/api/predict/sample-csv` | Download sample CSV |

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, CORS, startup/shutdown
│   ├── config.py            # Pydantic BaseSettings
│   ├── dependencies.py      # DI functions for Depends()
│   ├── middleware/
│   │   └── rate_limiter.py  # Per-IP rate limiting (30 req/min)
│   ├── schemas/
│   │   ├── prediction.py    # Request/response models
│   │   ├── dashboard.py
│   │   ├── eda.py
│   │   └── models.py
│   ├── services/
│   │   ├── text_preprocessor.py  # Lightweight text cleaner
│   │   ├── data_service.py       # Dataset loading & stats
│   │   ├── eda_service.py        # Word freq, n-grams, etc.
│   │   ├── model_service.py      # Comparison, confusion, history
│   │   └── predictor.py          # ML inference engine
│   └── routes/
│       ├── dashboard.py
│       ├── eda.py
│       ├── models.py
│       └── predict.py
├── scripts/
│   ├── check_deployment.py  # Pre-deploy readiness checker
│   └── test_api.py          # Smoke-test all endpoints
├── requirements.txt
├── Dockerfile
├── render.yaml
├── .env.example
└── .gitignore
```

## Deployment (Render)

### Option 1 — Deploy via Dashboard

1. Go to [render.com](https://render.com) → Sign in with GitHub
2. Click **New +** → **Web Service**
3. Connect your `twitter-sentiment-analysis` repository
4. Configure:
   - **Root Directory:** `backend`
   - **Runtime:** Docker
   - **Instance Type:** Free
5. Add environment variables:

| Variable | Value |
|----------|-------|
| `LIGHTWEIGHT_MODE` | `false` |
| `LAZY_LOADING` | `true` |
| `ALLOWED_ORIGINS` | `https://your-frontend.vercel.app` |

6. Click **Deploy Web Service**

### Option 2 — Deploy via Blueprint

1. Push the repo with `render.yaml` in `backend/`
2. Go to Render Dashboard → **New +** → **Blueprint**
3. Select the repository — Render reads `render.yaml` automatically

### After Deployment

Your API will be available at:
```
https://twitter-sentiment-api-xxxx.onrender.com
```

- Swagger UI: `https://your-url.onrender.com/docs`
- Health check: `https://your-url.onrender.com/api/health`

> **Note:** Render free tier spins down after 15 min of inactivity. First request after idle takes ~30-50s (cold start).

## Memory & Performance

- **LIGHTWEIGHT_MODE=false**: All 5 models available. LR loads at startup, heavy models lazy-load on first request.
- **LAZY_LOADING=true**: Only one heavy model kept in RAM at a time. Previous heavy model unloaded automatically.
- **Rate limiting**: 30 predictions/minute per IP on prediction endpoints.
- **Cache headers**: Dashboard (5 min), EDA (10 min), Models (1 hr), Predictions (no-cache).

## Scripts

```bash
# Check deployment readiness
python scripts/check_deployment.py

# Test all API endpoints (server must be running)
python scripts/test_api.py
```
