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
| Deployment | Docker, Railway |

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
├── requirements.txt
├── Dockerfile
├── railway.json
├── .env.example
└── .gitignore
```

## Deployment (Railway)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and link
railway login
railway link

# Deploy
railway up
```

### Environment Variables (set in Railway dashboard)

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `False` |
| `ALLOWED_ORIGINS` | CORS origins (comma-separated) | `localhost` |
| `PORT` | Server port (set by Railway) | `8000` |

> **Note:** Railway uses `$PORT` environment variable. The `railway.json` start command already references it.

## Memory Considerations

Loading all 5 models requires ~3-4 GB RAM. For Railway free tier:

- Use `tensorflow-cpu` instead of `tensorflow`
- Use `torch` CPU-only build
- See comments in `requirements.txt` for instructions
