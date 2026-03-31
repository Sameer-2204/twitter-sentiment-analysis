# Twitter Sentiment Analysis

Full-stack sentiment analysis platform using 5 ML/DL models to classify tweets as **Positive**, **Negative**, or **Neutral**.

## Models

| Model | Type | Framework |
|-------|------|-----------|
| Logistic Regression | TF-IDF + Classifier | scikit-learn |
| LSTM | Recurrent Neural Network | TensorFlow/Keras |
| BiLSTM | Bidirectional LSTM | TensorFlow/Keras |
| CNN | 1D Convolutional Network | TensorFlow/Keras |
| DistilBERT | Fine-tuned Transformer | HuggingFace/PyTorch |

## Architecture

```
┌─────────────────────┐       ┌──────────────────────────────┐
│   Frontend (React)  │──────▶│   Backend (FastAPI)          │
│   Deployed: Vercel  │ HTTPS │   Deployed: Oracle Cloud VM  │
└─────────────────────┘       │   4 ARM cores, 24 GB RAM     │
                              │   All 5 models loaded         │
                              └──────────────────────────────┘
```

## Project Structure

```
twitter_analysis/
├── backend/                    ← FastAPI application
│   ├── app/
│   │   ├── config.py           ← Settings & environment handling
│   │   ├── main.py             ← App entry point
│   │   ├── routes/             ← API endpoints
│   │   ├── services/           ← Business logic & ML inference
│   │   ├── schemas/            ← Pydantic models
│   │   └── middleware/         ← Rate limiting, etc.
│   ├── scripts/
│   │   └── test_api.py         ← API test suite
│   ├── Dockerfile              ← ARM64-optimized container
│   └── requirements.txt
├── frontend/                   ← React + Vite application
├── deployment/                 ← Oracle Cloud VM infrastructure
│   ├── docker-compose.yml
│   ├── nginx.conf
│   ├── setup.sh                ← One-time VM bootstrap
│   ├── start.sh / stop.sh      ← Container management
│   ├── test-deployment.sh      ← Deployment verification
│   ├── copy-files.md           ← File transfer guide
│   ├── ssl-setup.md            ← HTTPS setup guide
│   └── README.md               ← Full deployment docs
├── models/                     ← ML model files (git-ignored)
├── data/                       ← Dataset files (git-ignored)
├── DEPLOYMENT_CHECKLIST.md     ← Step-by-step deployment checklist
└── README.md                   ← This file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check with model status |
| `/api/info` | GET | API metadata and available models |
| `/api/dashboard/stats` | GET | Dashboard statistics |
| `/api/eda/*` | GET | Exploratory data analysis |
| `/api/models/comparison` | GET | Model performance comparison |
| `/api/predict/` | POST | Single text prediction |
| `/api/predict/all` | POST | All-models parallel prediction |
| `/api/predict/batch` | POST | CSV batch prediction |
| `/api/predict/models` | GET | Model status and metadata |
| `/docs` | GET | Interactive Swagger UI |

## Local Development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env             # Edit VITE_API_BASE if needed
npm run dev
```

## Deployment

### Frontend → Vercel

1. Push `frontend/` to GitHub
2. Connect the repo to [Vercel](https://vercel.com)
3. Set environment variable: `VITE_API_BASE=https://api.yourdomain.com`
4. Deploy

### Backend → Oracle Cloud ARM VM

The backend runs on an **Oracle Cloud Ampere A1 VM** (4 OCPU, 24 GB RAM, Ubuntu 22.04 ARM64).

**Quick start on VM:**

```bash
# 1. Bootstrap the VM (one-time)
sudo bash deployment/setup.sh

# 2. Copy model files (see deployment/copy-files.md)
rsync -avzP models/ ubuntu@VM_IP:/opt/twitter-sentiment/models/

# 3. Configure
cp deployment/.env.example deployment/.env
nano deployment/.env              # Set ALLOWED_ORIGINS, etc.

# 4. Build & start
bash deployment/start.sh

# 5. Verify
bash deployment/test-deployment.sh
```

**Detailed guides:**

- 📋 [Deployment Checklist](DEPLOYMENT_CHECKLIST.md) — step-by-step with checkboxes
- 📖 [Deployment README](deployment/README.md) — comprehensive documentation
- 📁 [File Transfer Guide](deployment/copy-files.md) — SCP, rsync, Object Storage
- 🔒 [SSL Setup Guide](deployment/ssl-setup.md) — Let's Encrypt + Nginx

## Testing

```bash
# API test suite (requires running backend)
pip install httpx
python backend/scripts/test_api.py http://localhost:8000

# Deployment verification (on VM)
bash deployment/test-deployment.sh
```

## License

MIT
