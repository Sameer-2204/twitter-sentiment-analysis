# Twitter Sentiment Analysis API

FastAPI service for sentiment analysis with 5 models:
- Logistic Regression + TF-IDF
- LSTM
- BiLSTM
- CNN
- DistilBERT

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Docs:
- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

## Deployment on Hugging Face Spaces (Free, recommended)

Hugging Face Spaces free CPU hardware is typically enough to keep all 5 models available.

1. Create a new Space with `Docker` SDK.
2. Push this repo to the Space.
3. The root `Dockerfile` is already configured for Spaces:
   - `DEPLOYMENT_TARGET=hf`
   - `LIGHTWEIGHT_MODE=false`
   - `LAZY_LOADING=false`
4. Set Space variables:
   - `ALLOWED_ORIGINS=https://twitter-sentiment-analysis-mocha.vercel.app,http://localhost:5173`
   - `DEBUG=false`

Detailed guide: `backend/HF_SPACES.md`

## Deployment on Render

### Recommended: parallel rollout from GHCR image

This is the safest path when model artefacts are local and large.

1. Build and push image:

```bash
docker build -f backend/Dockerfile -t ghcr.io/sameer-2204/twitter-sentiment-api:latest .
docker push ghcr.io/sameer-2204/twitter-sentiment-api:latest
```

2. In Render:
- New Web Service -> Existing Image
- Image: `ghcr.io/sameer-2204/twitter-sentiment-api:latest`
- Plan: free

3. Configure env vars:

| Variable | Value |
|---|---|
| `LIGHTWEIGHT_MODE` | `true` |
| `LAZY_LOADING` | `true` |
| `ALLOWED_ORIGINS` | `https://twitter-sentiment-analysis-mocha.vercel.app,http://localhost:5173` |
| `DEBUG` | `false` |

4. Keep the old Render service running during verification for rollback.

### Alternative: blueprint deploy (repo-based)

If all required data/models are available in build context:
- keep root `render.yaml`
- deploy via Render Blueprint

## Validation before frontend cutover

Run the rollout validator against the backend URL:

```bash
cd backend
python scripts/validate_rollout.py --base-url https://your-new-service.onrender.com
# optional free-tier cold-start check
python scripts/validate_rollout.py --base-url https://your-new-service.onrender.com --idle-seconds 960
# Hugging Face full-mode profile
python scripts/validate_rollout.py --base-url https://your-space.hf.space --mode hf-full
```

It verifies:
- `/api/health` and `/docs`
- CORS against frontend origin
- single prediction for each model
- `/api/predict/all` includes all 5 models
- repeated stability rounds

## Frontend cutover (Vercel)

After validator passes:
1. Set `VITE_API_BASE` to new backend URL in Vercel project settings
2. Redeploy production
3. Smoke test dashboard + predict pages

## Runtime behavior

- `DEPLOYMENT_TARGET=auto` (default): auto-detects runtime and picks defaults
- Render-like target: `LIGHTWEIGHT_MODE=true`, `LAZY_LOADING=true`
- Hugging Face target: `LIGHTWEIGHT_MODE=false`, `LAZY_LOADING=false`
- You can always override via environment variables.

## Scripts

```bash
# preflight checks
python scripts/check_deployment.py

# local endpoint smoke test
python scripts/test_api.py

# live rollout validation
python scripts/validate_rollout.py --base-url https://your-new-service.onrender.com
```
