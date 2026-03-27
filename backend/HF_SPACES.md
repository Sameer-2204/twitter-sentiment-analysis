# Hugging Face Spaces Deployment (Free)

Use this project as a Docker Space to run all 5 models on free CPU hardware.

## 1) Create the Space

1. Go to Hugging Face -> `New Space`
2. Select `Docker` SDK
3. Choose `CPU Basic (free)`
4. Connect/push this repository

## 2) Ensure the Docker build uses root `Dockerfile`

This repo now includes a root `Dockerfile` configured for Spaces:

- `DEPLOYMENT_TARGET=hf`
- `LIGHTWEIGHT_MODE=false`
- `LAZY_LOADING=false`
- app port defaults to `7860`

## 3) Set Space variables

In Space settings, add:

- `ALLOWED_ORIGINS=https://twitter-sentiment-analysis-mocha.vercel.app,http://localhost:5173`
- `DEBUG=false`

Optional:

- `MAX_SEQUENCE_LENGTH=128`

## 4) Validate backend

Run:

```bash
python backend/scripts/validate_rollout.py \
  --base-url https://<your-space-subdomain>.hf.space \
  --mode hf-full \
  --stability-rounds 2 \
  --sleep-seconds 5
```

## 5) Frontend cutover

When validator passes all checks:

1. Set Vercel `VITE_API_BASE=https://<your-space-subdomain>.hf.space`
2. Redeploy frontend
3. Smoke test predict + batch routes
