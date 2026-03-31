# Local Development Scripts

## Quick Start (Windows)

```
cd scripts
start_all.bat
```

This opens two windows:
1. **Backend** — FastAPI server on `localhost:8000`
2. **Tunnel** — Cloudflare tunnel exposing your backend to the internet

Then:
1. Wait for models to load in the Backend window
2. Copy the tunnel URL from the Tunnel window (looks like `https://xxxxx.trycloudflare.com`)
3. Update frontend `.env` with the tunnel URL:
   ```
   VITE_API_BASE=https://xxxxx.trycloudflare.com
   ```
4. Deploy frontend to Vercel

---

## Individual Scripts

| Script | Purpose |
|--------|---------|
| `start_backend.bat` | Activate venv → start FastAPI server with hot-reload |
| `start_tunnel.bat` | Start Cloudflare tunnel → expose `localhost:8000` |
| `start_all.bat` | Launch both in separate windows |

---

## Prerequisites

### Python Dependencies

```
cd backend
pip install -r requirements.txt
```

### Cloudflare Tunnel (`cloudflared`)

**Option 1 — winget (recommended):**
```
winget install Cloudflare.cloudflared
```

**Option 2 — manual download:**
1. Go to [cloudflared releases](https://github.com/cloudflare/cloudflared/releases)
2. Download `cloudflared-windows-amd64.exe`
3. Rename to `cloudflared.exe`
4. Place in a folder that's in your `PATH`

---

## Architecture

```
Your Computer                    Internet
┌─────────────────┐    ┌──────────────────┐    ┌────────────┐
│  FastAPI Backend │───▶│ Cloudflare Tunnel │───▶│   Vercel   │
│  localhost:8000  │    │  xxxxx.trycloudflare  │   Frontend  │
└─────────────────┘    └──────────────────┘    └────────────┘
```

## Notes

- Backend must be running before starting the tunnel
- Free tunnel URL **changes every restart** (use a named tunnel for a permanent URL)
- Your computer must stay **ON** for the backend to be accessible
- Free Cloudflare tunnel is perfect for demos and development
- Backend runs with **hot-reload** — code changes apply automatically
