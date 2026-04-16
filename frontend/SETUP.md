# Frontend Setup Guide

## Quick Start (Local Backend)

```bash
# 1. Start backend
cd backend
python run.py

# 2. Start frontend
cd frontend
npm run dev
```

Frontend connects to `http://localhost:8000` by default.

---

## With Cloudflare Tunnel

```bash
# 1. Start backend
cd backend
python run.py

# 2. Start tunnel (separate terminal)
cloudflared tunnel --url http://localhost:8000

# 3. Copy the tunnel URL (e.g. https://abc-xyz.trycloudflare.com)
```

4. Update `frontend/.env`:
```
VITE_API_BASE=https://abc-xyz.trycloudflare.com
```

5. Start frontend:
```bash
npm run dev
```

---

## Deploy to Vercel

1. Push code to GitHub

2. Connect repo to Vercel (import the `frontend/` directory)

3. Set environment variable in Vercel dashboard:
   - **Name:** `VITE_API_BASE`
   - **Value:** Your Cloudflare tunnel URL

4. Deploy

> **Important:** Your local computer must be running the backend + tunnel
> for the deployed Vercel frontend to work!

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VITE_API_BASE` | Backend API URL (no trailing slash) | `https://abc-xyz.trycloudflare.com` |

---

## ConnectionStatus Component

The `<ConnectionStatus />` component shows live backend connectivity:

- 🟢 **Connected** — API is reachable (shows latency)
- 🔴 **Offline** — API is unreachable (shows troubleshooting steps)
- 🟡 **Checking** — Testing connection

Import from common components:
```tsx
import { ConnectionStatus } from "./components/common";

// Place in your navbar, topbar, or layout:
<ConnectionStatus />

// Optional: custom poll interval (default 30s)
<ConnectionStatus pollInterval={60000} />
```

---

## Notes

- Tunnel URL **changes every restart** unless you set up a named tunnel
- Your computer must stay **ON** for the backend to be accessible
- Backend runs with **hot-reload** — code changes apply automatically
- Frontend runs with **Vite HMR** — UI updates instantly
