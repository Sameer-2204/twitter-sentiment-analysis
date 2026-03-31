@echo off
REM ================================================
REM  Twitter Sentiment Analyzer - Start Everything
REM ================================================
REM  Opens two windows:
REM    1. Backend server (FastAPI + uvicorn)
REM    2. Cloudflare tunnel (after 10s delay)
REM ================================================
echo.
echo ================================================
echo   Twitter Sentiment Analyzer - Start Everything
echo ================================================
echo.
echo   Starting backend in a new window...

start "Twitter Sentiment - Backend" cmd /k "cd /d "%~dp0" && start_backend.bat"

echo   Waiting 10 seconds for backend to initialize...
echo   (Models need time to load into memory)
timeout /t 10 /nobreak >nul

echo   Starting Cloudflare Tunnel in a new window...

start "Twitter Sentiment - Tunnel" cmd /k "cd /d "%~dp0" && start_tunnel.bat"

echo.
echo ================================================
echo   Both services are starting!
echo.
echo   Backend window:  FastAPI server on localhost:8000
echo   Tunnel window:   Cloudflare tunnel URL
echo.
echo   Next steps:
echo     1. Wait for backend to finish loading models
echo     2. Copy the tunnel URL from the Tunnel window
echo     3. Update frontend .env with the tunnel URL
echo     4. Deploy frontend to Vercel
echo ================================================
echo.
pause
