@echo off
REM ================================================
REM  Twitter Sentiment Analyzer - Cloudflare Tunnel
REM ================================================
echo.
echo ================================================
echo   Twitter Sentiment Analyzer - Cloudflare Tunnel
echo ================================================
echo.

REM ── Check if cloudflared is installed ───────────
cloudflared version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: cloudflared is not installed.
    echo.
    echo   ================================================
    echo   How to install cloudflared on Windows:
    echo   ================================================
    echo.
    echo   Option 1 - winget (recommended):
    echo     winget install Cloudflare.cloudflared
    echo.
    echo   Option 2 - Download manually:
    echo     1. Go to: https://github.com/cloudflare/cloudflared/releases
    echo     2. Download: cloudflared-windows-amd64.exe
    echo     3. Rename to: cloudflared.exe
    echo     4. Move to a folder in your PATH
    echo        (e.g., C:\Windows or create C:\Tools and add to PATH)
    echo.
    echo   After installing, run this script again.
    echo.
    pause
    exit /b 1
)

echo   cloudflared found:
cloudflared version
echo.

REM ── Check if backend is running ─────────────────
echo   Checking if backend is running on localhost:8000...
curl -s http://localhost:8000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   WARNING: Backend does not appear to be running.
    echo   Start the backend first with: start_backend.bat
    echo   Continuing anyway (tunnel will wait for backend)...
    echo.
) else (
    echo   Backend is running!
    echo.
)

REM ── Start Cloudflare Tunnel ─────────────────────
echo ------------------------------------------------
echo.
echo   Starting Cloudflare Tunnel...
echo   This will expose localhost:8000 to the internet.
echo.
echo   IMPORTANT: Copy the tunnel URL that appears below!
echo   It will look like: https://xxxxx.trycloudflare.com
echo   Use this URL in your frontend .env as VITE_API_BASE
echo.
echo ------------------------------------------------
echo.

cloudflared tunnel --url http://localhost:8000

REM ── If we get here, tunnel stopped ──────────────
echo.
echo   Tunnel stopped.
pause
