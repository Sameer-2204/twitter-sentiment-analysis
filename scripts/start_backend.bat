@echo off
REM ================================================
REM  Twitter Sentiment Analyzer - Start Backend
REM ================================================
echo.
echo ================================================
echo   Twitter Sentiment Analyzer - Start Backend
echo ================================================
echo.

REM ── Check Python ────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: Python is not installed or not in PATH.
    echo   Download from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo   Python found: 
python --version
echo.

REM ── Navigate to project root ────────────────────
cd /d "%~dp0.."
set "PROJECT_ROOT=%cd%"
echo   Project Root: %PROJECT_ROOT%

REM ── Activate virtual environment if it exists ───
if exist ".venv\Scripts\activate.bat" (
    echo   Activating virtual environment...
    call .venv\Scripts\activate.bat
    echo   Virtual environment activated.
) else if exist "venv\Scripts\activate.bat" (
    echo   Activating virtual environment...
    call venv\Scripts\activate.bat
    echo   Virtual environment activated.
) else (
    echo   WARNING: No virtual environment found.
    echo   Consider creating one:
    echo     python -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r backend\requirements.txt
    echo.
)

echo.

REM ── Navigate to backend and start ───────────────
cd backend
echo   Starting backend...
echo   Press Ctrl+C to stop.
echo.
echo ------------------------------------------------
echo.

python run.py

REM ── If we get here, something went wrong ────────
if %errorlevel% neq 0 (
    echo.
    echo   ERROR: Backend failed to start.
    echo   Make sure dependencies are installed:
    echo     pip install -r requirements.txt
    echo.
    pause
)
