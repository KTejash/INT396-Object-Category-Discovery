@echo off
title Object Discovery Lab — Launcher
color 0B

echo.
echo ============================================================
echo   Object Discovery Lab — Interactive Unsupervised AI Platform
echo   INT396 Project Launcher (Windows)
echo ============================================================
echo.

:: ── 1. Check Python ──────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo         Download Python 3.10+ from https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found.

:: ── 2. Navigate to project root ──────────────────────────────
cd /d "%~dp0"
echo [OK] Working directory: %CD%

:: ── 3. Create virtual environment if missing ─────────────────
if not exist ".venv\" (
    echo.
    echo [SETUP] Creating virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

:: ── 4. Activate virtual environment ──────────────────────────
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [OK] Virtual environment activated.

:: ── 5. Upgrade pip silently ───────────────────────────────────
echo.
echo [SETUP] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip is up to date.

:: ── 6. Install / verify requirements ─────────────────────────
echo.
echo [SETUP] Installing dependencies from requirements.txt...
echo         (This may take a few minutes on first run)
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Dependency installation failed.
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)
echo [OK] All dependencies installed.

:: ── 7. Create necessary directories if missing ───────────────
if not exist "data\raw"          mkdir data\raw
if not exist "data\processed"    mkdir data\processed
if not exist "data\labels"       mkdir data\labels
if not exist "models"            mkdir models
if not exist "experiments"       mkdir experiments
if not exist "outputs\embeddings" mkdir outputs\embeddings
if not exist "outputs\figures"   mkdir outputs\figures
if not exist "outputs\reports"   mkdir outputs\reports
echo [OK] Project directories verified.

:: ── 8. Launch Streamlit app ───────────────────────────────────
echo.
echo ============================================================
echo   Launching Object Discovery Lab...
echo   Open your browser at: http://localhost:8501
echo   Press Ctrl+C in this window to stop the app.
echo ============================================================
echo.

streamlit run app\streamlit_app.py --server.headless false --browser.gatherUsageStats false

:: ── On exit ──────────────────────────────────────────────────
echo.
echo [INFO] App stopped. Press any key to close this window.
pause >nul
