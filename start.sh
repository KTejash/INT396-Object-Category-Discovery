#!/usr/bin/env bash
# ============================================================
#  Object Discovery Lab — Interactive Unsupervised AI Platform
#  INT396 Project Launcher (macOS / Linux)
# ============================================================

set -euo pipefail

# Colours
CYAN="\033[0;36m"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
RESET="\033[0m"

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }

echo ""
echo -e "${CYAN}============================================================${RESET}"
echo -e "${CYAN}  Object Discovery Lab — Interactive Unsupervised AI Platform${RESET}"
echo -e "${CYAN}  INT396 Project Launcher (macOS / Linux)${RESET}"
echo -e "${CYAN}============================================================${RESET}"
echo ""

# ── 1. Move to the script's own directory ────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
ok "Working directory: $SCRIPT_DIR"

# ── 2. Detect Python 3 (prefer python3, fall back to python) ──
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    error "Python 3 is not installed."
    echo "  macOS:  brew install python3   (or download from https://www.python.org)"
    echo "  Linux:  sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
ok "Python $PY_VERSION found  →  $($PYTHON -c 'import sys; print(sys.executable)')"

# Require Python 3.10+
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [[ $PY_MAJOR -lt 3 || ($PY_MAJOR -eq 3 && $PY_MINOR -lt 10) ]]; then
    error "Python 3.10 or newer is required (found $PY_VERSION)."
    exit 1
fi

# ── 3. Create virtual environment if missing ──────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment (.venv)..."
    $PYTHON -m venv "$VENV_DIR"
    ok "Virtual environment created."
else
    ok "Virtual environment already exists."
fi

# ── 4. Activate virtual environment ──────────────────────────
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
ok "Virtual environment activated."

# ── 5. Upgrade pip ───────────────────────────────────────────
info "Upgrading pip..."
pip install --upgrade pip --quiet
ok "pip is up to date."

# ── 6. Install dependencies ───────────────────────────────────
info "Installing dependencies from requirements.txt..."
echo "  (This may take a few minutes on first run)"
pip install -r requirements.txt --quiet
ok "All dependencies installed."

# ── 7. Ensure project directories exist ──────────────────────
mkdir -p data/raw data/processed data/labels
mkdir -p models experiments
mkdir -p outputs/embeddings outputs/figures outputs/reports
ok "Project directories verified."

# ── 8. Launch Streamlit ───────────────────────────────────────
echo ""
echo -e "${CYAN}============================================================${RESET}"
echo -e "${CYAN}  Launching Object Discovery Lab...${RESET}"
echo -e "${CYAN}  Open your browser at: http://localhost:8501${RESET}"
echo -e "${CYAN}  Press Ctrl+C to stop the app.${RESET}"
echo -e "${CYAN}============================================================${RESET}"
echo ""

# Open browser automatically after a short delay (macOS only)
if [[ "$(uname)" == "Darwin" ]]; then
    (sleep 3 && open "http://localhost:8501") &
fi

streamlit run app/streamlit_app.py \
    --server.headless false \
    --browser.gatherUsageStats false

echo ""
info "App stopped."
