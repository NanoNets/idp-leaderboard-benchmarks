#!/usr/bin/env bash
set -euo pipefail

# First-time setup for idp-leaderboard-benchmarks
# Creates venv, installs dependencies, configures .env, and validates.

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR=".venv"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=10

# ── Colors ──────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}▸${NC} $*"; }
ok()    { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
fail()  { echo -e "${RED}✗${NC} $*"; }

# ── Find Python ─────────────────────────────────────────────────────────────

find_python() {
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            local ver
            ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

echo ""
echo -e "${BOLD}═══ idp-leaderboard-benchmarks setup ═══${NC}"
echo ""

# ── 1. Check Python ─────────────────────────────────────────────────────────

info "Checking Python version (need ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+)..."
PYTHON=$(find_python) || {
    fail "Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+ not found."
    echo "  Install from https://www.python.org/downloads/"
    exit 1
}
PY_VERSION=$("$PYTHON" --version 2>&1)
ok "$PY_VERSION ($(command -v "$PYTHON"))"

# ── 2. Create virtualenv ────────────────────────────────────────────────────

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    ok "Virtual environment already exists at ${VENV_DIR}/"
else
    info "Creating virtual environment in ${VENV_DIR}/..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
ok "Activated venv ($(python --version))"

# ── 3. Install core dependencies ────────────────────────────────────────────

info "Installing core dependencies from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
ok "Core dependencies installed"

# ── 4. Install Playwright (for olmocr evaluation) ───────────────────────────

info "Installing Playwright chromium (needed for olmOCR evaluation)..."
if python -m playwright install chromium 2>/dev/null; then
    ok "Playwright chromium installed"
else
    warn "Playwright chromium install failed (non-fatal — only needed for olmOCR eval)"
fi

# ── 5. Install docext (for IDP benchmark) ───────────────────────────────────

DOCEXT_DIR="$REPO_ROOT/../docext"
if [ ! -d "$DOCEXT_DIR" ]; then
    info "Cloning docext into ${DOCEXT_DIR}..."
    git clone https://github.com/NanoNets/docext.git "$DOCEXT_DIR"
    ok "docext cloned"
else
    ok "docext already present at ${DOCEXT_DIR}"
fi

info "Installing docext (editable)..."
pip install -e "$DOCEXT_DIR" --quiet
ok "docext installed"

# Comment out CLASSIFICATION task — not used by idp-leaderboard benchmarks
# and its dataset loading slows down startup.
DOCEXT_CFG="$DOCEXT_DIR/configs/benchmark.yaml"
if [ -f "$DOCEXT_CFG" ] && grep -q "^  - CLASSIFICATION" "$DOCEXT_CFG"; then
    sed -i.bak 's/^  - CLASSIFICATION/  # - CLASSIFICATION/' "$DOCEXT_CFG" && rm -f "$DOCEXT_CFG.bak"
    ok "Commented out CLASSIFICATION task in docext config"
fi

# ── 6. Install optional dependencies ────────────────────────────────────────

info "Installing optional dependencies (json_repair, pandas)..."
pip install json_repair pandas --quiet 2>/dev/null || true
ok "Optional dependencies installed"

# ── 7. Create .env template ─────────────────────────────────────────────────

if [ -f ".env" ]; then
    ok ".env file already exists"
else
    info "Creating .env template..."
    cat > .env <<'ENVEOF'
# API keys — uncomment and fill in the ones you need

# For GPT models (via LiteLLM)
# OPENAI_API_KEY=sk-...

# For Gemini models (via LiteLLM)
# GOOGLE_API_KEY=AIza...

# For Claude models (via LiteLLM)
# ANTHROPIC_API_KEY=sk-ant-...

# For Nanonets model
# NANONETS_API_KEY=...

# For publishing results to HuggingFace
# HF_WRITE_TOKEN=hf_...
ENVEOF
    ok ".env template created — edit it with your API keys"
fi

# ── 8. Create directories ───────────────────────────────────────────────────

mkdir -p caches ground_truth results
ok "Directories ready (caches/, ground_truth/, results/)"

# ── 9. Validate setup ───────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}── Validation ──${NC}"

CHECKS_PASSED=0
CHECKS_TOTAL=0

check() {
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if python -c "$1" 2>/dev/null; then
        ok "$2"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        fail "$2"
    fi
}

check "import httpx"                       "httpx"
check "import fitz"                        "PyMuPDF"
check "import PIL"                         "Pillow"
check "import yaml"                        "PyYAML"
check "import litellm"                     "litellm"
check "from olmocr.bench.tests import load_tests" "olmocr[bench]"

CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
if python -c "from docext.benchmark.benchmark import NanonetsIDPBenchmark" 2>/dev/null; then
    ok "docext (IDP benchmark)"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
else
    warn "docext not available (IDP benchmark will not work)"
fi

# ── 10. Check for OmniDocBench data ─────────────────────────────────────────

OMNIDOC_DIR="$REPO_ROOT/../OmniDocBench"
CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
OMNIDOC_FOUND=false
for candidate in "$OMNIDOC_DIR/OmniDocBench.json" "$REPO_ROOT/OmniDocBench/OmniDocBench.json"; do
    if [ -f "$candidate" ]; then
        ok "OmniDocBench.json found at $(dirname "$candidate")"
        OMNIDOC_FOUND=true
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
        break
    fi
done
if [ "$OMNIDOC_FOUND" = false ]; then
    info "Cloning OmniDocBench dataset into ${OMNIDOC_DIR}..."
    if git clone https://huggingface.co/datasets/opendatalab/OmniDocBench "$OMNIDOC_DIR" 2>&1; then
        ok "OmniDocBench cloned"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        warn "OmniDocBench clone failed (non-fatal — only needed for OmniDocBench benchmark)"
        echo "    You can clone it manually later:"
        echo "    git clone https://huggingface.co/datasets/opendatalab/OmniDocBench ../OmniDocBench"
    fi
fi

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}═══ Setup complete (${CHECKS_PASSED}/${CHECKS_TOTAL} checks passed) ═══${NC}"
echo ""
echo "  Activate the venv:    source ${VENV_DIR}/bin/activate"
echo "  Configure API keys:   edit .env"
echo ""
echo "  Quick test:"
echo "    python benchmarks/olmocr/run.py --model test --provider litellm \\"
echo "      --model-id openai/gpt-4.1-mini --workers 1 --categories arxiv_math"
echo ""
