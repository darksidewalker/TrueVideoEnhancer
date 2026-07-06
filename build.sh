#!/usr/bin/env bash
#
# Build script for DaSiWa TrueVideoEnhancer
# Sets up Python venv, installs deps, builds Go binary.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure ~/.local/bin is on PATH (where uv usually lives)
export PATH="$HOME/.local/bin:$PATH"

echo "=========================================="
echo "DaSiWa TrueVideoEnhancer - Build Script"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

die() { log_error "$@"; exit 1; }

# ── Resolve uv ────────────────────────────────────────────────
resolve_uv() {
    if command -v uv &>/dev/null; then
        echo "uv"
    elif [ -f "./runtime/bin/uv" ]; then
        echo "./runtime/bin/uv"
    else
        die "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
}

UV=$(resolve_uv)
log_info "Using uv: $($UV --version)"

# ── Step 1: Python venv + dependencies ──────────────────────
log_info "Step 1: Python virtual environment & dependencies"

VENV_PYTHON="./runtime/venv/bin/python"

if [ ! -d "./runtime/venv" ]; then
    log_info "Creating venv (Python 3.12)…"
    $UV venv --python 3.12 runtime/venv
fi

log_info "Installing requirements…"
$UV pip install --python "$VENV_PYTHON" --index-strategy unsafe-best-match \
    -r backend/requirements.txt

log_info "Requirements installed."

# ── Step 2: Go binary ───────────────────────────────────────
log_info "Step 2: Building Go binary…"

if ! command -v go &>/dev/null; then
    die "Go not found. Install Go 1.24+ from https://go.dev/dl/"
fi

go build -o dasiwa-true-video-enhancer ./cmd/dasiwa-true-video-enhancer/
chmod +x dasiwa-true-video-enhancer

log_info "Binary built: ./dasiwa-true-video-enhancer ($(du -h ./dasiwa-true-video-enhancer | cut -f1))"

# ── Step 3: Verification ────────────────────────────────────
echo ""
log_info "=== Verification ==="

"$VENV_PYTHON" -c "
import sys, importlib
ok = True
for pkg in ['torch', 'safetensors']:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', '?')
        print(f'  ✓ {pkg}: {v}')
    except Exception as e:
        print(f'  ✗ {pkg}: {e}')
        ok = False
sys.exit(0 if ok else 1)
"

log_info "Go: $(go version)"
log_info "Binary ready: ./dasiwa-true-video-enhancer"

echo ""
echo -e "${GREEN}✓ Build successful!${NC}"
echo ""
echo "Usage: ./dasiwa-true-video-enhancer --listen :8080 --root ~/Videos"
echo ""
