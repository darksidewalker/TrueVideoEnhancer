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

mkdir -p dist
BUILD_TMP="./.dasiwa-true-video-enhancer.build"
rm -f "$BUILD_TMP"
go build -o "$BUILD_TMP" ./cmd/dasiwa-true-video-enhancer/
chmod +x "$BUILD_TMP"

# Linux keeps a running executable's old inode after replacement. Stop only
# instances launched from this repository so stale embedded assets cannot stay active.
for proc_exe in /proc/[0-9]*/exe; do
    [ -e "$proc_exe" ] || continue
    running_exe="$(readlink -f "$proc_exe" 2>/dev/null || true)"
    if [ "$running_exe" = "$SCRIPT_DIR/dasiwa-true-video-enhancer-linux-amd64" ]; then
        pid="${proc_exe#/proc/}"
        pid="${pid%/exe}"
        log_info "Stopping running app process $pid before replacing binary…"
        kill "$pid" 2>/dev/null || true
        for _ in {1..30}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.1
        done
    fi
done

mv -f "$BUILD_TMP" ./dasiwa-true-video-enhancer-linux-amd64
cp -f ./dasiwa-true-video-enhancer-linux-amd64 ./dist/dasiwa-true-video-enhancer-linux-amd64
chmod +x ./dasiwa-true-video-enhancer-linux-amd64 ./dist/dasiwa-true-video-enhancer-linux-amd64

log_info "Binary built: ./dasiwa-true-video-enhancer-linux-amd64 ($(du -h ./dasiwa-true-video-enhancer-linux-amd64 | cut -f1))"
log_info "Release binary: ./dist/dasiwa-true-video-enhancer-linux-amd64"

# ── Step 2b: Windows cross-build (pure Go, no CGo) ─────────
log_info "Step 2b: Cross-building Windows binaries…"
mkdir -p dist
for arch in amd64 arm64; do
    name="dasiwa-true-video-enhancer-windows-${arch}.exe"
    GOOS=windows GOARCH="$arch" CGO_ENABLED=0 \
        go build -ldflags="-s -w" -o "$name" ./cmd/dasiwa-true-video-enhancer/
    cp -f "$name" "./dist/$name"
    log_info "Windows binary built: ./$name ($(du -h "$name" | cut -f1))"
done

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
log_info "Running binary HTTP smoke test…"
SMOKE_PORT=18612
DASIWA_NO_BROWSER=1 DASIWA_PORT="$SMOKE_PORT" ./dasiwa-true-video-enhancer-linux-amd64 >./.build-smoke.log 2>&1 &
SMOKE_PID=$!
trap 'kill "$SMOKE_PID" 2>/dev/null || true; rm -f ./.build-smoke.log "$BUILD_TMP"' EXIT
smoke_ok=0
for _ in {1..50}; do
    if curl -fsS "http://127.0.0.1:${SMOKE_PORT}/api/health" >/dev/null 2>&1; then
        smoke_ok=1
        break
    fi
    kill -0 "$SMOKE_PID" 2>/dev/null || break
    sleep 0.1
done
if [ "$smoke_ok" -ne 1 ]; then
    cat ./.build-smoke.log >&2 || true
    die "Built binary failed HTTP smoke test"
fi
kill "$SMOKE_PID" 2>/dev/null || true
wait "$SMOKE_PID" 2>/dev/null || true
rm -f ./.build-smoke.log
trap - EXIT
log_info "Binary HTTP smoke test passed."
log_info "Binary ready: ./dasiwa-true-video-enhancer-linux-amd64"

echo ""
echo -e "${GREEN}✓ Build successful!${NC}"
echo ""
echo "Usage: ./dasiwa-true-video-enhancer-linux-amd64"
echo ""
