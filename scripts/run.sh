#!/usr/bin/env bash
# Build (optional) and start DaSiWa TrueVideoEnhancer with GPU passthrough on
# docker or podman. Engine detection per the container-engine-compat skill;
# override with CONTAINER_ENGINE=docker|podman.
#
#   ./scripts/run.sh                  # build image if missing, start detached
#   TVE_BUILD=1 ./scripts/run.sh      # force rebuild first
#   TVE_GPU=0 ./scripts/run.sh        # CPU-only (no GPU flags)
set -euo pipefail
DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Fall back to bin/ shims (podman -> docker) when podman isn't installed.
PATH="$PATH:$DIR/bin"
cd "$DIR"

IMAGE=${IMAGE:-dasiwa/tve:latest}
MODELS_DIR=${MODELS_DIR:-$DIR/models}
DATA_DIR=${DATA_DIR:-$DIR/data}
CACHE_DIR=${CACHE_DIR:-$DIR/cache}
PORT=${TVE_PORT:-8612}
mkdir -p "$MODELS_DIR" "$DATA_DIR" "$CACHE_DIR"

detect_engine() {
  local force="${CONTAINER_ENGINE:-}"
  if [[ -n "$force" ]]; then
    command -v "$force" >/dev/null 2>&1 && { echo "$force"; return 0; }
    return 1
  fi
  command -v docker >/dev/null 2>&1 && { echo docker; return 0; }
  command -v podman >/dev/null 2>&1 && { echo podman; return 0; }
  return 1
}

CE=$(detect_engine) || { echo "No container engine found (install docker or podman)"; exit 1; }
echo "Using container engine: $CE"

if [[ "${TVE_BUILD:-0}" == "1" ]]; then
  # podman's default OCI format drops HEALTHCHECK; build as docker format there.
  if [[ "$CE" == "podman" ]]; then
    "$CE" build --format docker -t "$IMAGE" .
  else
    "$CE" build -t "$IMAGE" .
  fi
fi

image_present() {
  # `docker image exists` doesn't exist; docker's equivalent is `image inspect`.
  # On podman hosts (or via the bin/ shim) `image exists` works as-is.
  case "$CE" in
    docker) "$CE" image inspect "$IMAGE" >/dev/null 2>&1 ;;
    *)      "$CE" image exists "$IMAGE" ;;
  esac
}

if image_present; then
  echo "Image $IMAGE present."
else
  echo "Image $IMAGE missing — building..."
  if [[ "$CE" == "podman" ]]; then
    "$CE" build --format docker -t "$IMAGE" .
  else
    "$CE" build -t "$IMAGE" .
  fi
fi

GPU_ARGS=()
SMI_ARG=()
if [[ "${TVE_GPU:-1}" == "1" ]]; then
  case "$CE" in
    podman)
      # Rootless + CDI (/etc/cdi/nvidia.yaml on this host).
      GPU_ARGS=(--device nvidia.com/gpu=all)
      if [[ -x /usr/bin/nvidia-smi ]]; then
        SMI_ARG=(-v /usr/bin/nvidia-smi:/usr/local/bin/nvidia-smi:ro)
      fi
      ;;
    docker)
      GPU_ARGS=(--gpus all)
      if [[ -x /usr/bin/nvidia-smi ]]; then
        SMI_ARG=(-v /usr/bin/nvidia-smi:/usr/local/bin/nvidia-smi:ro)
      fi
      ;;
  esac
fi

# Docker has no --replace; remove any old container explicitly on both engines.
REPLACE_ARG=()
[[ "$CE" == "podman" ]] && REPLACE_ARG=(--replace)
"$CE" rm -f tve >/dev/null 2>&1 || true
"$CE" run \
  -d \
  --name tve "${REPLACE_ARG[@]}" \
  "${GPU_ARGS[@]}" \
  --ipc host \
  -p "$PORT:8612" \
  -e DASIWA_NO_BROWSER=1 \
  -e TMPDIR=/tmp/rve \
  -v "$MODELS_DIR":/app/models \
  -v "$DATA_DIR":/app/data \
  -v "$CACHE_DIR":/tmp/rve \
  "${SMI_ARG[@]}" \
  "$IMAGE" >/dev/null

echo "UI:  http://127.0.0.1:${PORT}/"
echo "Stop: $CE stop tve && $CE rm tve"
