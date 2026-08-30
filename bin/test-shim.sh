#!/usr/bin/env bash
# Self-check for bin/podman's arg rewrite. Run: ./bin/test-shim.sh
set -euo pipefail
cd "$(dirname "$0")"
export SHIM_DRY_RUN=1
fail=0
check() { # check <expected> <args...>
  local want="$1"; shift
  local got; got=$(./podman "$@")
  [ "$got" = "$want" ] || { echo "FAIL: $*"; echo "  want: $want"; echo "  got : $got"; fail=1; }
}

check "docker image inspect dasiwa/tve:latest" image exists dasiwa/tve:latest
check "docker run -d --name tve --gpus all --ipc host -v /m/models:/app/models -v /m/cache:/tmp/rve img" \
  run -d --name tve --replace --device nvidia.com/gpu=all --ipc host -v /m/models:/app/models -v /m/cache:/tmp/rve img
check "docker run --gpus all img" run --device=nvidia.com/gpu=all img
check "docker run -v /dev/kfd img" run -v /dev/kfd img
check "docker ps -a --filter name=tve --format {{.Names}}" ps -a --filter name=tve --format '{{.Names}}'
[ "$(./podman-compose -f compose.yaml up -d)" = "docker compose -f compose.yaml up -d" ] \
  || { echo "FAIL: podman-compose"; fail=1; }

[ $fail = 0 ] && echo "ok"
exit $fail
