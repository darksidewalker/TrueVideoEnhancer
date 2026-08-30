# DaSiWa TrueVideoEnhancer — portable Docker/Podman image.
# GPU access is selected at runtime by scripts/run.sh; no CUDA toolkit is needed
# to build because PyTorch and TensorRT runtime libraries are installed by uv.

FROM docker.io/library/golang:1.24 AS build
WORKDIR /src
COPY go.mod ./
COPY cmd/ cmd/
COPY internal/ internal/
ARG LDFLAGS="-s -w"
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="$LDFLAGS" \
    -o /dasiwa-true-video-enhancer ./cmd/dasiwa-true-video-enhancer

FROM docker.io/library/ubuntu:24.04 AS runtime
ENV PATH="/app/runtime/venv/bin:/root/.local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG UV_VERSION=0.11.21
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        python3.12 \
        python3.12-venv \
    && curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
        | env UV_UNSAFE_INSTALLATION=1 sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt /tmp/requirements.txt
RUN python3.12 -m venv runtime/venv \
    && uv pip install --python runtime/venv/bin/python \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

COPY go.mod ./
COPY backend/ backend/
COPY --from=build /dasiwa-true-video-enhancer /app/dasiwa-true-video-enhancer
RUN mkdir -p /app/models /app/data /tmp/rve

EXPOSE 8612
ENTRYPOINT ["/app/dasiwa-true-video-enhancer"]
