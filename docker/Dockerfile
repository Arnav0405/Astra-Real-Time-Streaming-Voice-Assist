# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: the Go binary (cgo) plus the ONNX Runtime shared library it dlopens.
# ---------------------------------------------------------------------------
FROM golang:1.26-bookworm AS builder

# ONNX Runtime has no distro package, so take Microsoft's release tarball.
# Pinned to the version the Python<->Go golden parity tests were run against on
# the host (brew's 1.29.0) — the container should not silently run a different
# inference engine than the one the numbers in the README came from.
ARG ORT_VERSION=1.29.0
ARG TARGETARCH
RUN set -eux; \
    case "$TARGETARCH" in \
      amd64) ort_arch=x64 ;; \
      arm64) ort_arch=aarch64 ;; \
      *) echo "unsupported TARGETARCH: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    curl -fsSL --retry 5 --retry-all-errors --retry-delay 2 -o /tmp/ort.tgz \
      "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${ort_arch}-${ORT_VERSION}.tgz"; \
    mkdir -p /out/ort; \
    tar -xzf /tmp/ort.tgz -C /out/ort --strip-components=1; \
    rm /tmp/ort.tgz

WORKDIR /src/services/backend
COPY services/backend/go.mod services/backend/go.sum ./
RUN go mod download
COPY services/backend/ ./
# yalue/onnxruntime_go ships onnxruntime_wrapper.c, so cgo must stay on.
RUN CGO_ENABLED=1 go build -o /out/astra ./cmd/astra

# ---------------------------------------------------------------------------
# Runtime. glibc, because ORT's release binary is glibc-linked — musl/Alpine
# would mean building onnxruntime from source.
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim

# ca-certificates: TLS to the ASR/LLM/TTS provider. libstdc++6: ORT's C++ runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

# internal/vad.DefaultLibPaths already probes this exact path, so -ort-lib
# never has to be passed.
COPY --from=builder /out/ort/lib/libonnxruntime*.so* /usr/local/lib/

WORKDIR /app
COPY --from=builder /out/astra /app/astra
COPY assets/models /app/assets/models
COPY assets/configs /app/assets/configs
COPY clients/web /app/web
# clients/web/astra.proto is a symlink up into proto/, which would land in the
# image dangling — and app.js fetches it at runtime. Copy the real file over it.
COPY proto/astra/v1/astra.proto /app/web/astra.proto

# Nothing is written at runtime, so the server has no reason to be root.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin astra
USER astra

EXPOSE 8080

# The paths live here rather than in the Go flag defaults, so `go run
# ./cmd/astra` from services/backend keeps working unchanged on the host.
# Exec form + empty CMD means `docker run astra -verbose` appends cleanly.
# -ww-config is omitted on purpose: main.go derives it from the model path.
ENTRYPOINT ["/app/astra", \
  "-addr", ":8080", \
  "-vad-model", "/app/assets/models/vad/vad_v1.onnx", \
  "-vad-config", "/app/assets/models/vad/vad_v1.json", \
  "-ww-model", "/app/assets/models/wakeword/ww_v2.onnx", \
  "-endpoint-config", "/app/assets/configs/endpoint.json", \
  "-asr-config", "/app/assets/configs/asr.json", \
  "-llm-config", "/app/assets/configs/llm.json", \
  "-tts-config", "/app/assets/configs/tts.json", \
  "-env-file", "/app/.env", \
  "-web-dir", "/app/web"]
CMD []
