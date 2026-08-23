#!/bin/bash
# Dispatch the TTS server process by TTS_ENGINE; always binds :8080.
set -e

case "${TTS_ENGINE:-fish}" in
  fish)
    cd /opt/fish-speech
    exec .venv/bin/python tools/api_server.py --listen 0.0.0.0:8080 --compile
    ;;
  higgs)
    # Memory is governed by per-stage total_gpu_memory_fraction in the YAML
    # (defaults are 0.03/0.85/0.10 and would OOM beside the LLM), not by
    # --mem-fraction-static, which only feeds the SGLang server args.
    exec /opt/higgs-venv/bin/sgl-omni serve \
        --config /app/higgs_4090.yaml \
        --allowed-local-media-path /app/references \
        --max-running-requests 2 \
        --cuda-graph-max-bs 2 \
        --max-total-tokens 8192 \
        --port 8080
    ;;
  *)
    echo "Unknown TTS_ENGINE '${TTS_ENGINE}' (known: fish, higgs)" >&2
    exit 1
    ;;
esac
