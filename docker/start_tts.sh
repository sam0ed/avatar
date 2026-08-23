#!/bin/bash
# Dispatch the TTS server process by TTS_ENGINE; always binds :8080.
set -e

case "${TTS_ENGINE:-fish}" in
  fish)
    cd /opt/fish-speech
    exec .venv/bin/python tools/api_server.py --listen 0.0.0.0:8080 --compile
    ;;
  higgs)
    TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    FRACTION=$(awk -v gib="${HIGGS_VRAM_GIB:-13}" -v total="$TOTAL_MIB" \
        'BEGIN { printf "%.3f", (gib * 1024) / total }')
    echo "higgs: mem-fraction ${FRACTION} (${HIGGS_VRAM_GIB:-13} GiB of ${TOTAL_MIB} MiB)"
    exec /opt/higgs-venv/bin/sgl-omni serve \
        --model-path "${HIGGS_MODEL:-bosonai/higgs-tts-3-4b}" \
        --allowed-local-media-path /app/references \
        --mem-fraction-static "$FRACTION" \
        --port 8080
    ;;
  *)
    echo "Unknown TTS_ENGINE '${TTS_ENGINE}' (known: fish, higgs)" >&2
    exit 1
    ;;
esac
