#!/bin/bash
# Avatar Stage 2 — Entrypoint for unified container.
# Downloads model weights (if not cached), then starts supervisord
# which manages LLM, TTS, and orchestrator processes.
set -e

# --- Environment defaults (override via Vast.ai -e flags) ---
export MODEL_REPO="${MODEL_REPO:-INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1-GGUF}"
export MODEL_FILE="${MODEL_FILE:-MamayLM-Gemma-2-9B-IT-v0.1-Q4_K_M.gguf}"
export MODEL_DIR="${MODEL_DIR:-/models}"
export LLM_HOST="${LLM_HOST:-0.0.0.0}"
export LLM_PORT="${LLM_PORT:-8001}"
export N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
export N_CTX="${N_CTX:-8192}"
export FLASH_ATTN="${FLASH_ATTN:-true}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8001}"
export TTS_BASE_URL="${TTS_BASE_URL:-http://localhost:8080}"

echo "============================================="
echo "  Avatar Stage 2 — Unified Container"
echo "============================================="
echo "  LLM:          localhost:${LLM_PORT:-8001}"
echo "  TTS:          localhost:8080"
echo "  Orchestrator: localhost:8000"
echo "============================================="

# --- 1. Download LLM model ---
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "[1/2] Downloading LLM model: ${MODEL_REPO} / ${MODEL_FILE} ..."
    mkdir -p "$MODEL_DIR"
    /opt/llm-venv/bin/python -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id=os.environ['MODEL_REPO'],
    filename=os.environ['MODEL_FILE'],
    local_dir=os.environ['MODEL_DIR'],
)
print('LLM model download complete.')
"
else
    echo "[1/2] LLM model already cached at ${MODEL_PATH}"
fi

# --- 2. Download TTS model ---
TTS_CHECKPOINT="/app/checkpoints/openaudio-s1-mini/model.pth"
if [ ! -f "$TTS_CHECKPOINT" ]; then
    echo ""
    echo "[2/2] Downloading TTS model: openaudio-s1-mini ..."
    cd /app
    uv run python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'fishaudio/openaudio-s1-mini',
    local_dir='checkpoints/openaudio-s1-mini',
    token=os.environ.get('HF_TOKEN'),
)
print('TTS model download complete.')
"
else
    echo "[2/2] TTS model already cached at ${TTS_CHECKPOINT}"
fi

# --- 3. Prepare directories ---
mkdir -p /app/references /var/log/supervisor

# --- 4. Start all services via supervisord ---
echo ""
echo "Starting services via supervisord ..."
exec /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf
