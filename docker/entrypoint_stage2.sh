#!/bin/bash
# Avatar Stage 2 — Entrypoint for unified container.
# Downloads model weights (if not cached), then starts supervisord
# which manages LLM, TTS, and orchestrator processes.
set -e

# --- Environment defaults (override via Vast.ai -e flags) ---
export MODEL_REPO="${MODEL_REPO:-INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1-GGUF}"
export MODEL_FILE="${MODEL_FILE:-MamayLM-Gemma-2-9B-IT-v0.1.Q4_K_M.gguf}"
export MODEL_DIR="${MODEL_DIR:-/models}"
export LLM_HOST="${LLM_HOST:-0.0.0.0}"
export LLM_PORT="${LLM_PORT:-8001}"
export N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
export N_CTX="${N_CTX:-8192}"
export FLASH_ATTN="${FLASH_ATTN:-true}"
export LLM_BASE_URL="${LLM_BASE_URL:-http://localhost:8001}"
export TTS_BASE_URL="${TTS_BASE_URL:-http://localhost:8080}"
export HF_HUB_ENABLE_HF_TRANSFER=1

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
TTS_CHECKPOINT="/opt/fish-speech/checkpoints/openaudio-s1-mini/model.pth"
if [ ! -f "$TTS_CHECKPOINT" ]; then
    echo ""
    echo "[2/2] Downloading TTS model: openaudio-s1-mini ..."
    /opt/fish-speech/.venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'fishaudio/openaudio-s1-mini',
    local_dir='/opt/fish-speech/checkpoints/openaudio-s1-mini',
    token=os.environ.get('HF_TOKEN'),
)
print('TTS model download complete.')
"
else
    echo "[2/2] TTS model already cached at ${TTS_CHECKPOINT}"
fi

# --- 3. MuseTalk venv + models ---
export FACE_ENABLED="${FACE_ENABLED:-false}"
if [ "$FACE_ENABLED" = "true" ]; then
    set +e

    MUSETALK_DIR="/opt/musetalk"
    MUSETALK_VENV="/opt/musetalk-venv"

    if [ ! -d "$MUSETALK_DIR" ]; then
        echo ""
        echo "[3a] Cloning MuseTalk ..."
        git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git "$MUSETALK_DIR"
    else
        echo "[3a] MuseTalk already cloned at $MUSETALK_DIR"
    fi

    PIN="numpy==1.23.5 opencv-python==4.9.0.80"
    MMCV_INDEX="https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html"

    if [ ! -d "$MUSETALK_VENV" ]; then
        echo "[3b] Creating MuseTalk venv + installing deps ..."
        python3.11 -m venv "$MUSETALK_VENV"
        "$MUSETALK_VENV/bin/pip" install --upgrade pip
        "$MUSETALK_VENV/bin/pip" install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
            --index-url https://download.pytorch.org/whl/cu118
        "$MUSETALK_VENV/bin/pip" install -r "$MUSETALK_DIR/requirements.txt"
        "$MUSETALK_VENV/bin/pip" install --no-cache-dir -U openmim
        "$MUSETALK_VENV/bin/mim" install "mmengine"
        "$MUSETALK_VENV/bin/pip" install "mmcv==2.0.1" --only-binary mmcv -f "$MMCV_INDEX"
        "$MUSETALK_VENV/bin/pip" install $PIN
        "$MUSETALK_VENV/bin/mim" install "mmdet==3.1.0"
        "$MUSETALK_VENV/bin/pip" install "matplotlib==3.7.5"
        "$MUSETALK_VENV/bin/pip" install "mmpose==1.1.0" --no-build-isolation
        "$MUSETALK_VENV/bin/pip" install $PIN
        "$MUSETALK_VENV/bin/pip" install fastapi uvicorn python-multipart hf_transfer hf_xet
    else
        echo "[3b] MuseTalk venv already exists at $MUSETALK_VENV"
    fi

    echo "[3b] Verifying MuseTalk imports ..."
    MUSETALK_OK=1
    for CHECK in \
        "import numpy, cv2; print('numpy', numpy.__version__, 'cv2', cv2.__version__)" \
        "import mmcv, mmcv.ops; print('mmcv.ops OK')" \
        "from mmpose.apis import inference_topdown, init_model; print('mmpose OK')" \
        "from diffusers import AutoencoderKL; print('diffusers OK')" \
        "from transformers import WhisperModel; print('transformers OK')"
    do
        "$MUSETALK_VENV/bin/python" -c "$CHECK" 2>&1 | tail -1 || MUSETALK_OK=0
    done
    if [ "$MUSETALK_OK" != "1" ]; then
        echo "[3b] ERROR: MuseTalk environment is broken; face animation will not start."
        echo "      Audio pipeline is unaffected. See AGENTS.md 'MuseTalk dependency order'."
    fi

    if [ ! -f "$MUSETALK_DIR/models/musetalkV15/unet.pth" ]; then
        echo "[3c] Downloading MuseTalk models ..."
        mkdir -p "$MUSETALK_DIR/models/face-parse-bisent"
        "$MUSETALK_VENV/bin/python" - <<PY
from huggingface_hub import snapshot_download

root = "$MUSETALK_DIR/models"
snapshot_download("TMElyralab/MuseTalk", local_dir=root,
                  allow_patterns=["musetalkV15/unet.pth", "musetalkV15/musetalk.json"])
snapshot_download("stabilityai/sd-vae-ft-mse", local_dir=f"{root}/sd-vae",
                  allow_patterns=["config.json", "diffusion_pytorch_model.bin"])
snapshot_download("openai/whisper-tiny", local_dir=f"{root}/whisper",
                  allow_patterns=["config.json", "pytorch_model.bin", "preprocessor_config.json"])
snapshot_download("yzd-v/DWPose", local_dir=f"{root}/dwpose",
                  allow_patterns=["dw-ll_ucoco_384.pth"])
print("HuggingFace weights downloaded.")
PY
        "$MUSETALK_VENV/bin/gdown" 154JgKpzCPW82qINcVieuPH3fZ2e0P812 \
            -O "$MUSETALK_DIR/models/face-parse-bisent/79999_iter.pth"
        curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
            -o "$MUSETALK_DIR/models/face-parse-bisent/resnet18-5c106cde.pth"
        echo "[3c] MuseTalk model download complete"
    else
        echo "[3c] MuseTalk models already cached"
    fi

    cp /app/orchestrator/src/face/face_server.py "$MUSETALK_DIR/"
    cp /app/orchestrator/src/face/musetalk_*.py "$MUSETALK_DIR/"
    echo "[3d] MuseTalk setup complete"
else
    echo "[3] FACE_ENABLED=false, skipping MuseTalk setup"
fi
set -e

# --- 4. Prepare directories ---
mkdir -p /app/references /app/avatars /var/log/supervisor

# --- 4.5. Ensure libcuda.so symlink for torch.compile/Triton ---
# The runtime image has libcuda.so.1 (from NVIDIA container toolkit) but
# Triton's gcc link step needs plain libcuda.so in the linker search path.
if [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
    if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
        ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
    elif [ -e /usr/local/cuda/lib64/stubs/libcuda.so ]; then
        ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so
    fi
fi

# --- 5. Start all services via supervisord ---
echo ""
echo "Starting services via supervisord ..."
/usr/bin/supervisord -c /etc/supervisor/supervisord.conf

# --- 6. Warmup TTS (torch.compile first-request penalty ~30-60s) ---
echo ""
echo "Waiting for TTS server to be ready ..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8080/v1/health > /dev/null 2>&1; then
        echo "TTS health OK after ${i}s"
        break
    fi
    sleep 1
done

echo "Sending TTS warmup request (torch.compile will trace on first call) ..."
WARMUP_START=$(date +%s)
curl -sf -X POST http://localhost:8080/v1/tts \
    -H 'Content-Type: application/json' \
    -d '{"text": "Hello world.", "streaming": false}' \
    -o /dev/null || echo "Warmup request failed (non-fatal)"
WARMUP_END=$(date +%s)
echo "TTS warmup completed in $((WARMUP_END - WARMUP_START))s"

# --- 7. Warmup MuseTalk (if enabled) ---
if [ "$FACE_ENABLED" = "true" ]; then
    echo ""
    echo "Waiting for MuseTalk server to be ready ..."
    for i in $(seq 1 90); do
        if curl -sf http://localhost:8002/health > /dev/null 2>&1; then
            echo "MuseTalk health OK after ${i}s"
            break
        fi
        sleep 1
    done
fi

# Keep container running — attach to supervisord
echo "All services ready. Entering supervisord loop."
exec tail -f /var/log/supervisor/*.log
