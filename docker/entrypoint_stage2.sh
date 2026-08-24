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
export HF_HUB_ENABLE_HF_TRANSFER=1
export TTS_ENGINE="${TTS_ENGINE:-fish}"
export HIGGS_VRAM_GIB="${HIGGS_VRAM_GIB:-13}"
export HIGGS_MODEL="${HIGGS_MODEL:-bosonai/higgs-tts-3-4b}"
export HIGGS_REVISION="${HIGGS_REVISION:-7556c17e05201fccd9c8cc120bc216dcc7b5d561}"

# GPU placement: services talk over localhost HTTP only, so placement is
# free to vary per host. One GPU: everything shares it. Two: MuseTalk gets
# its own (TTS<->MuseTalk contention is the measured bottleneck). Three+:
# one each.
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -ge 3 ]; then
    export LLM_GPU="${LLM_GPU:-0}" TTS_GPU="${TTS_GPU:-1}" FACE_GPU="${FACE_GPU:-2}"
elif [ "$GPU_COUNT" -eq 2 ]; then
    export LLM_GPU="${LLM_GPU:-0}" TTS_GPU="${TTS_GPU:-0}" FACE_GPU="${FACE_GPU:-1}"
else
    export LLM_GPU="${LLM_GPU:-0}" TTS_GPU="${TTS_GPU:-0}" FACE_GPU="${FACE_GPU:-0}"
fi
echo "GPU placement: count=${GPU_COUNT} llm=${LLM_GPU} tts=${TTS_GPU} face=${FACE_GPU}"

# VRAM preflight: sum known footprints per assigned GPU against the hardware,
# so an impossible configuration refuses at boot instead of OOMing mid-turn.
LLM_GIB=10
MUSETALK_GIB=8
if [ "$TTS_ENGINE" = "higgs" ]; then TTS_GIB="$HIGGS_VRAM_GIB"; else TTS_GIB=6; fi
for GPU_IDX in $(seq 0 $((GPU_COUNT - 1))); do
    NEED=0
    [ "$LLM_GPU" = "$GPU_IDX" ] && NEED=$((NEED + LLM_GIB))
    [ "$TTS_GPU" = "$GPU_IDX" ] && NEED=$((NEED + TTS_GIB))
    if [ "${FACE_ENABLED:-false}" = "true" ] && [ "$FACE_GPU" = "$GPU_IDX" ]; then
        NEED=$((NEED + MUSETALK_GIB))
    fi
    HAVE_MIB=$(nvidia-smi -i "$GPU_IDX" --query-gpu=memory.total --format=csv,noheader,nounits)
    HAVE=$((HAVE_MIB / 1024))
    if [ "$NEED" -gt "$HAVE" ]; then
        echo "FATAL: GPU $GPU_IDX needs ~${NEED} GiB (llm=${LLM_GIB}, tts[$TTS_ENGINE]=${TTS_GIB}, face=${MUSETALK_GIB}) but has ${HAVE} GiB."
        echo "       Use more/larger GPUs, or override LLM_GPU/TTS_GPU/FACE_GPU/HIGGS_VRAM_GIB."
        exit 1
    fi
done

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

# --- 2. Download TTS model (only the selected engine's weights) ---
if [ "$TTS_ENGINE" = "higgs" ]; then
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    DRV_MAJOR=${DRV%%.*}
    if [ "$DRV_MAJOR" -lt 580 ]; then
        echo "FATAL: driver ${DRV} < 580.65 required by the higgs stack (torch 2.11 / CUDA 13)."
        exit 1
    fi
    echo ""
    echo "[2/2] Downloading TTS model: ${HIGGS_MODEL} (expecting ${HIGGS_REVISION}) ..."
    # Unpinned so refs/main is written (the server resolves main at start);
    # the expected revision is asserted instead of forced.
    /opt/higgs-venv/bin/python -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(os.environ['HIGGS_MODEL'], token=os.environ.get('HF_TOKEN'))
expected = os.environ['HIGGS_REVISION']
assert expected[:12] in path, f'weights drifted: {path} != {expected}'
print('Higgs model download complete:', path)
"
    /opt/higgs-venv/bin/sgl-omni check-gpu --json --strict \
        || echo "check-gpu reported warnings (see above) — continuing"
else
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
    cp /app/orchestrator/src/face/frame_cycle.py "$MUSETALK_DIR/"
    echo "[3d] MuseTalk setup complete"
else
    echo "[3] FACE_ENABLED=false, skipping MuseTalk setup"
fi
set -e

# --- 4. Prepare directories ---
mkdir -p /app/references /app/avatars /var/log/supervisor

# --- 4.5. Repair the driver loader path ---
# Some Vast hosts mount only the versioned libcuda without the .so.1 alias;
# the loader then picks the image's CUDA forward-compat lib, which fails on
# GeForce with Error 804. Alias the real driver, disable compat, never stubs.
LIBDIR=/usr/lib/x86_64-linux-gnu
if [ ! -e "$LIBDIR/libcuda.so.1" ]; then
    VERSIONED=$(ls "$LIBDIR"/libcuda.so.* 2>/dev/null | grep -v '\.so\.1$' | head -1)
    if [ -n "$VERSIONED" ]; then
        ln -sf "$(basename "$VERSIONED")" "$LIBDIR/libcuda.so.1"
    fi
fi
if [ -e "$LIBDIR/libcuda.so.1" ] && [ ! -e "$LIBDIR/libcuda.so" ]; then
    ln -sf libcuda.so.1 "$LIBDIR/libcuda.so"
fi
if [ -d /usr/local/cuda/compat ]; then
    mv /usr/local/cuda/compat /usr/local/cuda/compat.disabled
fi
ldconfig

# --- 5. Start all services via supervisord ---
echo ""
echo "Starting services via supervisord ..."
/usr/bin/supervisord -c /etc/supervisor/supervisord.conf

# --- 6. Warmup TTS (torch.compile first-request penalty ~30-60s) ---
echo ""
if [ "$TTS_ENGINE" = "higgs" ]; then
    TTS_HEALTH_PATH="/health"
    TTS_WAIT_S=300
else
    TTS_HEALTH_PATH="/v1/health"
    TTS_WAIT_S=60
fi
echo "Waiting for TTS server ($TTS_ENGINE) to be ready ..."
for i in $(seq 1 "$TTS_WAIT_S"); do
    if curl -sf "http://localhost:8080${TTS_HEALTH_PATH}" > /dev/null 2>&1; then
        echo "TTS health OK after ${i}s"
        break
    fi
    sleep 1
done

echo "Sending TTS warmup request ..."
WARMUP_START=$(date +%s)
if [ "$TTS_ENGINE" = "higgs" ]; then
    curl -sf -X POST http://localhost:8080/v1/audio/speech \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"${HIGGS_MODEL}\", \"input\": \"Hello world.\", \"voice\": \"default\", \"response_format\": \"pcm\", \"stream\": false}" \
        -o /dev/null || echo "Warmup request failed (non-fatal)"
else
    curl -sf -X POST http://localhost:8080/v1/tts \
        -H 'Content-Type: application/json' \
        -d '{"text": "Hello world.", "streaming": false}' \
        -o /dev/null || echo "Warmup request failed (non-fatal)"
fi
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
