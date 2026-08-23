#!/bin/bash
# Dispatch the TTS server process by TTS_ENGINE; always binds :8080.
set -e

case "${TTS_ENGINE:-fish}" in
  fish)
    cd /opt/fish-speech
    exec .venv/bin/python tools/api_server.py --listen 0.0.0.0:8080 --compile
    ;;
  higgs)
    # Per-stage memory uses sglang-omni defaults: total_gpu_memory_fraction is
    # a fraction of FREE VRAM at stage load, so the pipeline adapts to whatever
    # the co-located LLM already holds (verified on 2x4090: LLM 9.1GB + Higgs
    # 12.4GB = 21.5/24GB; defaults and hand-tuned fractions land identically).
    # tvm_ffi JIT-compiles a few SGLang kernels at first boot: it resolves
    # `ninja` via PATH and nvcc via CUDA_HOME (default /usr/local/cuda has no
    # nvcc in the runtime base image). Point both at the venv's cu13 wheels.
    export PATH=/opt/higgs-venv/bin:$PATH
    export CUDA_HOME=/opt/higgs-venv/lib/python3.12/site-packages/nvidia/cu13
    export CUDA_PATH=$CUDA_HOME
    export PATH=$CUDA_HOME/bin:$PATH
    ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
    ln -sfn lib "$CUDA_HOME/lib64"
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
