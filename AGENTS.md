# Avatar — Digital Clone for Video Conferencing

## Project Overview

Real-time digital avatar clone: captures the user's voice via microphone, generates conversational responses via LLM, synthesizes speech in the user's cloned voice, animates the user's face, and pipes everything into Zoom/Teams as a virtual webcam + microphone. All open-source, no paid API services.

## Architecture

- **Local machine**: Windows, RTX 3060 6GB laptop. Runs orchestrator, ASR (faster-whisper), VAD (Silero), virtual camera/mic output.
- **Remote GPU**: Vast.ai, RTX 4090 24GB (Linux). Runs LLM, TTS, face animation. Docker-based deployment.
- **Communication**: WebSocket (binary/msgpack) between local client and Vast.ai server.

## Tech Stack & Tooling

| What | Tool |
|------|------|
| Package manager | **uv** (NOT pip, NOT conda) |
| Python version | 3.11 |
| LLM | MamayLM (Gemma 2 9B, 4-bit) via vLLM |
| TTS / Voice clone | OpenAudio S1-mini (Fish Speech successor, 0.5B) |
| ASR | faster-whisper large-v3-turbo + Silero VAD |
| Face animation | MuseTalk |
| Server framework | FastAPI + WebSocket |
| Local orchestrator | Python asyncio + websockets |
| Virtual camera | pyvirtualcam + OBS Virtual Camera |
| Virtual microphone | VB-Audio Virtual Cable + sounddevice |
| Containerization | Docker (python:3.11-slim-bookworm base) |
| GPU hosting | Vast.ai (CLI: `uv tool install vastai`) |
| Vast.ai CLI docs | https://docs.vast.ai/api-reference/commands |

## Project Structure

```
avatar/
├── AGENTS.md                 # This file — project memory
├── .github/
│   ├── copilot-instructions.md   # Coding conventions (always-on)
│   ├── prompts/
│   │   └── plan-digitalAvatarClone.prompt.md  # Master build plan
│   └── instructions/         # File-pattern-based rules (future)
├── server/                   # Vast.ai server code
│   ├── pyproject.toml        # uv-managed dependencies
│   ├── src/
│   │   ├── llm/              # vLLM wrapper for MamayLM
│   │   ├── tts/              # Fish Speech wrapper
│   │   ├── face/             # MuseTalk wrapper
│   │   └── api/              # FastAPI + WebSocket endpoints
│   └── Dockerfile
├── client/                   # Local Windows orchestrator
│   ├── pyproject.toml        # uv-managed dependencies
│   ├── src/
│   │   ├── asr/              # faster-whisper + Silero VAD
│   │   ├── audio/            # sounddevice capture/playback
│   │   ├── video/            # pyvirtualcam output
│   │   ├── orchestrator.py   # Main pipeline coordinator
│   │   └── tts_test.py       # Fish Speech TTS test client
├── scripts/
│   ├── record_voice.py       # Interactive voice recording for cloning
│   └── deploy_tts.py         # Deploy Fish Speech to Vast.ai
├── docker/                   # Dockerfiles, deployment scripts, vast.ai helpers
└── models/                   # Model configs and download scripts (gitignored)
```

## Vast.ai Deployment

- Install CLI: `uv tool install vastai`
- Set API key: `vastai set api-key <key>`
- Register SSH key: `vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"`
- Search for GPU: `vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.95' -o 'dph+'`
- SSH: `ssh -p <port> root@<ssh_host>` (get from `vastai ssh-url <id>`)
- Target: RTX 4090 24GB, EU datacenter, on-demand (not interruptible)

### Stage 0 (avatar-server — WebSocket echo)
- Image: `sam0ed/avatar-server:latest` (python:3.11-slim-bookworm, 1.07GB)
- Create: `vastai create instance <id> --image sam0ed/avatar-server:latest --disk 20 --direct --env '-p 8000:8000' --onstart-cmd 'uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000'`
- Verified: Denmark instance, $0.27/hr, avg RTT 48ms

### Stage 1 (Fish Speech TTS)
- Image: `fishaudio/fish-speech:server-cuda` (nvidia/cuda:12.6-runtime, ~4.9GB)
- Deploy: `HF_TOKEN=hf_xxx uv run scripts/deploy_tts.py`
- Model: `fishaudio/openaudio-s1-mini` (gated — requires HF token + license acceptance)
- Weights downloaded at startup via onstart-cmd (~3.6GB: model.pth 1.74GB + codec.pth 1.87GB)
- Server port: 8080 inside container, mapped to random high port by Vast.ai
- API: POST `/v1/tts` (msgpack body), GET `/v1/health`, POST `/v1/references/add`, GET `/v1/references/list`
- GPU memory: ~4.9GB after warmup
- Current instance: ID 31466745, Netherlands, $0.30/hr, IP 38.117.87.41, port 8080→46682

## Current Stage

**Stage 1: Voice Clone Proof-of-Concept** — IN PROGRESS

Fish Speech (OpenAudio S1-mini) deployed and verified on Vast.ai. TTS synthesis working (default voice). Voice recording script and TTS test client created.
Remaining: Record user voice samples, test zero-shot voice cloning, evaluate quality.

## Progress Log

| Date | Stage | What was done |
|------|-------|---------------|
| 2026-02-15 | Planning | Created master build plan, selected tech stack, established project structure |
| 2026-02-15 | Stage 0 | Infrastructure setup: .gitignore, directory scaffold, server/client pyproject.toml (uv), FastAPI+WebSocket server, client connectivity test + orchestrator skeleton, Dockerfile (python:3.11-slim-bookworm, 1.07GB), deployment scripts. Local ping/echo verified (avg RTT <1ms). Deployed to Vast.ai (RTX 4090, Denmark). Remote ping/echo verified (avg RTT 48ms, within <100ms target). |
| 2026-02-15 | Stage 1 | TTS model evaluation: compared S1-mini, Orpheus 3B, XTTS v2, Kokoro, MaskGCT, StyleTTS2-Ukrainian, Piper — selected OpenAudio S1-mini. Created voice recording script (`scripts/record_voice.py`), TTS test client (`client/src/tts_test.py`), deployment script (`scripts/deploy_tts.py`). Deployed Fish Speech S1-mini on Vast.ai (RTX 4090, Netherlands, $0.30/hr). TTS synthesis verified: 303KB WAV in ~8s for 52 chars, 4.9GB GPU memory. Default voice works; voice cloning pending user voice samples. |

## Important Decisions & Context

- **uv everywhere**: Use `uv` for all Python dependency management. `pyproject.toml` for both server/ and client/. No requirements.txt, no pip freeze.
- **No WSL2**: Everything local runs on native Windows. All dependencies have Windows CUDA wheels.
- **MamayLM over Qwen/Llama**: Chosen for SOTA Ukrainian (outperforms 10x larger models) + strong English from Gemma 2 base.
- **OpenAudio S1-mini over alternatives**: Successor to Fish Speech v1.5. #1 on TTS-Arena2, 0.5B params (~2-3GB VRAM), zero-shot voice cloning, streaming (RTF ~1:7 on 4090), Apache-2.0 code + CC-BY-NC-SA-4.0 weights (fine for personal use). Evaluated Orpheus 3B, XTTS v2, Kokoro, MaskGCT, StyleTTS2-Ukrainian, Piper — S1-mini is the best fit for quality, VRAM, and streaming.
- **ASR runs locally**: faster-whisper on RTX 3060 (2GB VRAM) to save ~100ms network round trip.
- **Single Vast.ai GPU**: All server models (LLM + TTS + MuseTalk) coexist on one RTX 4090 (~16-19GB total VRAM).
- **Latency budget**: 1.5-2.5s end-to-end from end of speech to first avatar response is acceptable.
- **Docker base image**: Switched from `nvidia/cuda:12.1.1-devel` (~12GB) to `python:3.11-slim-bookworm` (1.07GB). Neither Fish Speech nor MuseTalk compile CUDA code — pre-compiled PyTorch wheels, mmcv ships pre-built wheels.
- **S1-mini over Fish Speech v1.5**: v1.5 codebase is being deprecated. S1-mini uses a new DAC codec (replaces VQGAN), different API flags (`--decoder-checkpoint-path` + `--decoder-config-name modded_dac_vq`), and weights at `fishaudio/openaudio-s1-mini`. Checkpoint: model.pth (1.74GB) + codec.pth (1.87GB). 12GB VRAM recommended officially but actual usage ~4.9GB after warmup.
- **Fish Speech Docker**: Use `fishaudio/fish-speech:server-cuda` tag (not `latest-server-cuda`). The `fish` user (UID 1000) defined in the Dockerfile does not exist in Vast.ai runtime — run `start_server.sh` as root directly (not via `su fish`). Model weights are gated on HuggingFace — requires HF token.
- **HF_TOKEN required**: The `fishaudio/openaudio-s1-mini` model is gated. Must accept license at https://huggingface.co/fishaudio/openaudio-s1-mini and provide token via `HF_TOKEN` env var or `.env` file.
- **Vast.ai SSH**: Use `vastai create ssh-key` (not `set ssh-key`). Key propagation to running instances may take a moment.

## Known Risks

- Face animation quality: MuseTalk is the only real-time option; quality is acceptable at Zoom compression but has artifacts
- Vast.ai instances can be preempted: use on-demand, have Docker image pre-built
- Voice clone may need fine-tuning beyond zero-shot for convincing results
- Audio/video sync requires timestamp-based jitter buffering
- Fish Speech first-request latency: ~8-10s for short text (includes model warmup); subsequent requests faster
- Gated model access: HF token must be kept in `.env` (gitignored), never committed

## Notes for Agent

- Always check this file at the start of a session to understand project state
- Update the Progress Log after completing any stage or significant milestone
- Update Current Stage when transitioning between stages
- When writing Python code: use type hints, async/await for I/O, and uv for dependencies
- The plan file at `.github/prompts/plan-digitalAvatarClone.prompt.md` has the detailed stage-by-stage breakdown
