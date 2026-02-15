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
| TTS / Voice clone | Fish Speech v1.5 |
| ASR | faster-whisper large-v3-turbo + Silero VAD |
| Face animation | MuseTalk |
| Server framework | FastAPI + WebSocket |
| Local orchestrator | Python asyncio + websockets |
| Virtual camera | pyvirtualcam + OBS Virtual Camera |
| Virtual microphone | VB-Audio Virtual Cable + sounddevice |
| Containerization | Docker (nvidia/cuda:12.1 base) |
| GPU hosting | Vast.ai (CLI: `pip install vastai`) |
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
│   │   └── orchestrator.py   # Main pipeline coordinator
├── scripts/                  # Voice recording, data prep, fine-tuning
├── docker/                   # Dockerfiles, deployment scripts, vast.ai helpers
└── models/                   # Model configs and download scripts (gitignored)
```

## Vast.ai Deployment

- Install CLI: `uv pip install vastai` or `pip install vastai`
- Set API key: `vastai set api-key <key>`
- Search for GPU: `vastai search offers 'gpu_name=RTX_4090 num_gpus=1 reliability>0.98' -o 'dph+'`
- Create instance: `vastai create instance <id> --image <dockerhub-image> --disk 50`
- SSH: `ssh $(vastai ssh-url <instance_id>)`
- Target: RTX 4090 24GB, EU datacenter, on-demand (not interruptible)

## Current Stage

**Stage 0: Infrastructure Setup** — COMPLETE

All infrastructure scaffolding is in place. Server starts, WebSocket ping/echo verified locally.
Next: Stage 1 (Voice Clone Proof-of-Concept) or deploy to Vast.ai to verify remote connectivity.

## Progress Log

| Date | Stage | What was done |
|------|-------|---------------|
| 2026-02-15 | Planning | Created master build plan, selected tech stack, established project structure |
| 2026-02-15 | Stage 0 | Infrastructure setup: .gitignore, directory scaffold, server/client pyproject.toml (uv), FastAPI+WebSocket server, client connectivity test + orchestrator skeleton, Dockerfile, deployment scripts. Local ping/echo verified (avg RTT <1ms). |

## Important Decisions & Context

- **uv everywhere**: Use `uv` for all Python dependency management. `pyproject.toml` for both server/ and client/. No requirements.txt, no pip freeze.
- **No WSL2**: Everything local runs on native Windows. All dependencies have Windows CUDA wheels.
- **MamayLM over Qwen/Llama**: Chosen for SOTA Ukrainian (outperforms 10x larger models) + strong English from Gemma 2 base.
- **Fish Speech over XTTS/Kokoro**: Best multilingual (EN + Ukrainian) with streaming and zero-shot cloning.
- **ASR runs locally**: faster-whisper on RTX 3060 (2GB VRAM) to save ~100ms network round trip.
- **Single Vast.ai GPU**: All server models (LLM + TTS + MuseTalk) coexist on one RTX 4090 (~16-19GB total VRAM).
- **Latency budget**: 1.5-2.5s end-to-end from end of speech to first avatar response is acceptable.

## Known Risks

- Face animation quality: MuseTalk is the only real-time option; quality is acceptable at Zoom compression but has artifacts
- Vast.ai instances can be preempted: use on-demand, have Docker image pre-built
- Voice clone may need fine-tuning beyond zero-shot for convincing results
- Audio/video sync requires timestamp-based jitter buffering

## Notes for Agent

- Always check this file at the start of a session to understand project state
- Update the Progress Log after completing any stage or significant milestone
- Update Current Stage when transitioning between stages
- When writing Python code: use type hints, async/await for I/O, and uv for dependencies
- The plan file at `.github/prompts/plan-digitalAvatarClone.prompt.md` has the detailed stage-by-stage breakdown
