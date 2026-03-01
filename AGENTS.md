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
| LLM | MamayLM (Gemma 2 9B, GGUF Q4_K_M) via llama-cpp-python |
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
│   │   ├── llm/
│   │   │   ├── client.py         # Async LLM client (OpenAI-compat SSE streaming)
│   │   │   ├── chunker.py        # Sentence boundary detection for streaming TTS
│   │   │   └── system_prompt.txt  # System prompt (placeholder)
│   │   ├── tts/
│   │   │   └── client.py         # Async Fish Speech TTS client
│   │   ├── face/             # MuseTalk wrapper (future)
│   │   └── api/
│   │       └── server.py     # FastAPI + WebSocket (chat/ping/echo)
│   └── Dockerfile
├── client/                   # Local Windows orchestrator
│   ├── pyproject.toml        # uv-managed dependencies
│   ├── src/
│   │   ├── asr/
│   │   │   ├── transcriber.py     # Moonshine Voice ASR wrapper (VAD+ASR, CPU)
│   │   │   └── smart_turn.py      # Smart Turn v3.2 ONNX end-of-turn detector
│   │   ├── audio/
│   │   │   └── playback.py        # Audio player with cancel support
│   │   ├── video/            # pyvirtualcam output (future)
│   │   ├── orchestrator.py   # Main pipeline coordinator
│   │   ├── voice_client.py   # Voice conversation client (mic → ASR → LLM → TTS → speaker)
│   │   ├── chat_client.py    # Terminal chat client with audio playback
│   │   └── tts_test.py       # Fish Speech TTS test client
├── scripts/
│   ├── record_voice.py       # Interactive voice recording for cloning
│   ├── deploy_tts.py         # Deploy Fish Speech to Vast.ai
│   └── deploy_stage2.py      # Deploy Stage 2 single container to Vast.ai
├── docker/                   # Dockerfiles, deployment scripts, vast.ai helpers
│   ├── Dockerfile            # Stage 0 orchestrator container (slim)
│   ├── supervisord.conf      # Process manager config for Stage 2 (3 services)
│   └── entrypoint_stage2.sh  # Model download + supervisord start
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

### Stage 2 (LLM + TTS Conversational Pipeline)
- Image: `fishaudio/fish-speech:server-cuda` (same base as Stage 1 — no custom image needed)
- Deploy: `HF_TOKEN=hf_xxx uv run scripts/deploy_stage2.py` (fire-and-forget, same pattern as Stage 1)
- How it works: deploy script sends a short `--onstart-cmd` (~700 chars) that `git clone`s the public repo,
  installs supervisor + llama-cpp-python, sets up orchestrator, downloads models, starts supervisord.
  No custom Docker image build/push needed. Repo must be public on GitHub.
- Architecture: Single container with supervisord managing 3 processes:
  - LLM (llama-cpp-python, port 8001) — isolated venv at `/opt/llm-venv`
  - TTS (Fish Speech S1-mini, port 8080) — from base image, `start_server.sh`
  - Orchestrator (FastAPI, port 8000) — own uv project at `/app/orchestrator`
- Models downloaded at first boot by `entrypoint_stage2.sh` (~5.76GB LLM + ~3.6GB TTS)
- LLM API: OpenAI-compatible at port 8001 (`/v1/chat/completions` with SSE streaming)
- Pipeline: WebSocket `chat` msg → stream LLM tokens → sentence chunking → parallel TTS synthesis → audio back to client
- Voice references: Upload via orchestrator's `POST /voice/reference` endpoint (writes WAV+lab directly to `/app/references/<ref_id>/`, supports multiple files). Enable via `POST /voice/enable?ref_id=<id>`. Script: `uv run scripts/setup_voice.py --url http://localhost:8000`
- Client: `client/src/chat_client.py` — multi-turn terminal chat with real-time token display + audio playback
- Logs: `ssh -p <port> root@<host> 'supervisorctl status'` or `'tail -f /var/log/supervisor/*.log'`

## Current Stage

**Stage 3: Speech-to-Speech Conversation** — IN PROGRESS

Core voice loop implemented. Moonshine Voice ASR + barge-in with pre-filters + Smart Turn v3.2 end-of-turn detection. New files: `client/src/asr/transcriber.py` (SpeechTranscriber with speech duration tracking + audio ring buffer), `client/src/asr/smart_turn.py` (SmartTurnAnalyzer — ONNX inference), `client/src/audio/playback.py` (AudioPlayer with cancel), `client/src/voice_client.py` (VoiceClient — full voice conversation loop with barge-in pre-filters + Smart Turn). Server updated: chat runs as background asyncio task, `chat_cancel` message type for barge-in. Pending: end-to-end testing with Vast.ai server.



## Progress Log

| Date | Stage | What was done |
|------|-------|---------------|
| 2026-02-15 | Planning | Created master build plan, selected tech stack, established project structure |
| 2026-02-15 | Stage 0 | Infrastructure setup: .gitignore, directory scaffold, server/client pyproject.toml (uv), FastAPI+WebSocket server, client connectivity test + orchestrator skeleton, Dockerfile (python:3.11-slim-bookworm, 1.07GB), deployment scripts. Local ping/echo verified (avg RTT <1ms). Deployed to Vast.ai (RTX 4090, Denmark). Remote ping/echo verified (avg RTT 48ms, within <100ms target). |
| 2026-02-15 | Stage 1 | TTS model evaluation: compared S1-mini, Orpheus 3B, XTTS v2, Kokoro, MaskGCT, StyleTTS2-Ukrainian, Piper — selected OpenAudio S1-mini. Created voice recording script (`scripts/record_voice.py`), TTS test client (`client/src/tts_test.py`), deployment script (`scripts/deploy_tts.py`). Deployed Fish Speech S1-mini on Vast.ai (RTX 4090, Netherlands, $0.30/hr). TTS synthesis verified: 303KB WAV in ~8s for 52 chars, 4.9GB GPU memory. Default voice works; voice cloning pending user voice samples. |
| 2026-02-21 | Stage 2 | Implementation: supervisord config + entrypoint for unified container. Server-side: async LLM client with SSE streaming (`server/src/llm/client.py`), async TTS client (`server/src/tts/client.py`), sentence chunker (`server/src/llm/chunker.py`), chat WebSocket handler in `server/src/api/server.py`. Client-side: multi-turn terminal chat client with audio playback (`client/src/chat_client.py`). Placeholder system prompt created. Widened `server/pyproject.toml` Python constraint to `>=3.11` for Fish Speech image compatibility. |
| 2026-02-22 | Stage 2 | Attempted multiple deploy approaches: (1) base64 tarball in onstart-cmd — hit Windows 32K char CreateProcess limit; (2) onstart file — hit Vast.ai 4048-char API limit; (3) custom Docker image with BuildKit — base image layers re-uploaded every push due to OCI format recompression (~5GB on 7Mbps upload); (4) legacy Docker builder — same re-upload issue. Final approach: git clone from public GitHub repo in a short onstart-cmd (~700 chars). Created Dockerfile.stage2 (kept for reference), .dockerignore. Made repo public. |
| 2026-02-26 | Stage 2 | Rewrote `deploy_stage2.py` for git clone approach: removed Docker build+push logic, uses off-the-shelf `fishaudio/fish-speech:server-cuda` image, onstart-cmd (~750 chars) git clones repo + installs deps + runs entrypoint. Set up GitHub remote, pushed all code to `sam0ed/avatar` (public). Ready for deployment. |
| 2026-03-01 | Stage 2 | **Stage 2 COMPLETE.** Voice cloning via filesystem-based multi-file references (`setup_voice.py`). Optimized first audio from ~2.5s → **1.0–1.6s** (server-side reference caching, persistent httpx pool, smaller PCM buffer). End-to-end verified. |
| 2026-03-XX | Stage 3 | Implementation: Moonshine Voice ASR wrapper (`client/src/asr/transcriber.py`), audio player with cancel (`client/src/audio/playback.py`), voice conversation client (`client/src/voice_client.py`). Server restructured: chat as background task + `chat_cancel` message type (`server/src/api/server.py`). Mic muting during playback (no barge-in V1). Plan + architecture diagram updated. |
| 2026-03-01 | Stage 3 | **Barge-in implemented.** Mic stays active during avatar speech — Moonshine VAD detects user interruption → cancels playback (`AudioPlayer.cancel()`) + server pipeline (`chat_cancel`) → barge-in text feeds directly into next turn. Relies on laptop mic's built-in echo cancellation to filter speaker output. Researched Pipecat, LiveKit, Vocode barge-in architectures (see `docs/research-asr-turn-taking.md`). |
| 2026-03-01 | Stage 3 | **Barge-in pre-filters + Smart Turn.** Added three pre-filters to reduce false interruptions: backchannel regex (mhm/yeah/okay/etc.), `MIN_INTERRUPTION_WORDS=2`, `MIN_INTERRUPTION_DURATION=0.5s`. Filtered speech is silently ignored (zero audio disruption). Added Smart Turn v3.2 ONNX end-of-turn detection (`client/src/asr/smart_turn.py`): analyzes prosody on up to 8s audio after Moonshine VAD silence, accumulates speech segments until turn complete or 3s timeout. New deps: `transformers>=4.40`, `onnxruntime>=1.17`, `huggingface-hub>=0.23`. Pause/resume and preemptive generation deferred. |

## Important Decisions & Context

- **uv everywhere**: Use `uv` for all Python dependency management. `pyproject.toml` for both server/ and client/. No requirements.txt, no pip freeze.
- **No WSL2**: Everything local runs on native Windows. All dependencies have Windows CUDA wheels.
- **MamayLM over Qwen/Llama**: Chosen for SOTA Ukrainian (outperforms 10x larger models) + strong English from Gemma 2 base.
- **llama-cpp-python over vLLM**: vLLM GGUF support is "highly experimental and under-optimized" (~30-50 tok/s). llama-cpp-python achieves 120-150 tok/s for Q4_K_M on RTX 4090 with native CUDA backend, OpenAI-compatible API.
- **GGUF Q4_K_M over AWQ/GPTQ**: No official AWQ/GPTQ quantized MamayLM exists. Self-quantizing has quality verification concerns. Official GGUF quants from INSAIT (Q4_K_M: 5.76GB, Q8_0: 9.83GB as upgrade path).
- **Single container over VM + Docker Compose**: Originally planned to use Vast.ai KVM VMs with Docker Compose (3 separate containers). KVM instances have a platform bug on Vast.ai. Refactored to a single Docker container with supervisord managing all 3 processes. Base image: `fishaudio/fish-speech:server-cuda` (already has CUDA runtime, PyTorch, and Fish Speech). Added llama-cpp-python via pre-built CUDA wheel in an isolated venv (`/opt/llm-venv`). Orchestrator runs in its own uv project at `/app/orchestrator`. Three isolated Python environments, no dependency conflicts. Regular Docker instances are proven (Stage 0 verified 48ms RTT). Tradeoff: can't independently scale services, but simpler and more reliable.
- **Git clone over custom Docker image**: Stage 2 uses the same off-the-shelf `fishaudio/fish-speech:server-cuda` image as Stage 1 — no custom image build/push. The deploy script sends a short `--onstart-cmd` (~700 chars, well under Vast.ai's 4048-char limit) that `git clone`s the public GitHub repo, installs supervisor + llama-cpp-python, sets up the orchestrator, downloads models, and starts supervisord. This avoids the painfully slow Docker build+push cycle (the ~5GB Fish Speech base image layers get re-uploaded due to BuildKit recompression, taking 60-80 min on a 7 Mbps upload). The repo is made public (secrets stay in `.env` which is gitignored). Tradeoff: ~2 min extra boot time for dep install (negligible vs model download), and code must be public. Previously tried: (1) base64 tarball in onstart-cmd — hit Windows 32K CreateProcess limit; (2) onstart file upload — hit Vast.ai 4048-char API limit; (3) custom Docker image — base layers re-uploaded every time due to BuildKit OCI format mismatch.
- **No KVM VMs on Vast.ai**: KVM instances (`vastai/kvm:cuda-12.6.1-auto`) have a platform bug. Do NOT use. Regular Docker instances work fine.
- **Pre-built CUDA wheels for llama-cpp-python**: Instead of compiling from source (which requires nvcc/devel image), use pre-built wheels from `https://abetlen.github.io/llama-cpp-python/whl/cu126`. This allows using the Fish Speech `runtime` base image without CUDA dev tools.
- **Sentence-level TTS streaming**: LLM tokens are accumulated and split at sentence boundaries (`.!?…`) before dispatching to TTS. This avoids waiting for the full LLM response, achieving first audio within ~1-1.5s.
- **OpenAudio S1-mini over alternatives**: Successor to Fish Speech v1.5. #1 on TTS-Arena2, 0.5B params (~2-3GB VRAM), zero-shot voice cloning, streaming (RTF ~1:7 on 4090), Apache-2.0 code + CC-BY-NC-SA-4.0 weights (fine for personal use). Evaluated Orpheus 3B, XTTS v2, Kokoro, MaskGCT, StyleTTS2-Ukrainian, Piper — S1-mini is the best fit for quality, VRAM, and streaming.
- **Moonshine Voice over faster-whisper**: Moonshine Voice English Small Streaming (123M, ONNX) replaces faster-whisper + Silero VAD. Single library handles both VAD and ASR. Runs on CPU only (0 VRAM), 73ms latency, built-in turn detection. Uses `Transcriber` (manual audio feeding) over `MicTranscriber` for mute/unmute control without session restart and no sounddevice conflicts. See `docs/research-asr-turn-taking.md` for full comparison of Moonshine Small (73ms, 7.84% WER), Moonshine Medium (107ms, 6.65% WER), Moonshine Base Ukrainian (14.55% WER), faster-whisper, RealtimeSTT, Distil-Whisper.
- **ASR runs locally**: Moonshine Voice on CPU to save ~100ms network round trip. Frees GPU VRAM entirely for other tasks.
- **Turn-taking V1 (Moonshine built-in)**: Moonshine's `LineCompleted` event used as turn boundary. VAD segments speech into "lines" with natural pause detection. Configurable `vad_threshold`, `vad_window_duration`. May split long compound sentences mid-thought — acceptable for V1.
- **Turn-taking V2 (Smart Turn — IMPLEMENTED)**: Pipecat Smart Turn v3.2 ONNX model (~8M params, Whisper Tiny backbone, BSD-2-Clause). Runs after Moonshine VAD silence, classifies "complete turn" vs "incomplete turn" on last 8s of audio. ~10-65ms CPU inference. 23 languages incl Ukrainian. Uses `WhisperFeatureExtractor(chunk_length=8)` from `transformers` (no torch required). Model auto-downloaded from `pipecat-ai/smart-turn-v3` on HuggingFace (~8MB, cached). Disableable via `--no-smart-turn` CLI flag. If turn incomplete, waits for more speech (3s timeout fallback).
- **Barge-in V1 (none)**: Mic muted during PROCESSING/SPEAKING. User must wait for avatar to finish. Avoids echo cancellation complexity.
- **Barge-in V2a (headphone VAD)**: Keep mic active during playback, use Moonshine VAD to detect user speech. On `line_started` → cancel playback + server task. Requires headphones (no echo). Simplest, most reliable. ~30 lines changed.
- **Barge-in pre-filters (IMPLEMENTED)**: Three-layer filtering before treating speech as interruption: (1) backchannel regex (mhm, yeah, okay, sure, right, etc.), (2) `MIN_INTERRUPTION_WORDS=2`, (3) `MIN_INTERRUPTION_DURATION=0.5s`. Filtered speech causes zero audio disruption — playback continues uninterrupted, barge-in listener restarts. Pause/resume approach (LiveKit style) deferred — pre-filters handle the common case without any audio glitch.
- **Barge-in V2b (pause/resume — DEFERRED)**: LiveKit/Pipecat pattern — PAUSE playback on VAD trigger, wait 0.5-1s for ASR to confirm real words. If real speech → fully cancel. If noise/echo → resume playback. Would require `pause()`/`resume()` in AudioPlayer. Not needed with current pre-filter approach.
- **Barge-in V2c (energy-gate, optional)**: For speaker-only use without WebRTC. RMS energy monitoring in audio callback — gate audio forwarding to Moonshine during playback. User voice near mic is louder than speaker echo. Only needed if speakers must be supported before WebRTC migration.
- **Barge-in framework research**: Studied Pipecat v0.0.99+ (VAD start strategies, Smart Turn v3 default stop, 5 user mute strategies), LiveKit Agents (AEC warmup 3.0s, false-interruption pause/resume, min_interruption_words/duration), Vocode (mute_during_speech, backchannel regex filtering, interrupt_sensitivity). All rely on transport-layer WebRTC AEC — no framework provides standalone local AEC. See `docs/research-asr-turn-taking.md` for full analysis.
- **No open-source AEC for local Python on Windows**: Evaluated speexdsp, webrtc-audio-processing, pyaec, PINTO0309/onnx-aec, Pipecat filters. None viable. WebRTC will handle AEC when we move to browser-based I/O.
- **Single Vast.ai GPU**: All server models (LLM + TTS + MuseTalk) coexist on one RTX 4090 (~14-17GB total VRAM) in a single Docker container. Can scale by moving a service to a separate Vast.ai instance (change one URL).
- **Latency budget**: 1.5-2.5s end-to-end from end of speech to first avatar response is acceptable.
- **Docker base image**: Switched from `nvidia/cuda:12.1.1-devel` (~12GB) to `python:3.11-slim-bookworm` (1.07GB). Neither Fish Speech nor MuseTalk compile CUDA code — pre-compiled PyTorch wheels, mmcv ships pre-built wheels.
- **S1-mini over Fish Speech v1.5**: v1.5 codebase is being deprecated. S1-mini uses a new DAC codec (replaces VQGAN), different API flags (`--decoder-checkpoint-path` + `--decoder-config-name modded_dac_vq`), and weights at `fishaudio/openaudio-s1-mini`. Checkpoint: model.pth (1.74GB) + codec.pth (1.87GB). 12GB VRAM recommended officially but actual usage ~4.9GB after warmup.
- **Fish Speech Docker**: Use `fishaudio/fish-speech:server-cuda` tag (not `latest-server-cuda`). The `fish` user (UID 1000) defined in the Dockerfile does not exist in Vast.ai runtime — run `start_server.sh` as root directly (not via `su fish`). Model weights are gated on HuggingFace — requires HF token.
- **HF_TOKEN required**: The `fishaudio/openaudio-s1-mini` model is gated. Must accept license at https://huggingface.co/fishaudio/openaudio-s1-mini and provide token via `HF_TOKEN` env var or `.env` file.
- **Vast.ai SSH**: Use `vastai create ssh-key` (not `set ssh-key`). Key propagation to running instances may take a moment.

## Vast.ai Instance Log

Track which machines/offers work, which have direct ports, SSH, etc.

| Date | Offer ID | Instance ID | Location | $/hr | Direct Ports | SSH | Status | Notes |
|------|----------|-------------|----------|------|-------------|-----|--------|-------|
| 2026-02-15 | ? | ? | Denmark, DK | ~$0.27 | Yes | ? | OK | Stage 0 echo test. 48ms RTT. |
| 2026-02-15 | ? | 31466745 | Netherlands, NL | $0.30 | Yes | ? | OK | Stage 1 TTS. IP 38.117.87.41, port 8080→46682. |
| 2026-02-28 | ? | 32170567 | South Africa | ~$0.24 | ? | ? | BAD | Destroyed quickly — not found when checked. |
| 2026-02-28 | ? | 32170653 | Hong Kong, HK | $0.276 | Yes (1024-9549) | No (DNS failure) | BAD | Caddy 401 on all ports. `vastai logs` also failed (DNS). Instance had no outbound internet — onstart-cmd likely never completed. mach_id=53826, host_id=134693. **AVOID host 134693 / machine 53826.** |
| 2026-02-28 | 30834393 | 32171485 | Nevada, US | $0.268 | No (`direct_port_start=-1`) | Yes | OK | SSH tunnel works. All 3 services healthy. mach_id=54431, host_id=74292. Access via `ssh -p 11484 root@ssh8.vast.ai` + port forwarding. |

**Takeaways:**
- Not all machines support direct port mapping (`direct_port_start=-1`). Use SSH tunnel (`-L 8000:localhost:8000`) as fallback.
- Some datacenters (Hong Kong host 134693) have broken networking. Avoid them.
- SSH works on some instances (Nevada) but not others. Try SSH first; if it fails, need direct ports.
- Prefer machines where previous deploys succeeded (host_id 74292 = Nevada, known good).

## Known Risks

- Face animation quality: MuseTalk is the only real-time option; quality is acceptable at Zoom compression but has artifacts
- Vast.ai instances can be preempted: use on-demand, have Docker image pre-built
- Voice clone may need fine-tuning beyond zero-shot for convincing results
- Audio/video sync requires timestamp-based jitter buffering
- Fish Speech first-request latency: ~8-10s for short text (includes model warmup); subsequent requests faster
- Gated model access: HF token must be kept in `.env` (gitignored), never committed
- **Vast.ai networking is inconsistent**: Some machines have no direct ports (need SSH tunnel), some have broken outbound DNS/internet, some have Caddy auth proxies. Always verify health endpoints after deploy. See "Vast.ai Instance Log" above.

## Notes for Agent

- Always check this file at the start of a session to understand project state
- Update the Progress Log after completing any stage or significant milestone
- Update Current Stage when transitioning between stages
- **Use `uv` for ALL Python operations** — never use pip, conda, or poetry. Run scripts with `uv run`, manage deps with `uv add`. Both `server/` and `client/` have their own `pyproject.toml`. The Vast.ai CLI is installed via `uv tool install vastai`.
- When writing Python code: use type hints, async/await for I/O, and uv for dependencies
- **Vast.ai access varies by machine** — some have direct ports, some need SSH tunnels, some have broken networking entirely. Always try SSH first (`ssh -p <port> root@<ssh_host>`), then fall back to direct ports. Check the "Vast.ai Instance Log" section for known-good and known-bad hosts.
- The plan file at `.github/prompts/plan-digitalAvatarClone.prompt.md` has the detailed stage-by-stage breakdown
