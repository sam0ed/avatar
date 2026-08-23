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
├── CLAUDE.md                 # Auto-loaded pointer to this file + the two habits
├── AGENTS.md                 # This file — project memory
├── .github/
│   ├── copilot-instructions.md   # Coding conventions (always-on)
│   └── prompts/
│       └── plan-digitalAvatarClone.prompt.md  # Master build plan
├── server/                   # Vast.ai server code
│   ├── pyproject.toml        # uv-managed dependencies
│   └── src/
│       ├── llm/
│       │   ├── client.py         # Async LLM client (OpenAI-compat SSE streaming)
│       │   ├── chunker.py        # Sentence boundary detection for streaming TTS
│       │   └── system_prompt.txt # System prompt (placeholder)
│       ├── tts/
│       │   └── client.py         # Async Fish Speech TTS client
│       ├── face/                 # MuseTalk service — copied into /opt/musetalk at boot
│       │   ├── client.py             # Async client the orchestrator talks to
│       │   ├── face_server.py        # FastAPI service on :8002, sessions
│       │   ├── musetalk_models.py    # Weight loading, output config
│       │   ├── musetalk_avatar.py    # Reference video → frames/latents/masks + disk cache
│       │   ├── musetalk_audio.py     # PCM → sliding-window whisper chunks
│       │   └── musetalk_render.py    # Batched UNet + VAE + blending → JPEG
│       └── api/
│           └── server.py     # FastAPI + WebSocket, decoupled A/V pipeline
├── client/                   # Local Windows client
│   ├── pyproject.toml        # uv-managed dependencies
│   └── src/
│       ├── asr/
│       │   ├── transcriber.py     # Moonshine Voice ASR wrapper (VAD+ASR, CPU)
│       │   └── smart_turn.py      # Smart Turn v3.2 ONNX end-of-turn detector
│       ├── audio/
│       │   └── playback.py        # Audio player with pause/resume/cancel
│       ├── video/
│       │   └── display.py         # OpenCV window, live + idle playback
│       ├── voice_client.py        # Voice-only conversation client
│       ├── face_voice_client.py   # Voice + face conversation client
│       ├── chat_client.py         # Terminal chat client with audio playback
│       └── tts_test.py            # Fish Speech TTS test client
├── scripts/
│   ├── deploy_tts.py         # Deploy Fish Speech to Vast.ai (Stage 1)
│   ├── deploy_stage2.py      # Deploy audio-only pipeline to Vast.ai
│   ├── deploy_face.py        # Deploy with FACE_ENABLED=true (Stage 4)
│   ├── setup_voice.py        # Upload voice references + enable cloning
│   └── setup_face.py         # Upload reference video + enable face
├── docker/
│   ├── Dockerfile            # Stage 0 orchestrator container (slim)
│   ├── supervisord.conf      # Process manager config (LLM, TTS, musetalk, orchestrator)
│   └── entrypoint_stage2.sh  # Model download + venv setup + supervisord start
├── docs/                     # Research notes (untracked, local only)
├── video/                    # Reference video for face animation (gitignored)
├── recordings/               # Voice samples for cloning (gitignored)
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

### Stage 4 (Face Animation — MuseTalk 1.5)
- Image: `ghcr.io/sam0ed/avatar-stage4:latest`, built by GitHub Actions from `docker/Dockerfile.stage4`
- Deploy: `HF_TOKEN=hf_xxx uv run scripts/deploy_face.py` (FACE_ENABLED=true, 120GB disk, ports 8000/8001/8002/8080)
- The image bakes system deps, the LLM venv and the MuseTalk checkout + venv. Model weights are **not** baked —
  they download at first boot (~16GB) and `hf_transfer`/`hf_xet` make those parallel.
- GitHub Actions is free and unlimited for public repos on standard runners; the workflow reclaims ~25GB of
  runner disk first because the image is ~12GB. `ghcr.io` is free for public packages and has no anonymous pull
  rate limit, unlike Docker Hub — which matters because Vast.ai pulls anonymously.
- **The GHCR package must be set to Public once**, after the first successful build. Packages default to private
  and Vast.ai has no registry credentials, so the pull fails with a 401 until this is done.
- Setup: `uv run scripts/setup_face.py --video <reference.mp4> --url http://<ip>:8000`
- Client: `cd client && uv run python src/face_voice_client.py ws://<ip>:8000/ws`

## Current Stage

**Stage 4: Face Animation (MuseTalk 1.5)** — REWRITTEN, NOT YET RUN

The face service has been rebuilt against MuseTalk's real API and split into `musetalk_models.py`,
`musetalk_avatar.py`, `musetalk_audio.py`, `musetalk_render.py` and `face_server.py` under `server/src/face/`.
It has still **never been executed** — no instance has been deployed with `FACE_ENABLED=true`. Treat the first
deploy as the verification step for the whole stage, not as a routine boot.

Stage 3 (speech-to-speech with barge-in) is complete and verified in real spoken conversations, and is
unaffected by any of the above: face animation is gated behind `FACE_ENABLED` at five points (deploy script,
entrypoint, supervisord `autostart`, orchestrator import, and a per-turn `_active_avatar_id` check), and the
entrypoint's face block is non-fatal, so `deploy_stage2.py` cannot reach face code at all.

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
| 2026-03-08 | Stage 3 | **Stage 3 COMPLETE — verified in real conversations.** Barge-in rewritten to pause/verify/resume (LiveKit pattern) with fade-out instead of hard cancel (`462db4f`). Fixed self-interruption by waiting for playback to start before monitoring (`f2c15c4`). Added ASR error tolerance + interruption context for the LLM (`bc23e1d`). Fixed the interrupted marker leaking into speech output (`a0c6cd6`). Fixed TTS torch.compile failure by symlinking `libcuda.so` for Triton's link step (`64e6bcd`) — the `-lcuda` error in `logs.md` predates this fix and is resolved. |
| 2026-03-08 | Stage 4 | **Code written, NEVER EXECUTED.** ~2,300 lines committed in `1c8c0fb`: `face_server.py` (standalone MuseTalk service, :8002), `face/client.py`, decoupled A/V fork in `server.py`, `video/display.py`, `face_voice_client.py`, `deploy_face.py`, `setup_face.py`, MuseTalk venv + model download in the entrypoint, supervisord program. No instance was ever deployed with `FACE_ENABLED=true` and there are no follow-up fix commits — every other stage has several, produced by actually running it. Reference video recorded (`video/WIN_20260308_14_11_18_Pro.mp4`, 1280×720, 16s, 479 frames) but never uploaded. `face_server.py` was written against a MuseTalk API that does not exist and would have crashed at startup. This row was written retroactively on 2026-08-15; the original commit recorded no caveat, which is why `CLAUDE.md` now asks for verification status in commit messages. |
| 2026-08-15 | Stage 4 | **Face service rewritten against MuseTalk's real API. Still not run.** Split the 703-line `face_server.py` into `musetalk_models.py` (loading, weight preflight), `musetalk_avatar.py` (reference video → frames/latents/masks), `musetalk_audio.py` (PCM → `[T,50,384]` whisper chunks), `musetalk_render.py` (batched UNet + VAE + blending) and a thin `face_server.py` (HTTP + sessions). Every MuseTalk call site verified against upstream `main` rather than recalled. Orchestrator fixes: `/face/enable` compared against the wrong dict level so face could never be enabled; `/face/prepare` passed `avatar_id=None`; `/face/avatars` double-nested its response; prepare timeout raised from 60s. Client fix: `VideoDisplay.advance_frame()` was never called, so the window stayed blank — the display thread now paces itself at 25fps. Entrypoint: MuseTalk pins, official weight layout, whisper download added, `gdown --id` (removed in gdown 6.x) → positional id, and the whole face block made non-fatal. Concurrency: GPU work serialised off the event loop, sessions hold an `AvatarData` snapshot so re-preparing an avatar cannot mutate a live stream, idle sessions swept. Verified by an adversarial subagent review against upstream source; 12 findings, all triaged. |

| 2026-08-16 | Stage 2/4 | **TTS restored and measured on a live instance (47872115).** `fishaudio/fish-speech:server-cuda` had drifted onto the S2-beta tokenizer rewrite, which cannot read `openaudio-s1-mini`'s `tokenizer.tiktoken`. Fixed by installing fish-speech from pinned commit `d3df50503b36` into `/opt/fish-speech` via its own `uv.lock` (`--extra cu126`), removing the base image's shadowing S2 tree at `/app`, and pinning the base by digest. Verified end to end with voice cloning from the 5 recordings: full WebSocket conversation in Ukrainian, first audio median **1.14s** / mean 1.41s / p90 2.39s over 10 turns. Engine-level TTS with cloning: TTFB **0.53s** (7 chars) / **0.85s** (50) / **2.03s** (190), RTF 2.20 / 0.53 / 0.26. VRAM: LLM 9142 MiB, TTS 5044 MiB, MuseTalk stopped, **9881 MiB free** of 24564. Face animation not exercised in this run (`--no-face`). |

| 2026-08-16 | Stage 4 | **Stage 4 VERIFIED WORKING for the first time, on an RTX A6000 (instance 47877527).** Avatar prepared from `video/WIN_20260308_14_11_18_Pro.mp4`: 958 frames in 186s. A full WebSocket turn produced 275 JPEG frames at 853×480 with **11.00s of video against 11.01s of audio — exact A/V sync** — plus correct lip articulation and clean blending. Voice cloning active. This is the first time the face pipeline has run end to end since it was written in March. Remaining problem is throughput, not correctness: see the MuseTalk fps entry below. |

| 2026-08-23 | Stage 4 | **MuseTalk reaches real time: 5.5 → 28.9 fps on the same A6000 benchmark; end-to-end first video 20.8s → 2.51s, delivery 26.0 fps (instance 48459526).** Per-stage instrumentation (CUDA-synced StageTimer, `scripts/bench_face_profile.py` run over localhost) attributed the loss to three causes, fixed and measured one at a time: (1) a ~17s one-time first-feed cost (lazy imports + cuDNN autotune) now paid by a startup warmup pass; (2) serial JPEG encode, parallelised 8.4 → 1.5 ms/frame; (3) the big one — `tensorflow==2.12` (unused) initialises OpenMP in the MuseTalk process, ~36 spinning threads burn the container's cgroup CPU quota (2,053s of accumulated `throttled_usec`), and the resulting throttling made identical mel-extraction work cost anywhere from 3ms to 2.4s. Pinning `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` in the musetalk supervisord environment stabilised mel at 1.7 ms/frame. Remaining steady-state cost is the model itself: VAE 13.6 + UNet 8.3 ms/frame (GPU), blend 7.8 ms/frame (CPU, GIL-bound — thread pool only bought 8.3→7.8). |

| 2026-08-23 | Stage 2/4 | **TTS TTFB made flat via true incremental streaming: 0.77/1.34/3.67s → 0.75/0.92/0.76s at 7/50/190 chars (A6000, cloning on); E2E first audio 3.00s → 1.84s with video enabled.** Root cause was serving, not the model: pinned fish-speech's `generate_long` yields codes once after the WHOLE text is generated, so first byte waited for the last semantic token (TTFB ≈ prefill + 22.5 tokens/audio-second ÷ ~60 tok/s on A6000 — arithmetic matched measurements exactly). Patch `docker/patches/fish-speech-streaming.patch` adds `generate_stream` (verbatim sampling loop, yields 25-frame code chunks, 1-frame holdback mirrors the batch path's `[:-1]`) and growing-prefix DAC decode in the engine, gated by `FISH_STREAM_CHUNK_FRAMES` (0 = stock path). Quality proven numerically, not by ear alone: on identical codes, growing-prefix decode vs monolithic = **SNR 108.8 dB** (float noise; 16-bit audio ceils at ~96). The obvious left-context-window decode was tried first and REJECTED at 2-4 dB SNR — the DAC needs its full left prefix, while right context beyond ~0 frames is irrelevant (prefix decode matched full decode to 2e-5). |

| 2026-08-23 | Stage 2/4 | **KV-prefix reuse added to the streaming patch: prefill 0.45s → 0.09s, engine first audio now 0.31/0.40/0.41s at 7/50/190 chars; E2E first audio 1.24s, first video 1.69s with face enabled.** Measured first: the prompt is ~5.7k tokens (5 references as VQ codes + their transcripts + request text) and prefill of it cost 0.45–1.07s per request — `use_memory_cache` (fish's own feature) caches only the DAC **encoding** of references, never the LLaMA prefill. Since the reference prefix is byte-identical across requests and KV slots are addressed by absolute position, the previous request's KV for the longest common prefix with the previous prompt is reused as-is (no re-forward at all); only the text tail is prefilled. Mathematically identity-preserving: same KV → same logits → same tokens. Self-healing via LCP: changed references → LCP < 256 → full prefill. `audio_masks`/`audio_parts` are None for VQ-reference prompts (raw-audio-in-context only), which the reuse path requires and asserts by gating. Full TTFB journey this day: 3.67s → 0.76s (incremental decode) → **0.41s** (KV reuse) at 190 chars — 9x. |

| 2026-08-23 | Stage 2/4 | **TTS decode 65 → 99 tok/s (+54%) by switching decode attention from SDPBackend.MATH to EFFICIENT_ATTENTION.** Found while profiling why decode costs 15.4 ms/token against a ~3.5 ms floor: inductor-generated code materializes a `(1, 8, 8192, 128)` bf16 buffer per token — the MATH sdpa backend (forced upstream "better for Inductor to codegen") materializes attention intermediates over the FULL 8192-slot KV cache on every token, every layer. EFFICIENT_ATTENTION computes the same attention tiled, without materialization; FLASH is not applicable (rejects the explicit mask the model passes). Same math → no quality tradeoff; ear-check sample sent. The A6000 now matches the 4090's old 103 tok/s, and TTS occupies the GPU ~35% less per sentence, directly easing the TTS↔MuseTalk contention. Also learned: per-token decode cost scales with KV length — the 5.7k-token reference prompt costs on EVERY token, not just prefill, so reference length is now a decode-speed lever too, not only VRAM/prefill. Profiling harness caveat recorded: a standalone loop around the compiled decode re-triggers cudagraph warmup per call and leaks eager buffers (38 GB OOM) — profile inside the real serving loop instead. |

| 2026-08-23 | Stage 4 | **Multi-GPU topology verified on 2×4090 (instance 48472949, Spain): boot-time auto-placement works, sustained delivery 24.8 → 29.3 fps.** Entrypoint counts GPUs and assigns LLM_GPU/TTS_GPU/FACE_GPU (1 GPU: all shared; 2: MuseTalk isolated; 3+: one each), supervisord pins via CUDA_VISIBLE_DEVICES — same image and code on any host. Verified live: `GPU placement: count=2 llm=0 tts=0 face=1`, GPU0 = LLM 9.1 + TTS 5.1 GiB, GPU1 = MuseTalk 7.4 GiB — the full stack fits 24 GB cards now. Long-reply test: 91 s of A/V delivered in 78.8 s (29.3 fps sustained), warm first audio 1.11–1.35 s, first video 1.31–1.54 s, avatar prep 102.9 s (was 186–314 on A6000). |

## Important Decisions & Context

- **After the GPU-side fixes, TTS decode is bound by host-CPU single-core speed, not by the GPU** — established 2026-08-23 on the Spain 2×4090 host. Evidence chain: 55 tok/s there vs 99 on the A6000 despite a faster GPU at 98 % util, full clocks, no throttling; MATH vs EFFICIENT_ATTENTION A/B on that host showed **zero difference** (55.4 vs 55), so attention cost was no longer binding; cudagraphs record and replay (verified via TORCH_LOGS); cgroup quota generous and unthrottled; single-core spin benchmark 143 ms vs ~80–90 ms on fast cores. Conclusion: per-token host-side orchestration (Python loop + dispatch around the graph) costs ~10 ms/token on a fast core and ~18 ms on EPYC 9754-class dense cores, and it scales with CPU clock, not GPU tier. Consequences: (1) host selection for TTS must weigh single-core CPU speed alongside disk_bw — check `cpu_name` in offers; (2) the next software lever is removing per-token Python from the loop (multi-token graph capture), which no GPU upgrade substitutes for. |
- **Same-seed TTS output is NOT reproducible across process restarts, and decode is not bit-deterministic even within one process** — established 2026-08-23 while validating the streaming patch. Two same-seed runs in one process produced identical lengths but different md5s; the stock (unpatched) path in a fresh process produced a different token count than the same request before the restart. Consequence: wav-diff against a stored reference can never validate a change on this stack; isolate the changed numeric stage and compare it on fixed inputs instead (that is what the SNR 108.8 dB growing-prefix measurement does).
- **Never let OpenMP defaults loose in a cgroup-quota'd container** — root-caused 2026-08-23. Vast.ai containers are CPU-quota'd; a library that spins up per-core OpenMP threads (here: TensorFlow, imported by transformers merely because MuseTalk's requirements install it) burns the quota on spin-waits — one mel-extraction call showed 1,234ms of CPU time for 188ms of wall — and the kernel then throttles the whole container in 100ms windows, randomly stalling every CPU stage and even host-side GPU dispatch. Diagnose with `/sys/fs/cgroup/cpu.stat` (`throttled_usec` climbing) and wall-vs-`process_time` divergence. Fix: `OMP_NUM_THREADS=1` etc. in the program's supervisord environment. This, not hardware, was most of the "MuseTalk is 2x too slow" gap; the same mechanism likely explains the 314s-vs-186s avatar-prep variance between identical A6000s.
- **The full stack does not fit on a 24GB card; it needs ~23.2 GiB and OOMs under voice cloning** — measured 2026-08-16 on a 4090 with `FACE_ENABLED=true`. Process-level VRAM: LLM 8.86 GiB, TTS 7.04 GiB, MuseTalk 7.25 GiB = 23.15 of 23.52 GiB, leaving 365 MiB. Voice cloning then fails with `CUDA out of memory` while encoding references, because fish encodes **every** reference sample into the prompt on every request and our five recordings are 34–42s each. On an RTX A6000 (48GB) the identical stack uses 21074 of 49140 MiB with **27.4 GB free** and cloning works (`warmup=ok`). Two traps: MuseTalk's `/health` reports `vram_mb` from `torch.cuda.memory_allocated`, which showed 1931 MiB against a real process footprint of 7280 MiB — nearly 4x understated, so never size hardware from that field. And the earlier "9881 MiB free" figure was measured with `--no-face`, a configuration we do not ship.
- **MuseTalk renders ~16 fps where 25 fps is needed, and it is not the transport** — measured 2026-08-16 on an A6000. Over localhost inside the container: 73 frames in 4.58s = **15.9 fps**; the same test from a workstation gave 10.2 fps, so ~35% of that gap was internet round-trip and must not be attributed to the service. Larger audio chunks do not fix it: 0.3s → 10.2 fps, 1.0s → 11.7, 2.0s → 11.2 (measured over the network), so it plateaus rather than scaling with batch. Models already run fp16 (`.half()` on UNet, VAE and `pe`). CPU-side profiling on the box: JPEG encode ≤9.5 ms/frame (worst case, incompressible input), base64 ≤0.8 ms/frame, `json.dumps` of an 8-frame batch ≤8.4 ms — together at most ~3% of a 440–500ms feed. **The base64-in-JSON transport on the orchestrator↔face hop is NOT the bottleneck**; roughly 85% is GPU work (UNet + VAE decode + whisper features) at ~50 ms/frame. Upstream claims 30fps+ on a Tesla V100, which is weaker than an A6000, so ~1.6x is unexplained and the next step is instrumenting `render_frames` itself rather than changing hardware or transport.
- **A6000 is materially slower than a 4090 for TTS, more than clock ratios suggested** — measured 2026-08-16. Engine TTFB with cloning, 4090 → A6000: 7 chars 0.53s → 0.77s; 50 chars 0.85s → 1.34s; 190 chars 2.03s → 3.67s. End-to-end first audio median 1.14s → 1.73s, p90 2.39s → 2.51s. Caveat: the A6000 run had `FACE_ENABLED=true` so MuseTalk was resident (though idle during the engine benchmark), while the 4090 run had it off — the comparison is not perfectly clean. Bandwidth was **not** the mechanism: fish's own log reports `Bandwidth achieved: 88.59 GB/s` against the 4090's 1008 GB/s peak, under 9% utilisation, so the A6000's 768 GB/s is not the constraint. Lower clocks and fewer SMs are.
- **Blackwell (RTX 5090) is blocked by a specific version floor, not by absence of builds** — checked 2026-08-16. Official `mmcv` wheels 404 for cu124/cu126/cu128 (they stop at `cu121/torch2.4.0`) and `llama-cpp-python` stops at `cu125`. Third-party builds do exist — `miropsota.github.io/torch_packages_builder` ships `mmcv==2.2.0+pt2.7.0cu128` — but `mmdet==3.1.0` and `mmpose==1.1.0` require **`mmcv<2.2.0`**, which is exactly where the cu128 build starts, and the community `llama-cpp-python` cu128 wheels are `win_amd64` while we run Linux. Source-building mmcv under CUDA 12.8 is reported to fail at the CUDA extension stage. fish-speech alone is already cu128-ready (its `pyproject.toml` ships `cu128`/`cu129` extras). Note also that `requirements.txt` pins `tensorflow==2.12.0`, which is what forces `numpy<1.24` and hence the fragile 1.23.5 ABI pin — TensorFlow appears unused by this PyTorch model and dropping it would be the first move in any future migration.
- **First audio is bounded by the LLM's first sentence, not by TTS** — measured 2026-08-16. Across 10 turns, `first_token` was a flat 0.07–0.11s and TTS TTFB was 0.5–0.9s for short text, yet end-to-end first audio ranged 0.73s → 2.47s and tracked **reply length**, not prompt length: a 1.8s reply gave first audio at 0.82s, an 11.8s reply at 2.39s. The orchestrator accumulates tokens until a sentence boundary before dispatching to TTS, so a verbose answer means a long first sentence means late first audio. The highest-leverage latency work is therefore cutting the *first* chunk at a clause/comma boundary and reverting to sentences afterwards — not swapping in a faster engine.
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
- **Vast.ai CLI 1.x renders a TUI, so always script against `--raw`**: from 1.0 onward `vastai show instances` prints a colored, box-drawn table. On Windows that crashes outright with `'charmap' codec can't encode characters` unless `PYTHONIOENCODING=utf-8` is set, and even then the output carries ANSI escapes that make text parsing useless. Every script must use `--raw` and parse JSON. The flags the deploy scripts rely on (`--image`, `--disk`, `--direct`, `--env`, `--onstart-cmd`) and the `search offers --raw` key names are unchanged from 0.5.0 to 1.5.4.
- **MuseTalk is driven through its own API, never re-derived**: use `scripts/realtime_inference.py` in the MuseTalk checkout as the reference for any change to the face service. The first Stage 4 attempt was written from memory of that API and every call site was wrong. All MuseTalk model paths are relative to its repo root, so the face service runs with `cwd=/opt/musetalk` (set by supervisord `directory=` and again by `enter_musetalk_root()`). Note the upstream folder is spelled `face-parse-bisent`, not `bisenet`.
- **`fishaudio/fish-speech:server-cuda` is a mutable tag and it drifted** — verified 2026-08-15. Upstream now defaults to a different model: the image ships `LLAMA_CHECKPOINT_PATH=checkpoints/s2-pro` and `DECODER_CHECKPOINT_PATH=checkpoints/s2-pro/codec.pth`, while the entrypoint downloads `openaudio-s1-mini`. TTS starts, then dies with `FileNotFoundError: 'checkpoints/s2-pro'`. `start_server.sh` reads those paths from the environment, so `supervisord.conf` now pins all three explicitly (`DECODER_CONFIG_NAME=modded_dac_vq` happened to already match). Anything inherited from that base image should be treated as unpinned and re-asserted; the long-term fix is to pin the base by digest in `Dockerfile.stage4`.
- **S1-mini broke because fish-speech rewrote its tokenizer, and no released version ever supported S1** — root-caused 2026-08-16. Pinning the checkpoint paths (above) exposed the real failure: `'NoneType' object has no attribute 'encode'`. On 2026-03-10, commits `daa9b4f3`/`b72bcb31` ("S2 beta") replaced `FishTokenizer.__init__` — `import tiktoken` → `tiktoken.core.Encoding` reading `tokenizer.tiktoken`, became `AutoTokenizer.from_pretrained(model_path)` wanting HuggingFace-format files. `openaudio-s1-mini` ships only `tokenizer.tiktoken`, so the new code cannot load it. That code reached us on 2026-06-09 when the `server-cuda` tag was repointed — two days after our last working session on 2026-03-08. There is nothing to roll back to: fish-speech's releases go `v1.5.1` (2025-05-31, the *fish-speech-1.5* model, default checkpoint `checkpoints/fish-speech-1.5` + `firefly-gan` vocoder — it predates S1 by three days) straight to `v2.0.0-beta` (2026-03-10, the break). **The entire ten-month OpenAudio S1 era shipped only through mutable tags and was never released.** Docker Hub has 27 tags total; the only surviving image inside the S1 window is `latest-dev`/`nightly-dev` (2025-09-21, one digest), and its config has zero CUDA/NVIDIA references — it is the CPU web-UI dev image. GHCR denies anonymous pulls. We never recorded the working digest, so the image we ran is unaddressable. **Fix: build fish-speech ourselves from commit `d3df50503b36`** (2026-01-08, "Docs/readme") — the last commit before S2 beta, and therefore what `server-cuda` was built from during our Feb–Mar 2026 working window. Everything between it and `781bf1cd` (the last tokenizer edit) is docs, pre-commit autoupdates and a Gradio 6.x fix, so it is the most-developed S1-era state. Confirmation that it is the right pin: its `parse_args` defaults are `checkpoints/openaudio-s1-mini`, `checkpoints/openaudio-s1-mini/codec.pth`, `modded_dac_vq` — exactly the three values `supervisord.conf` had been force-pinning. It installs deterministically via its committed `uv.lock` with `uv sync --frozen --extra cu126`, whose `[tool.uv.index]` points torch at the cu126 wheel index, matching the base's CUDA 12.6 runtime. `tools/api_server.py` there serves the same msgpack `/v1/tts` the orchestrator already speaks, so no client changes. A commit SHA cannot drift.
- **The base image's S2 source at `/app` shadows our pinned copy** — caught by a build assertion 2026-08-16. The base declares `WORKDIR /app` and `COPY . .`, so the whole v2.0.0-beta tree (including `fish_speech/`) sits there. Python puts the working directory first on `sys.path`, so *any* process started from `/app` imports the S2 `fish_speech` no matter which venv runs it — the exact `AutoTokenizer` code we are avoiding. `Dockerfile.stage4` now does `rm -rf /app` after installing `/opt/fish-speech`, and the build asserts `fish_speech.__file__` starts with `/opt/fish-speech/`. Relying on `supervisord`'s `directory=` alone is not enough; the shadowing copy has to be gone.
- **MuseTalk dependency order is load-bearing — verified on a live instance 2026-08-15**: three separate steps silently drag `numpy` forward off MuseTalk's pinned 1.23.5, and every C extension built against 1.x then dies with `Unable to convert function return value to a Python type` — a pybind11/numpy ABI error that surfaces three imports deep inside `diffusers`, nowhere near its cause. The working sequence, in this exact order: torch → `requirements.txt` → openmim + mmengine → **`mmcv==2.0.1 --only-binary mmcv -f <cu118/torch2.0.0 index>`** → re-pin → `mmdet==3.1.0` → **`matplotlib==3.7.5`** → **`mmpose==1.1.0 --no-build-isolation`** → re-pin. Three traps: `-f` only *adds* an index so pip still prefers PyPI's sdist and builds mmcv from source without `nvcc`, producing a package with **no CUDA ops that reports success** — `--only-binary mmcv` is mandatory. Current matplotlib requires numpy>=1.25, so it must be pinned to 3.7.5. And `mmpose` builds from source with a `setup.py` that needs torch visible, which pip's build isolation hides — hence `--no-build-isolation`. A known-good 168-package freeze from the verified environment is at `docker/musetalk-freeze.txt`.
- **MuseTalk venv pins torch 2.0.1 / cu118**: `mmcv==2.0.1` only publishes pre-built wheels for torch 2.0.x on cu118. On any newer torch, `mim` falls back to compiling from source, which needs `nvcc` that the Fish Speech runtime base image does not have. Do not "modernise" this pin without also solving mmcv. It is isolated in `/opt/musetalk-venv` and cannot affect the LLM (`/opt/llm-venv`) or TTS (`/app/.venv`) environments — that isolation is the reason four Python environments exist in one container.
- **Face setup is non-fatal**: the `FACE_ENABLED` block in `entrypoint_stage2.sh` runs with `set +e` because face animation is optional, and a failed MuseTalk install must not abort the entrypoint before supervisord starts the LLM, TTS and orchestrator.
- **Streaming lip sync is approximate by construction**: whisper's encoder attends globally over its 30s input, so each new audio chunk shifts encoder output at every position in the trailing segment — including positions backing frames already sent. Early frames of a response are computed against a mostly zero-padded future. Making streaming features match offline ones requires a fixed sliding window with explicit left context, which is a design change, not a constant to tune.

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
