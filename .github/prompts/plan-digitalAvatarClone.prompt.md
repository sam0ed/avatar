## Plan: Real-Time Digital Avatar Clone

Build a self-hosted digital clone that captures your voice, generates conversational responses via an LLM, synthesizes speech in your cloned voice, animates your face, and pipes everything into Zoom as a virtual webcam + microphone. All open-source, compute-heavy inference on Vast.ai (RTX 4090 24GB), local orchestrator + ASR on your RTX 3060 laptop (Windows).

**Architecture overview:**

```
LOCAL (RTX 3060 laptop, Windows)     VAST.AI (RTX 4090 24GB, Linux)
┌─────────────────────────┐ WebSocket ┌──────────────────────────────────────┐
│ Mic → Silero VAD (CPU) │───audio───────→│ Single container (supervisord)     │
│ faster-whisper (GPU,2GB)│ │ ┌──────────────────────────────┐ │
│ Orchestrator (Python) │───text────────→│ │ orchestrator (FastAPI :8000) │ │
│ │ │ └──────┬───────────┬──────────┘ │
│ │ │ │ │ │
│ │ │ ┌──────▼─────┐ ┌───▼──────────┐ │
│ │ │ │ LLM :8001  │ │ TTS :8080    │ │
│ │ │ │ llama.cpp  │ │ Fish Speech  │ │
│ │←──audio chunks─│ │ MamayLM 9B │ │ S1-mini      │ │
│ │←──video frames─│ └────────────┘ └──────────────┘ │
│ VB-Cable → Virtual Mic │ │ ┌──────────────────────┐ │
│ pyvirtualcam → Zoom │ │ │ Face Anim (MuseTalk) │ │
│ │ │ └──────────────────────┘ │
└─────────────────────────┘ └──────────────────────────────────────┘
```

Server runs on a **Vast.ai Docker instance** — a single container with **supervisord** managing all services (LLM, TTS, face animation, orchestrator) as separate processes sharing the GPU. The image is pre-built and pushed to Docker Hub. Any service can be moved to a separate Vast.ai instance by changing one URL.

Total VRAM on Vast.ai: ~14-17GB (LLM ~5.5GB + TTS ~5GB + MuseTalk 4-6GB). Fits comfortably on a 24GB card. Cost: ~$0.30-0.50/hr.

---

### Stage 0: Infrastructure Setup

1. **Create a Vast.ai account** and verify billing. Identify RTX 4090 instances in a EU/US datacenter close to you for lowest latency (~20-50ms RTT).
2. **Initialize the project repository**:
   - Create `.gitignore` (include `models/`, `.env`, `__pycache__/`, `*.pyc`, `.venv/`, `dist/`).
   - `uv init` in `server/` and `client/` to create `pyproject.toml` for each.
   - Create the directory scaffold: `server/src/{llm,tts,face,api}/`, `client/src/{asr,audio,video}/`, `scripts/`, `docker/`, `models/`.
3. **Install local dependencies** via uv on Windows: `uv add faster-whisper silero-vad pyvirtualcam sounddevice websockets numpy opencv-python` in `client/`. All have native Windows CUDA wheels.
4. **Install Windows-side tools**: OBS Studio (for Virtual Camera driver), VB-Audio Virtual Cable.
5. **Build a minimal Docker image** for the Vast.ai server. Base on `nvidia/cuda:12.1-devel-ubuntu22.04`. Install Python 3.11, uv, PyTorch 2.x. Do NOT pre-install model-specific dependencies yet — those are added incrementally in later stages. Push to Docker Hub.
6. **Build a hello-world FastAPI + WebSocket server** in `server/src/api/`. Deploy it to a Vast.ai instance to validate the full connectivity loop: expose port (e.g., 8000 via `--ports 8000/tcp`), confirm GPU access (`nvidia-smi`), confirm WebSocket round-trip from your laptop.
7. **Verification**: WebSocket ping from laptop to Vast.ai server returns <100ms RTT. GPU visible via `nvidia-smi`.

### Stage 1: Voice Clone Proof-of-Concept

1. **Record voice samples**: 3-5 minutes of clean speech. Use a Python script with `sounddevice` to record 16kHz WAV files. Read a diverse script covering different intonations, emotions, and sentence structures. Record in a quiet room.
2. **Deploy OpenAudio S1-mini** (Fish Speech successor) on Vast.ai. Use the official Fish Speech Docker image (`fishaudio/fish-speech:latest-server-cuda`) or add dependencies to our image. S1-mini chosen over alternatives because:
   - #1 on TTS-Arena2, best open-source TTS quality (WER 0.008, CER 0.004)
   - Zero-shot voice cloning from 10-30s reference audio
   - Streaming output support (RTF ~1:7 on RTX 4090)
   - 0.5B params, ~2-3GB VRAM — fits alongside LLM + MuseTalk
   - Emotion control via markers (50+ emotions, tone control)
   - Active open-source project (24.9k stars), Apache-2.0 code
3. **Test voice cloning**: Send text + reference audio clip → receive synthesized audio. Compare quality. Using all reference samples simultaneously gives the best zero-shot results. Iterate on reference clips if needed.
4. **Pre-upload references**: Before using the TTS server for live synthesis, upload all reference audio samples to the server's `/v1/references/add` endpoint. This avoids sending ~20-30MB of audio on every request and cuts synthesis latency from ~60s to ~8s.
5. **Fine-tune** (optional at this stage): If zero-shot quality isn't convincing enough, fine-tune Fish Speech on your full recording set for a tighter voice match.
6. **Verification**: Play synthesized audio to someone who knows your voice. They should recognize it as "you" (or at least "close to you"). Test both English and Ukrainian if relevant.

### Stage 2: LLM + TTS Conversational Pipeline

1. **Set up Vast.ai single container**: Deploy using the same Fish Speech base image as Stage 1. The deploy script sends a short `--onstart-cmd` (~700 chars) that `git clone`s the public GitHub repo, installs supervisor + llama-cpp-python, sets up the orchestrator, downloads models, and starts all 3 services via supervisord. No custom Docker image build/push needed. Repo must be public (secrets stay in `.env`, gitignored).
2. **Deploy MamayLM** ([INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1-GGUF](https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1-GGUF), Q4_K_M) via **llama-cpp-python** with CUDA backend, serving an OpenAI-compatible API. MamayLM chosen because:
   - Gemma 2 9B fine-tuned specifically for Ukrainian by INSAIT/ETH Zurich
   - State-of-the-art Ukrainian: outperforms Qwen 2.5 72B and Llama 3.1 70B on Ukrainian benchmarks
   - Retains strong English capabilities from the Gemma 2 base
   - Expert in Ukrainian cultural and linguistic nuances
   - Official GGUF quants from INSAIT: Q4_K_M (5.76GB), Q8_0 (9.83GB) as upgrade path
   - llama-cpp-python chosen over vLLM because vLLM GGUF support is "highly experimental and under-optimized" (~30-50 tok/s), while llama.cpp achieves 120-150 tok/s for Q4_K_M on RTX 4090
3. **Write a deployment script** to automate instance creation on Vast.ai. Follow the fire-and-forget pattern established for TTS deployment.
4. **Wire the LLM→TTS streaming pipeline** on the server: stream LLM tokens → split into sentences at boundaries → dispatch each sentence to TTS immediately. Key: **don't wait for the full LLM response before starting TTS**.
5. **Write a system prompt** capturing your personality, speaking style, knowledge areas, and typical response patterns.
6. **Extend the WebSocket server** to accept chat input, stream LLM tokens and audio chunks back to the client.
7. **Build a terminal chat client**: Text input → send to server over WebSocket → receive and play audio chunks. Display LLM tokens in real-time as they stream in.
8. **Verification**: Type questions, hear your clone respond in your voice. First audio chunk within ~1-1.5s. LLM generation speed ≥100 tok/s.

### Stage 3: Speech-to-Speech Conversation

1. **Set up local ASR**: Install `faster-whisper` (large-v3-turbo, INT8 quantized, ~2GB VRAM) on your RTX 3060 (native Windows + CUDA). Pair with **Silero VAD** (CPU) for voice activity detection.
2. **Implement the voice pipeline locally**:
   - Continuously capture mic audio via `sounddevice`
   - Feed into Silero VAD to detect speech segments (start/end of utterance)
   - On utterance end → transcribe with faster-whisper locally
   - Send transcription text over WebSocket to Vast.ai server
   - Receive audio chunks back → play through speakers (or VB-Cable)
3. **Handle turn-taking**: Implement a simple state machine — `LISTENING` → `PROCESSING` → `SPEAKING` → `LISTENING`. Mute mic input while avatar is speaking to avoid feedback loops.
4. **Verification**: Have a full voice conversation with your clone. Speak naturally, hear responses in your voice. Measure end-to-end latency (target: 1.5-2.5s from end of speech to first audio).

### Stage 4: Face Animation

1. **Record/prepare a reference video or photo** of yourself: well-lit, front-facing, neutral expression, high resolution. For MuseTalk, a short (5-10s) video of your face works best as a driving reference.
2. **Deploy MuseTalk** on the Vast.ai instance alongside the existing models. MuseTalk chosen because:
   - Real-time capable (~30 FPS on RTX 4090)
   - Audio-driven (takes TTS audio as input, generates lip-synced face video)
   - Reasonable quality at Zoom compression levels
3. **Extend the server pipeline**: After TTS generates audio chunks, feed them into MuseTalk → generate video frames → stream frames back to client as JPEG or H.264 chunks over WebSocket.
4. **Build the local video display**: Receive frames → display in an OpenCV window. Verify lip sync matches audio.
5. **Add idle animations**: When the avatar is listening (not speaking), generate subtle head movements, eye blinks, and micro-expressions. MuseTalk can be driven with silence + slight pose variations for this.
6. **Optimization**: Tune the video frame size (720p or 480p), compression (JPEG quality ~80), and frame rate (25-30 FPS) to balance quality vs bandwidth.
7. **Verification**: Have a voice conversation while watching the animated face. Lip sync should be reasonably aligned with audio. Face should look recognizably like you.

### Stage 5: Zoom Integration

1. **Set up pyvirtualcam**: Pipe received video frames from the face animation into a virtual webcam via `pyvirtualcam`. On Windows, this requires the OBS Virtual Camera driver. Frames are written as numpy arrays.
2. **Set up VB-Audio Virtual Cable**: Route TTS audio output to VB-Cable Input. In Zoom, select "VB-Cable Output" as the microphone source.
3. **Build the full local client**:
   - Capture real mic → VAD → ASR (local) → WebSocket to Vast.ai
   - Receive audio → play to VB-Cable (virtual mic)
   - Receive video → write to pyvirtualcam (virtual camera)
   - In Zoom: select virtual camera + virtual mic
4. **Handle Zoom-specific quirks**: Test with different Zoom video settings (720p, mirroring, HD video toggle). Ensure frame rate is stable.
5. **Verification**: Join a Zoom call. The other participant should see your animated avatar and hear your cloned voice responding to their questions. This is the minimal prototype milestone.

### Stage 6: LLM Fine-tuning on Personal Data

1. **Collect personal data**: Export Telegram messages, emails, documents, notes — anything that represents how you write and think. Clean and format into conversation-style training data (instruction/response pairs).
2. **Fine-tune with QLoRA**: Use **Unsloth** or **axolotl** to LoRA fine-tune MamayLM (Gemma 2 9B) on your data. 4-bit QLoRA needs ~12-14GB VRAM, fits on the Vast.ai GPU. Typical training: a few hours on a single GPU.
3. **Evaluate**: Compare fine-tuned vs base model responses. Check that it captures your tone, opinions, and knowledge without hallucinating or losing general capability.
4. **Merge and deploy**: Merge LoRA weights, quantize to GGUF Q4_K_M (or Q8_0), serve via llama-cpp-python replacing the base model.
5. **Verification**: Have conversations with colleagues. Ask them if the responses "sound like you."

### Stage 7: Control Panel UI

1. **Build a local web-based control panel** using a lightweight framework (e.g., FastAPI + htmx, or Gradio). This runs locally and provides a single dashboard to manage the entire avatar pipeline.
2. **Server Management tab**: Start/stop Vast.ai instances, select GPU offers (with region filtering), monitor instance status and costs, view server logs.
3. **Voice Setup tab**: Upload/manage reference audio samples to the TTS server, test synthesis with different reference combinations, preview and compare cloned voice output, record new samples directly from the browser.
4. **Pipeline Status tab**: Show real-time status of all components (TTS server, LLM, ASR, face animation), display latency metrics, toggle individual pipeline stages on/off.
5. **Settings tab**: Configure system prompt for LLM, adjust TTS parameters (temperature, top_p, chunk_length), set audio/video device preferences.
6. **Verification**: All pipeline components can be started, configured, and monitored from the UI without touching the terminal.

---

## Verification Checklist

| Stage | Test | Pass Criteria |
|-------|------|---------------|
| 0 | `nvidia-smi` on Vast.ai + WebSocket ping from laptop | GPU visible, <100ms RTT |
| 1 | Play cloned voice to someone who knows you | They recognize it as you |
| 2 | Type text → hear response in your voice | First audio chunk <1.5s, LLM ≥100 tok/s |
| 3 | Full voice conversation | <2.5s response latency |
| 4 | Animated face with lip sync | Lip movement matches audio |
| 5 | Complete Zoom call through avatar | Other participant sees/hears avatar |
| 6 | Personality fidelity | Colleagues confirm responses sound like you |
| 7 | Manage pipeline from UI without terminal | All components controllable via browser |

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| TTS | OpenAudio S1-mini (0.5B) | #1 TTS-Arena2, streaming, zero-shot cloning, 2-3GB VRAM, emotion control |
| LLM | MamayLM (Gemma 2 9B, GGUF Q4_K_M) via llama-cpp-python | SOTA Ukrainian (beats 10x larger models), strong English, fine-tunable with QLoRA. Official GGUF quants from INSAIT. llama.cpp chosen over vLLM (vLLM GGUF is experimental/slow). ~5.5GB VRAM, 120-150 tok/s on RTX 4090. |
| Face Animation | MuseTalk | Only real-time option at sufficient quality (~30 FPS); LivePortrait as future upgrade |
| ASR | faster-whisper large-v3-turbo + Silero VAD | Runs locally on RTX 3060 Windows (2GB VRAM), saves ~100ms network hop |
| Communication | WebSocket (binary/msgpack) | Simpler than WebRTC, sufficient for 1-2s latency target |
| Vast.ai GPU | RTX 4090 24GB | All server models fit in ~16-19GB, cost-effective at ~$0.40/hr |
| Deployment | Vast.ai Docker instance + git clone onstart-cmd | Same off-the-shelf base image as Stage 1. Short onstart-cmd (~700 chars) git clones public GitHub repo — no custom image build/push needed. Supervisord manages all services. Can move any service to a separate GPU by changing one URL. |
| Local setup | Windows native | All dependencies have Windows wheels; direct access to virtual camera/mic without interop |

---

## Key Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Face animation quality (uncanny valley) | HIGH | Use 480-720p, leverage Zoom compression; MuseTalk quality acceptable at call resolution |
| End-to-end latency >2.5s | HIGH | Stream everything (token-by-token LLM→TTS), run ASR locally, choose close datacenter |
| Vast.ai instance preemption | MEDIUM | Use on-demand (not interruptible), pre-baked Docker image, fallback to real webcam |
| Voice clone not convincing | MEDIUM | Fine-tune Fish Speech on 3-5 min of clean recordings; iterate on reference clips |
| Audio/video sync drift | MEDIUM | Timestamp frames, 100-200ms jitter buffer at output stage |
| Ukrainian language quality | LOW | Fish Speech has good multilingual support; fallback to English-only if needed |

---

## Tech Stack Summary

| Component | Tool | Runs On | VRAM |
|-----------|------|---------|------|
| Voice Activity Detection | Silero VAD | Local (CPU) | 0 |
| Speech-to-Text | faster-whisper large-v3-turbo | Local (GPU) | ~2GB |
| LLM | MamayLM (Gemma 2 9B, GGUF Q4_K_M) via llama-cpp-python | Vast.ai | ~5.5GB |
| Text-to-Speech | OpenAudio S1-mini (0.5B) | Vast.ai | ~5GB |
| Face Animation | MuseTalk | Vast.ai | ~4-6GB |
| Virtual Camera | pyvirtualcam + OBS Virtual Camera | Local | 0 |
| Virtual Microphone | VB-Audio Virtual Cable + sounddevice | Local | 0 |
| Orchestrator | Python (asyncio + websockets) | Local | 0 |
| Control Panel UI | FastAPI + htmx (or Gradio) | Local | 0 |
| Server | FastAPI + WebSocket | Vast.ai | 0 |
| Containerization | Single Docker container + supervisord | Vast.ai | — |