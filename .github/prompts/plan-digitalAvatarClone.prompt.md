## Plan: Real-Time Digital Avatar Clone

Build a self-hosted digital clone that captures your voice, generates conversational responses via an LLM, synthesizes speech in your cloned voice, animates your face, and pipes everything into Zoom as a virtual webcam + microphone. All open-source, compute-heavy inference on Vast.ai (RTX 4090 24GB), local orchestrator + ASR on your RTX 3060 laptop (Windows).

**Architecture overview:**

```
LOCAL (RTX 3060 laptop, Windows) VAST.AI (RTX 4090 24GB, Linux)
┌─────────────────────────┐ WebSocket ┌──────────────────────────────┐
│ Mic → Silero VAD (CPU) │───audio───────→│ (optional) ASR fallback │
│ faster-whisper (GPU,2GB)│ │ │
│ Orchestrator (Python) │───text────────→│ LLM (MamayLM 9B, vLLM) │
│ │ │ ↓ token stream │
│ │←──audio chunks─│ TTS (Fish Speech v1.5) │
│ │←──video frames─│ ↓ │
│ VB-Cable → Virtual Mic │ │ Face Anim (MuseTalk) │
│ pyvirtualcam → Zoom │ └──────────────────────────────┘
└─────────────────────────┘
```

Total VRAM on Vast.ai: ~16-19GB (LLM 6-7GB + TTS 3-4GB + MuseTalk 4-6GB). Fits comfortably on a 24GB card. Cost: ~$0.30-0.50/hr.

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
2. **Deploy Fish Speech v1.5** on Vast.ai. Add Fish Speech dependencies to the Docker image and redeploy. Fish Speech is chosen over alternatives because:
   - Excellent zero-shot cloning from short reference audio
   - Native multilingual support (English + Ukrainian)
   - Streaming output support (~400ms to first chunk on RTX 4090)
   - Active open-source project with good documentation
3. **Test voice cloning**: Send text + reference audio clip → receive synthesized audio. Compare quality. Iterate on reference clips if needed.
4. **Fine-tune** (optional at this stage): If zero-shot quality isn't convincing enough, fine-tune Fish Speech on your full recording set for a tighter voice match.
5. **Verification**: Play synthesized audio to someone who knows your voice. They should recognize it as "you" (or at least "close to you"). Test both English and Ukrainian if relevant.

### Stage 2: LLM + TTS Conversational Pipeline

1. **Deploy MamayLM** ([INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1](https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-2-9B-IT-v0.1), 4-bit quantized) via **vLLM** on the same Vast.ai instance. Add vLLM to the Docker image. MamayLM chosen because:
   - Gemma 2 9B fine-tuned specifically for Ukrainian by INSAIT/ETH Zurich
   - State-of-the-art Ukrainian: outperforms Qwen 2.5 72B and Llama 3.1 70B on Ukrainian benchmarks
   - Retains strong English capabilities from the Gemma 2 base
   - Expert in Ukrainian cultural and linguistic nuances
   - Available in GGUF format as fallback for llama.cpp serving
   - 9B params, 4-bit quantized ≈ ~6-7GB VRAM
2. **Write a system prompt** capturing your personality, speaking style, knowledge areas, and typical response patterns. Keep it concise but specific.
3. **Wire the LLM→TTS streaming pipeline** on the server: extend the existing FastAPI WebSocket endpoints to accept text input → stream through LLM → pipe output sentence-by-sentence into Fish Speech TTS → return audio chunks. Key: **token-level streaming** — don't wait for the full LLM response before starting TTS.
4. **Build a simple local client**: Text input (terminal) → send to server over WebSocket → receive and play audio chunks via `sounddevice`.
5. **Verification**: Type questions in the terminal, hear your clone respond in your voice. Latency target: first audio chunk within ~1-1.5s of hitting Enter.

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
4. **Merge and deploy**: Merge LoRA weights, quantize to 4-bit AWQ, serve via vLLM replacing the base model.
5. **Verification**: Have conversations with colleagues. Ask them if the responses "sound like you."

---

## Verification Checklist

| Stage | Test | Pass Criteria |
|-------|------|---------------|
| 0 | `nvidia-smi` on Vast.ai + WebSocket ping from laptop | GPU visible, <100ms RTT |
| 1 | Play cloned voice to someone who knows you | They recognize it as you |
| 2 | Type text → hear response in your voice | First audio chunk <1.5s |
| 3 | Full voice conversation | <2.5s response latency |
| 4 | Animated face with lip sync | Lip movement matches audio |
| 5 | Complete Zoom call through avatar | Other participant sees/hears avatar |
| 6 | Personality fidelity | Colleagues confirm responses sound like you |

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| TTS | Fish Speech v1.5 | Best multilingual (EN + UK), streaming support, strong zero-shot cloning |
| LLM | MamayLM (Gemma 2 9B, 4-bit) | SOTA Ukrainian (beats 10x larger models), strong English, fine-tunable with QLoRA |
| Face Animation | MuseTalk | Only real-time option at sufficient quality (~30 FPS); LivePortrait as future upgrade |
| ASR | faster-whisper large-v3-turbo + Silero VAD | Runs locally on RTX 3060 Windows (2GB VRAM), saves ~100ms network hop |
| Communication | WebSocket (binary/msgpack) | Simpler than WebRTC, sufficient for 1-2s latency target |
| Vast.ai GPU | RTX 4090 24GB | All server models fit in ~16-19GB, cost-effective at ~$0.40/hr |
| Deployment | Docker on Vast.ai | Reproducible, fast cold starts, no manual setup per instance |
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
| LLM | MamayLM (Gemma 2 9B, 4-bit) via vLLM | Vast.ai | ~6-7GB |
| Text-to-Speech | Fish Speech v1.5 | Vast.ai | ~3-4GB |
| Face Animation | MuseTalk | Vast.ai | ~4-6GB |
| Virtual Camera | pyvirtualcam + OBS Virtual Camera | Local | 0 |
| Virtual Microphone | VB-Audio Virtual Cable + sounddevice | Local | 0 |
| Orchestrator | Python (asyncio + websockets) | Local | 0 |
| Server | FastAPI + WebSocket | Vast.ai | 0 |
| Containerization | Docker (nvidia/cuda base) | Vast.ai | — |