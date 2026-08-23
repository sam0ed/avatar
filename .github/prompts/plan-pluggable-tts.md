# Design: Pluggable TTS Engines

Status: proposed, not implemented
Date: 2026-08-23 (supersedes the 2026-08-16 design; history in git)

## Objective

Add Higgs Audio v3 (patriotyk's Ukrainian fine-tune — the weights the listening test
selected) alongside the working fish-speech S1-mini, behind one engine interface, so both
can be compared live on the same hardware, same avatar, same conversation. The code must
stay general: adding engine N is one module and one registry entry.

## What we know now (inputs to this design)

| Fact | Measured |
|---|---|
| fish S1-mini is the fast baseline | first audio 0.31–0.6 s engine-level, 99 tok/s (fast-CPU host), streaming + KV-prefix reuse + efficient attention patched in |
| TTS decode is host-CPU single-core bound | 10 ms/token fast core vs 18 ms EPYC-dense core; GPU tier does not substitute |
| Sample rates differ | fish 44 100 Hz, Higgs 24 000 Hz; MuseTalk hardcodes 44 100 — unplumbed, lip sync desyncs silently |
| Client needs no change for 24 kHz | `playback.py` reads the rate from each chunk's WAV header |
| Resampling is exact at both rates | group 441 → 160 @44.1k and 441 → 294 @24k (441 divisible by 3) |
| Multi-GPU placement works | boot-time `GPU placement` in entrypoint, `CUDA_VISIBLE_DEVICES` per program; no cross-GPU traffic exists |
| VRAM (process-level) | LLM 9.1, fish 5.1, MuseTalk 7.4 GiB; Higgs ~10 GiB bf16 + SGLang KV pool |
| Higgs serving | SGLang-Omni, OpenAI `/v1/audio/speech`, `stream:true` + `response_format:"pcm"` @24 kHz, cloning via `references[{audio_path,text}]` |
| Transcripts are mandatory for cloning refs | missing `ref_text` triggers per-request ASR-class costs; our `/app/references` has `.lab` for every sample |

## Architecture

### One interface, engines as adapters

```
server/src/tts/
  engine.py               TTSEngine protocol, EngineSpec registry, create_engine()
  engine_fish_speech.py   msgpack /v1/tts adapter (today's client.py, moved)
  engine_openai_audio.py  OpenAI /v1/audio/speech adapter (Higgs; any future
                          OpenAI-compatible TTS comes free)
```

```python
@runtime_checkable
class TTSEngine(Protocol):
    @property
    def sample_rate(self) -> int: ...
    @property
    def voice_enabled(self) -> bool: ...
    @property
    def reference_id(self) -> str | None: ...
    def set_reference_id(self, ref_id: str) -> None: ...
    def clear_reference_id(self) -> None: ...
    async def health_check(self) -> bool: ...
    async def warmup(self) -> bool: ...
    def synthesize_streaming(self, text: str) -> AsyncIterator[bytes]: ...
```

The members are exactly what `server.py` already calls on `tts_client`
(`voice_enabled`/`reference_id` back `/voice/enable` and `/voice/status`), plus
`sample_rate`. `synthesize_streaming` yields **raw PCM at `engine.sample_rate`** — the
engine's wire format stops leaking into the orchestrator.

An `ENGINES` registry maps name → `EngineSpec(factory, base_url, port, sample_rate)`.
`create_engine(name)` raises on unknown names — a silent fallback would invalidate every
comparison this design exists to enable.

### Contract change: raw PCM in, per-chunk WAV out

Today the fish client yields WAV-wrapped chunks and `_video_producer` strips headers with
`wav_chunk[44:]`. Under the new contract engines yield raw PCM; the orchestrator forwards
those bytes to the video queue unmodified and wraps **each** client-bound chunk with a WAV
header built at `engine.sample_rate`. The client already takes its playback rate from that
header, so 24 kHz reaches it with zero client changes.

### Sample-rate plumbing (the sharpest edge)

Four hops, all currently missing; without them MuseTalk computes whisper features on the
wrong timebase and lip sync drifts silently:

1. `musetalk_audio.append_pcm` / `resample_for_whisper` take the source rate as an
   argument (no module constant).
2. `AnimationSession` stores the rate it was started with.
3. `face_server` `/session/start` accepts a `sample_rate` form field.
4. `face/client.py.start_session` sends `engine.sample_rate` — the hop that today has no
   parameter at all.

`server.py`'s `batch_threshold = 26460` becomes `int(engine.sample_rate * 2 * 0.3)` so the
video feed granularity stays 0.3 s of audio at any rate.

### Selection: deploy-time processes, runtime activation

Two separate questions, answered separately:

- **Which engine servers run** is a deploy-time decision: `TTS_ENGINES` env (comma list,
  default `fish`). The entrypoint downloads weights and supervisord starts one program per
  listed engine — fish on 8080, higgs on 8081. Engines not listed cost nothing.
- **Which engine the orchestrator uses** is a runtime decision: it starts with the first
  listed engine and `POST /tts/engine {"name": ...}` switches among the *running* ones.
  Switching is cheap by construction — the orchestrator-side engine is a stateless HTTP
  client plus a sample rate; the expensive part (the model server) is already up.

This gives A/B within one conversation on one avatar, which is the point of the exercise,
without any hot-loading machinery. Face sessions are started per turn and already receive
the rate per session, so a mid-conversation switch is safe at turn boundaries.

### Higgs specifics

- Weights: patriotyk's Ukrainian fine-tune of `bosonai/higgs-audio-v3-tts-4b` (the model
  behind the HF space the listening test picked; exact repo id read off that space at
  implementation time and pinned by revision).
- Serving: SGLang-Omni in its own venv `/opt/higgs-venv`, launched by supervisord as
  `program:tts-higgs`; low-latency (c1) configuration, since we are single-user.
- Cloning: profile built from `/app/references/<ref_id>/` — all `.wav`+`.lab` pairs sent
  as `references[]`; a missing `.lab` raises rather than silently degrading to ASR.
- SGLang gives prefix caching (RadixAttention) and CUDA-graph serving out of the box — the
  mechanisms we hand-built for fish arrive free here, which keeps the comparison fair.

### Image: one image, engine layers

One image remains the artifact (already the pattern: isolated venvs under `/opt`). The
Higgs venv is one more cacheable layer. Weights are never baked; the entrypoint downloads
only the engines listed in `TTS_ENGINES`. If image size ever hurts, the two-stage
base+engine split is the known escape hatch; not now.

### GPU placement, engine-aware

The entrypoint's placement table gains engine weight-classes rather than hardcoded names:

| GPUs | LLM | fish (5 GiB) | higgs (~14 GiB) | MuseTalk |
|---|---|---|---|---|
| 1 | 0 | 0 | 0 (fits 48 GB; refuses to start both on 24 GB) | 0 |
| 2 | 0 | 0 | 1 | 1 |
| 3+ | 0 | 1 | 1 | 2 |

All assignments remain env-overridable (`TTS_GPU`, `HIGGS_GPU`, …). On 2×24 GB with both
engines: GPU0 = LLM 9.1 + fish 5.1; GPU1 = MuseTalk 7.4 + Higgs with
`--mem-fraction-static` capped to fit — measured before being trusted.

## Sequencing

1. **Contract layer, local**: protocol + registry + fish adapter move + raw-PCM/WAV-wrap
   in the orchestrator. No GPU needed; unit tests on fixtures.
2. **Rate plumbing, local**: the four hops + derived `batch_threshold`; exactness tests
   for 441→160 and 441→294.
3. **Fish regression on hardware**: deploy, standard bench set + full A/V conversation.
   The working engine must be provably unbroken before the new one lands.
4. **Higgs engine + serving**: adapter, venv, supervisord program, entrypoint weights +
   placement. Confirm the request shape against the running server before building on it.
5. **Bake-off on 2×4090**: bench set per engine, first video-at-24kHz conversation (the
   real test of step 2), ear A/B on identical sentences, VRAM under load. Numbers into
   AGENTS.md; the loser is a config value away, not deleted.

## Testing

Local, no GPU: registry raises on unknown engines; every adapter satisfies `TTSEngine`
structurally; OpenAI adapter's request bodies (streaming, references) against a stub HTTP
server; WAV headers parse back through `wave.open` at both rates; resample exactness.

On hardware: health/warmup per engine; `/tts/engine` switch mid-conversation at a turn
boundary; full A/V conversation per engine; sustained-delivery fps with Higgs+MuseTalk
sharing GPU1.

## Out of scope

- OmniVoice (cut earlier: architecturally cannot stream; ~1.7 s flat TTFA).
- Quantized Higgs; automatic engine fallback; hot-swapping server processes.
- Porting fish's serving optimizations anywhere new — they live in the fish patch.

## Risks

| Risk | Mitigation |
|---|---|
| Higgs + MuseTalk share GPU1: contention returns for the higgs path | Measured in step 5 with the same profiling that caught it last time; 3-GPU placement is a flag away |
| SGLang KV pool overshoots 24 GB alongside MuseTalk | `--mem-fraction-static` capped; VRAM measured under a real conversation before acceptance |
| patriotyk fine-tune's serving config drifts from base Higgs assumptions | Request shape confirmed against the live server before the adapter hardens |
| Rate plumbing missed somewhere → silent lip-sync drift | Rate travels explicitly per session; step 3 regression + step 5 24 kHz conversation are the gates |
| Host-CPU lottery skews the bake-off | Both engines benched on the same instance; host CPU recorded in AGENTS.md with the numbers |
