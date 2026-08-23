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

### Selection: one engine per deployment

`TTS_ENGINE` env (default `fish`), consumed in three places: the entrypoint downloads only
that engine's weights, supervisord's single `[program:tts]` runs `start_tts.sh` which
dispatches on the value and execs the selected server **always on port 8080**, and the
orchestrator calls `create_engine(TTS_ENGINE)`. An unknown value fails loudly in all
three. Nothing else in the system knows more than one engine exists; the registry is a
code-structure device, not runtime machinery.

An idle second engine server was considered and rejected: Higgs holds its ~12–14 GiB KV
pool whether or not it is speaking, which is too much VRAM to pay for the convenience of
mid-conversation switching. A/B runs as it has all along — the same benchmark sentences
and recorded wavs across two deploys, host CPU noted next to the numbers.

### Deployment details (each item is a previously-paid-for lesson)

- **Env forwarding through supervisord.** `supervisord.conf` `environment=` blocks
  override inherited container env — the exact mechanism that silently pinned
  `TTS_BASE_URL` in the first design review. Both `[program:tts]` and
  `[program:orchestrator]` must carry `TTS_ENGINE="%(ENV_TTS_ENGINE)s"` explicitly, and
  the entrypoint must export `TTS_ENGINE` before starting supervisord (same contract as
  `FACE_ENABLED`).
- **Per-engine readiness.** The entrypoint's TTS wait loop is `curl /v1/health` with a
  60 s budget — fish-shaped twice over. SGLang serves `/health`, and a cold Higgs start
  (weights + graph capture) runs minutes, not seconds. The wait loop and the supervisord
  `startsecs` switch on `TTS_ENGINE`: fish `/v1/health`, 60 s; higgs `/health`, 300 s.
- **VRAM preflight, fail at boot.** `TTS_ENGINE=higgs` + `FACE_ENABLED=true` needs
  ~29.5 GiB (LLM 9.1 + Higgs ~13 + MuseTalk 7.4) and cannot fit one 24 GB card. The
  entrypoint compares the selected configuration's known footprints against detected
  VRAM and exits with an actionable message instead of letting voice cloning OOM
  mid-conversation, which is how the 24 GB limit announced itself last time.
- **Deploy script.** `deploy_face.py --tts-engine` passes `-e TTS_ENGINE=...`; disk stays
  at 120 GB (Higgs weights ~10 GB fit the existing budget).

### Higgs specifics

- Weights: patriotyk's Ukrainian fine-tune of `bosonai/higgs-audio-v3-tts-4b` (the model
  behind the HF space the listening test picked; exact repo id read off that space at
  implementation time and pinned by revision).
- Serving: SGLang-Omni in its own venv `/opt/higgs-venv`, launched by supervisord as
  `program:tts-higgs`; low-latency (c1) configuration, since we are single-user.
- Cloning: profile built from `/app/references/<ref_id>/` — all `.wav`+`.lab` pairs sent
  as `references[]` with filesystem paths, which works because the SGLang server runs in
  the same container (the same assumption fish's server-side folder read already makes).
  A missing `.lab` raises rather than silently degrading to ASR.
- SGLang gives prefix caching (RadixAttention) and CUDA-graph serving out of the box — the
  mechanisms we hand-built for fish arrive free here, which keeps the comparison fair.

### Image: one image, engine layers

One image remains the artifact (already the pattern: isolated venvs under `/opt`). The
Higgs venv is one more cacheable layer. Weights are never baked; the entrypoint downloads
only the engine named by `TTS_ENGINE`. If image size ever hurts, the two-stage
base+engine split is the known escape hatch; not now.

### GPU placement: unchanged, engine-agnostic

The existing placement table already encodes the right principle — the concurrent
workloads (TTS and MuseTalk) get separated, the pipelined ones (LLM and TTS) share:

| GPUs | LLM | TTS (any engine) | MuseTalk |
|---|---|---|---|
| 1 | 0 | 0 | 0 |
| 2 | 0 | 0 | 1 |
| 3+ | 0 | 1 | 2 |

No engine-aware machinery. VRAM per configuration: 1×48 GB fits any engine; 2×24 GB with
Higgs puts LLM 9.1 + Higgs on GPU0, which requires capping SGLang's
`--mem-fraction-static` to ~13 GiB — measured under a real conversation in step 5 before
being trusted. If a future engine cannot share GPU0 with the LLM on a given card, the fix
is the existing `TTS_GPU` override, not new policy.

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

On hardware: health/warmup per engine; full A/V conversation per engine;
sustained-delivery fps and VRAM headroom with Higgs sharing GPU0 with the LLM.

## Out of scope

- OmniVoice (cut earlier: architecturally cannot stream; ~1.7 s flat TTFA).
- Quantized Higgs; automatic engine fallback; hot-swapping server processes.
- Porting fish's serving optimizations anywhere new — they live in the fish patch.

## Risks

| Risk | Mitigation |
|---|---|
| LLM + Higgs on one 24 GB card: KV-pool cap may starve Higgs throughput, or bursts may collide | Measured in step 5 with the same profiling that caught the last contention; `TTS_GPU=1` override and 3-GPU placement are one env away |
| SGLang KV pool overshoots 24 GB alongside MuseTalk | `--mem-fraction-static` capped; VRAM measured under a real conversation before acceptance |
| patriotyk fine-tune's serving config drifts from base Higgs assumptions | Request shape confirmed against the live server before the adapter hardens |
| Rate plumbing missed somewhere → silent lip-sync drift | Rate travels explicitly per session; step 3 regression + step 5 24 kHz conversation are the gates |
| Host-CPU lottery skews the bake-off | Both engines benched on the same instance; host CPU recorded in AGENTS.md with the numbers |
