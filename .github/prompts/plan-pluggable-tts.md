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
| fish S1-mini is the fast baseline | engine-level first audio 0.31–0.41 s on a fast-CPU host, 0.50–0.63 s on a dense-EPYC host (decode is host-CPU bound); streaming + KV-prefix reuse + efficient attention patched in |
| TTS decode is host-CPU single-core bound | 10 ms/token fast core vs 18 ms EPYC-dense core; GPU tier does not substitute |
| Sample rates differ | fish 44 100 Hz, Higgs 24 000 Hz; MuseTalk hardcodes 44 100 — unplumbed, lip sync desyncs silently |
| Client needs no change for 24 kHz | `playback.py` opens its output stream once per turn from the first chunk's WAV header — safe because a deployment has exactly one rate |
| Resampling is exact at both rates | group 441 → 160 @44.1k and 441 → 294 @24k (441 divisible by 3) |
| Multi-GPU placement works | boot-time `GPU placement` in entrypoint, `CUDA_VISIBLE_DEVICES` per program; no cross-GPU traffic exists |
| VRAM (process-level) | LLM 9.1, fish 5.1, MuseTalk 7.4 GiB; Higgs ~10 GiB bf16 + SGLang KV pool |
| Higgs serving | SGLang-Omni, OpenAI `/v1/audio/speech`, `stream:true` + `response_format:"pcm"` @24 kHz, cloning via `references[{audio_path,text}]` |
| Higgs VRAM budget is ONE number | weights ~10 GiB; total resident = the `HIGGS_VRAM_GIB` cap (default 13), enforced via SGLang `--mem-fraction-static`. The launch command, the boot preflight and every figure in this document read the same constant — three diverging numbers is how the last OOM got past review |
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
   argument (no module constant), and the resample group is **derived**, not hardcoded:
   `group_in = rate // gcd(rate, 16000)`. 441 happens to be exact for both 44 100 and
   24 000, but it is a coincidence, not an invariant — a 32 kHz engine would desync
   silently. Exactness is asserted per registered engine rate, not per known ratio.
2. `AnimationSession` stores the rate it was started with — both construction sites
   (`start_session` and `_warm_render_path`'s warmup session) pass it.
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

An idle second engine server was considered and rejected: Higgs holds its full configured
budget (weights + KV pool, `HIGGS_VRAM_GIB`) whether or not it is speaking, which is too
much VRAM to pay for the convenience of mid-conversation switching. A/B runs as it has all along — the same benchmark sentences
and recorded wavs across two deploys, host CPU noted next to the numbers.

### Deployment details (each item is a previously-paid-for lesson)

- **Env forwarding through supervisord.** `supervisord.conf` `environment=` blocks
  override inherited container env — the exact mechanism that silently pinned
  `TTS_BASE_URL` in the first design review. Both `[program:tts]` and
  `[program:orchestrator]` must carry `TTS_ENGINE="%(ENV_TTS_ENGINE)s"` explicitly, and
  the entrypoint must export `TTS_ENGINE` before starting supervisord (same contract as
  `FACE_ENABLED`).
- **Per-engine readiness AND warmup.** The entrypoint is fish-shaped in three places, not
  two: the health wait loop (`/v1/health`, 60 s budget) and also the warmup request that
  follows it, which POSTs a fish-JSON body to fish's `/v1/tts`. All three branch on
  `TTS_ENGINE`: fish `/v1/health` + 60 s + `/v1/tts` warmup; higgs `/health` + 300 s (cold
  start is minutes: weights + graph capture) + a minimal `/v1/audio/speech` warmup.
  Supervisord `startsecs` follows the same split.
- **VRAM preflight, fail at boot — for single-GPU hosts.** On one GPU everything shares
  the card (placement table row 1), so `TTS_ENGINE=higgs` + `FACE_ENABLED=true` needs
  LLM 9.1 + `HIGGS_VRAM_GIB` + MuseTalk 7.4 ≈ 29.5 GiB and cannot fit a single 24 GB
  card; on 2+ GPUs Higgs and MuseTalk never share, and GPU0 needs 9.1 + `HIGGS_VRAM_GIB`.
  The entrypoint sums the selected configuration's footprints per assigned GPU against
  detected VRAM and exits with an actionable message. The preflight and the SGLang launch
  read the same `HIGGS_VRAM_GIB` constant, so the number the gate checks is the number
  the server enforces; the constant itself is validated under a real conversation in
  step 5 (a passing preflight with a wrong cap would reproduce the last OOM).
- **Dead-export cleanup.** `entrypoint_stage2.sh` still exports `TTS_BASE_URL`, which
  supervisord's hardcoded `environment=` value has silently overridden ever since — the
  live specimen of the trap this section exists to avoid. Remove the dead export when
  threading `TTS_ENGINE`.
- **Deploy script.** `deploy_face.py --tts-engine` passes `-e TTS_ENGINE=...`; disk stays
  at 120 GB (Higgs weights ~10 GB fit the existing budget). Its docstring's static
  download/VRAM estimates become engine-conditional and are updated with the flag.

### Higgs specifics

- Weights: **base** `bosonai/higgs-tts-3-4b` (upstream renamed from
  `higgs-audio-v3-tts-4b`; 9.31 GiB safetensors, verified via the HF API). The
  "patriotyk fine-tune" assumption was checked and was wrong: his HF space's `app.py`
  loads a transformers repack of the base model — the voice the listening test picked IS
  base Higgs v3 with zero-shot cloning, which natively covers Ukrainian. Pinned by
  revision at implementation.
- Serving: SGLang-Omni in its own venv `/opt/higgs-venv` (python 3.12 — the upstream
  validated ABI; installed from the committed `docker/higgs-lock.txt`, since
  `--prerelease=allow` otherwise drifts across rebuilds). There is no separate supervisord
  program: the single `[program:tts]`'s `start_tts.sh` execs `sgl-omni serve --config
  /app/higgs_4090.yaml` when `TTS_ENGINE=higgs`. Memory is governed by per-stage
  `total_gpu_memory_fraction` overrides in that YAML — the defaults (0.03/0.85/0.10 of the
  WHOLE card) would OOM beside the LLM, and `--mem-fraction-static` does not size the
  stage budgets. The stack is CUDA-13 (torch 2.11 + cu13 kernels): hosts need driver
  ≥ 580.65, which the deploy filter and a boot gate both enforce; prebuilt flashinfer
  cubin/jit-cache wheels are baked so first-request JIT never runs on a box without nvcc.
  Base-image premise corrected: it is CUDA 12.9 runtime on Ubuntu 24.04, not 12.6.
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
server; WAV headers parse back through `wave.open` at both rates; resample-group
derivation is exact for every rate in the registry (general assertion, not just the two
known ratios).

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
| Upstream renamed the weights repo once already (`higgs-audio-v3-tts-4b` → `higgs-tts-3-4b`) — mutable-name drift, the fish lesson | Pin by revision hash, not by name; request shape confirmed against the live server before the adapter hardens |
| Rate plumbing missed somewhere → silent lip-sync drift | Rate travels explicitly per session; step 3 regression + step 5 24 kHz conversation are the gates |
| Host-CPU lottery skews the bake-off | Both engines benched on the same instance; host CPU recorded in AGENTS.md with the numbers |
