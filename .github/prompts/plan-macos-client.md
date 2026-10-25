# Design: The Client Runs on macOS (Meeting Project 1 of 3)

Status: proposed, not implemented
Date: 2026-10-25

## Objective

Make the local client install and run on macOS (Apple Silicon) with every entry point working,
while keeping the Windows client working. Development has moved to a Mac; Windows stays a supported
platform that is not smoke-tested for now, so its code must stay current rather than rot.

This is the first of three meeting projects, done in order:

1. **The client runs on macOS** (this document).
2. **"As me" in Google Meet on macOS**: the avatar replaces the user's camera and mic under the
   user's own account (port of the Windows `--meeting` mode).
3. **"As a participant" in Google Meet**: the avatar joins as its own participant, so everyone,
   including the user, can talk to it.

Projects 2 and 3 get their own design documents. This one only prepares the ground they stand on.

## What we know now (inputs to this design)

| Fact | Source |
|---|---|
| `uv sync` fails on macOS | `pyaudiowpatch` ships only `win32`/`win_amd64` wheels and no sdist; reproduced with uv 0.12.22 |
| Every other client dependency has a macOS arm64 cp311 wheel | PyPI: moonshine-voice (needs macOS 15+), onnxruntime (14+), pyvirtualcam, sounddevice, opencv-python, scipy 1.17.x |
| The OpenCV window crashes off the main thread on macOS | `VideoDisplay` runs `namedWindow`/`imshow`/`waitKey` on a `threading.Thread`; on this Mac that raises `cv2.error: Unknown C++ exception` at `namedWindow`; the same calls work on the main thread |
| Only one module is genuinely platform-bound | `client/src/output/meeting_ears.py` (WASAPI loopback). pyvirtualcam, sounddevice and cv2 select their own backends; the VB-Cable lookup in `devices.py` differs only by device name |
| The client has no echo cancellation in code | barge-in on speakers worked on Windows only because the laptop mic driver cancels echo; plain PortAudio capture on macOS gets none |
| No `uv.lock` is committed | the root `.gitignore` ignores `uv.lock` |
| ASR is CPU-only | Moonshine runs its own bundled ONNX Runtime on CPU; Smart Turn uses the onnxruntime CPU provider; no CUDA assumptions in the client |
| Baseline to compare against | fish regression on 2x4090, 2026-08-23 (instance 48488087): warm first audio 1.22-1.29 s, first video 1.41-1.46 s, Windows client |
| Development machine | Apple M5, 16 GB, macOS 27.0.1; Homebrew and uv present; OBS, BlackHole and ffmpeg not installed |

## Architecture

### Platform selection: one enum, one factory, nothing more

```
client/src/host_platform.py    HostPlatform(StrEnum): WINDOWS = "win32", MACOS = "darwin"
                               current_platform() -> HostPlatform(sys.platform)
```

`current_platform()` raises immediately on any other OS. Code that differs per platform is
selected by `match` on the enum, and the platform module is imported inside the matching branch,
so Windows-only packages are only ever imported on Windows.

In this project the only platform-selected piece is meeting-audio capture:

```
client/src/output/
  loopback.py            LoopbackSource Protocol + open_loopback() factory
  loopback_windows.py    WASAPI loopback via pyaudiowpatch (today's capture code, moved unchanged)
  meeting_ears.py        platform-neutral: thread, downmix, resample, 1600-sample blocks
```

The macOS loopback backend is project 2. Until then, `--meeting` and `--ears meeting` on macOS
fail at startup with an explicit "not supported on macOS yet" error rather than a stack trace.

Per-platform device names (VB-Cable on Windows, BlackHole on macOS) are data, not code: a table
keyed by `HostPlatform`. The macOS entry is filled in by project 2.

Rejected as over-engineering: plugin registries or entry points, base-class hierarchies, wrappers
around pyvirtualcam/sounddevice/cv2, separate per-OS packages, CI (deferred by decision).

### Dependencies and the lockfile

```toml
dependencies = [ ..., "pyaudiowpatch>=0.2.12.8; sys_platform == 'win32'", ... ]

[tool.uv]
environments = ["sys_platform == 'win32'", "sys_platform == 'darwin'"]
required-environments = [
    "sys_platform == 'win32' and platform_machine == 'AMD64'",
    "sys_platform == 'darwin' and platform_machine == 'arm64'",
]
```

`client/uv.lock` is committed (un-ignored for `client/`). `required-environments` makes `uv lock`
on the Mac fail if any dependency stops shipping a Windows x64 or macOS arm64 wheel: the cheapest
available guard for the untested platform. Dependencies added later for macOS only (pyobjc,
`catap`) get a `sys_platform == 'darwin'` marker.

### The window owns the main thread

The rule on every GUI toolkit: GUI calls happen on the main thread; background work runs elsewhere
and hands data in. The client is restructured to match, on both platforms, with one code path:

- **Main thread**: the window loop. Every ~16 ms it shows the current frame and reads a key.
  This is today's `_display_loop` body, run by the caller instead of by a private thread.
- **Worker thread**: the asyncio event loop with everything that exists today (WebSocket, ASR,
  playback, barge-in), unchanged inside.
- **Two crossing points only**: the frame buffer the display already locks, and key presses,
  delivered to the event loop with `loop.call_soon_threadsafe`.
- **Shutdown**: Esc in the window or the end of the async session stops both sides. Python
  delivers Ctrl+C only to the main thread, so the window loop catches it and asks the event loop
  to stop.

OpenCV stays the window for now. Moving to a real UI toolkit (for project 3's join/leave and
status controls) would be its own decision; this structure already lets one drop in.

`voice_client.py` has no window and keeps `asyncio.run` on the main thread.

### Echo cancellation: measured first, then built once

Barge-in on the MacBook's own speakers needs acoustic echo cancellation (AEC). Two candidates:

| | Apple voice processing (AVAudioEngine VP, via pyobjc) | WebRTC AEC3 (`livekit` `AudioProcessingModule`) |
|---|---|---|
| Platforms | macOS only | macOS and Windows wheels |
| Playback | must move into the same AVAudioEngine as the mic; it cancels only the engine's own output | keeps sounddevice playback; the app feeds the played audio as the reference |
| Known side effects | ducks other apps' audio (-30 dB default, -8 dB at `.min`), attenuates other processes' mics, multi-channel input formats, silent failures on some devices | suppresses near-end speech for the first seconds of a session; needs the reference aligned with the echo (estimator range about 0-512 ms) |
| Other people's evidence | the one same-machine MacBook test removed the echo completely | the same test left about 26% of the leak, still not transcribed |

The side effects that drove other projects away from Apple's VP mostly do not apply here: AEC is
only on when talking to the avatar directly on the Mac. In both meeting modes the meeting audio is
captured digitally and never reaches the mic, so AEC is off there.

**Spike (throwaway code)**: on this Mac's built-in speakers and mic, run both candidates and
compare on four checks, decided in advance:

1. Echo removed during playback only, at low, medium and high volume.
2. The user's voice preserved while the avatar talks (double-talk).
3. The first barge-in after client startup.
4. Silent failures (silence or a wrong channel delivered without an error).

The spike reports numbers; the winner is built. There is no default preference.

**Built once, for the winner only**:

- If WebRTC wins: a filter between mic capture and the transcriber, fed with the playback blocks
  as they reach the device. The same code runs on Windows.
- If Apple VP wins: a macOS audio I/O backend that owns both mic capture and TTS playback inside
  one AVAudioEngine, behind the interface `AudioPlayer` and `SpeechTranscriber` already expose
  (`play_queue`, `pause`/`resume`/`cancel`, `played_seconds`, 16 kHz mono float32 blocks). Python
  never runs on the real-time audio thread. Windows keeps today's sounddevice path.

Either way: on by default on macOS, off by default on Windows (the laptop driver already cancels
echo; two cancellers must never stack), off in both meeting modes, overridable with a flag. A
startup self-test fails loudly when the cancelled capture delivers silence.

### Everything runs on macOS

Every entry point must start and do its job on macOS:

| Entry point | Today | Needed |
|---|---|---|
| `face_voice_client.py` | main client | the window restructure; AEC |
| `voice_client.py` | voice only | AEC |
| `chat_client.py`, `tts_test.py`, `connectivity_test.py` | take their target from CLI args | verify only |
| `latency_test.py`, `client/test_ws.py`, `client/test_llm_client.py` | hardcode `localhost` or Feb 2026 instance IPs | take the target as a CLI argument |

## Sequencing

Each step ends with something runnable and is committed separately, with the commit message saying
what was run.

1. **Install**: the `pyaudiowpatch` marker, `environments`/`required-environments`, committed
   `client/uv.lock`. Done when `uv sync --locked` passes on the Mac and `uv lock` resolves both
   platforms.
2. **Platform split**: `HostPlatform`, the loopback Protocol and factory, the Windows backend moved
   unchanged, the fail-fast error for meeting flags on macOS, the device-name table.
3. **Window on the main thread**: the restructure above, on both platforms.
4. **Headphone smoke test** (needs a deployed server, approved by the user before it is rented):
   a full spoken conversation on the Mac with headphones, idle and talking video, barge-in;
   first audio and first video measured. Host egress is measured first, because a thin route
   (the 2026-08-24 finding) would masquerade as a client regression.
5. **Every entry point** runs on macOS, including the scripts that need a CLI target.
6. **AEC spike**: both candidates, four checks, numbers reported.
7. **AEC built** for the winner, with the flag, defaults and self-test.
8. **Speaker test**: the same conversation as step 4 on the MacBook's built-in speakers, barge-in
   included.
9. **AGENTS.md** updated in the same commit as the stage move: drafted by one subagent, checked
   line by line by a second, diff approved by the user.

## Testing

- Unit tests (new `client/tests/`, pytest): platform selection (supported and unsupported
  `sys.platform` values), the platform-neutral half of meeting ears (downmix, resample, 1600-sample
  blocking), the key hand-off from the window loop to the event loop, and whichever AEC glue is
  built (reference alignment for WebRTC, channel selection for Apple VP).
- An import smoke test that imports every client module except platform modules for the other OS.
- Live, on the Mac: steps 4, 5 and 8 above. First audio and first video within about 10% of the
  baseline on a comparable host is the sanity check; more than that is investigated, not accepted.
- Windows: not run. `uv lock` covering both platforms and keeping Windows code behind the same
  interfaces is the whole safety net; commit messages and AGENTS.md say "Windows path not
  re-verified" wherever Windows code was touched.

## Out of scope

- Meeting modes on macOS (project 2) and the participant bot (project 3).
- Any server change; the GPU server stays on Vast.ai as it is.
- CI, including a Windows job (deferred by decision).
- Conversation content: system prompt, knowledge, the "presenter" idea.
- Replacing OpenCV with a UI toolkit.

## Risks

- **Moonshine on macOS 27**: its wheels target macOS 15+; untested on 27 until step 4.
- **Capture at 16 kHz**: the transcriber asks PortAudio for 16 kHz directly; Core Audio devices
  run at 44.1/48 kHz, so this relies on PortAudio's resampling. If it fails, capture at the device
  rate and resample, as `meeting_ears.py` already does.
- **Permissions**: the Microphone (and later Camera) permission is granted to the app that launches
  the client (VS Code or Terminal), not to the client.
- **Apple VP, if it wins**: multi-channel input formats, rebuilds on device change, A/V latency
  reporting that has to be recalibrated, macOS 27 deprecations of the non-`error:` engine calls.
- **WebRTC AEC3, if it wins**: reference alignment against the real device latency; near-end
  suppression at session start weakens the first barge-in.
- **Untested Windows**: the restructured window loop and the moved loopback code are not run on
  Windows until someone has a Windows machine at hand.
