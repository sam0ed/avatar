# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
#     "sounddevice>=0.4",
# ]
# ///
"""Record voice samples for Fish Speech voice cloning.

Usage:
    uv run scripts/record_voice.py
    uv run scripts/record_voice.py --duration 30 --output recordings/sample_01.wav
    uv run scripts/record_voice.py --list-devices

Records 16kHz mono WAV files suitable for Fish Speech reference audio.
Suggested: record 3-5 clips of 20-60 seconds each, reading diverse text.
"""

import argparse
import logging
import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

logger = logging.getLogger("avatar.record")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

SAMPLE_RATE = 16000  # Fish Speech expects 16kHz or will resample internally
CHANNELS = 1
DTYPE = "int16"


def list_audio_devices() -> None:
    """Print available audio input devices."""
    print("\n=== Audio Input Devices ===")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " *" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']} (inputs: {dev['max_input_channels']}){marker}")
    print("\n  * = default device\n")


def record_audio(duration: float, device: int | None = None) -> np.ndarray:
    """Record audio from microphone.

    Args:
        duration: Recording duration in seconds.
        device: Audio input device index, or None for default.

    Returns:
        Audio data as int16 numpy array.
    """
    logger.info("Recording for %.1f seconds... (speak now!)", duration)
    logger.info("Press Ctrl+C to stop early.")

    try:
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=device,
        )
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        # Get what was recorded so far
        logger.info("Recording stopped early by user.")
        frames_recorded = int(sd.get_stream().time * SAMPLE_RATE) if sd.get_stream() else len(audio)
        audio = audio[:frames_recorded]

    # Trim trailing silence (below threshold)
    amplitude = np.abs(audio.flatten())
    if amplitude.max() == 0:
        logger.warning("No audio detected! Check your microphone.")
        return audio.flatten()

    # Find last sample above noise floor
    threshold = amplitude.max() * 0.01
    last_nonsilent = np.where(amplitude > threshold)[0]
    if len(last_nonsilent) > 0:
        end_idx = min(last_nonsilent[-1] + SAMPLE_RATE, len(amplitude))  # Keep 1s padding
        audio = audio[:end_idx]

    duration_actual = len(audio) / SAMPLE_RATE
    peak_db = 20 * np.log10(amplitude.max() / 32768) if amplitude.max() > 0 else -96
    logger.info("Recorded %.1f seconds, peak level: %.1f dB", duration_actual, peak_db)

    if peak_db < -30:
        logger.warning("Recording level is low! Move closer to the microphone.")
    if peak_db > -3:
        logger.warning("Recording is clipping! Move further from the microphone.")

    return audio.flatten()


def save_wav(audio: np.ndarray, path: Path) -> None:
    """Save audio as 16kHz mono WAV.

    Args:
        audio: Audio data as int16 numpy array.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    logger.info("Saved: %s (%.1f seconds)", path, len(audio) / SAMPLE_RATE)


def interactive_session(output_dir: Path, device: int | None = None) -> None:
    """Run an interactive recording session with multiple takes.

    Args:
        output_dir: Directory to save recordings.
        device: Audio input device index, or None for default.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Voice Recording Session for Fish Speech Cloning")
    print("=" * 60)
    print("""
Tips for best results:
  - Record in a quiet room, no background noise
  - Keep a consistent distance from the microphone (~20cm)
  - Speak naturally, with varied intonation and emotions
  - Read diverse content (questions, statements, exclamations)
  - Aim for 3-5 clips, 20-60 seconds each

Suggested scripts to read:
  1. "Hello! My name is [your name]. I work as a [job] and I love
     talking about technology, science, and programming."
  2. "That's a really interesting question! Let me think about it
     for a moment. I believe the answer depends on several factors."
  3. "I'm not sure I agree with that perspective. In my experience,
     things tend to work quite differently in practice."
  4. "Wow, that's amazing! I had no idea that was possible. Can you
     tell me more about how that works?"
  5. "Let me explain this step by step. First, you need to set up
     the environment. Then, install the dependencies. Finally, run
     the main script and check the output."
""")

    take = 1
    existing = list(output_dir.glob("sample_*.wav"))
    if existing:
        take = max(int(f.stem.split("_")[1]) for f in existing) + 1
        print(f"  Found {len(existing)} existing recording(s), starting at take {take}.\n")

    while True:
        cmd = input(f"  [Take {take}] Press ENTER to start recording (or 'q' to quit): ").strip().lower()
        if cmd == "q":
            break

        duration = 60.0  # Max duration; user can Ctrl+C to stop early
        print("  🎙️  Recording... Press Ctrl+C when done speaking.\n")

        try:
            audio = record_audio(duration, device=device)
        except Exception as e:
            logger.error("Recording failed: %s", e)
            continue

        if len(audio) < SAMPLE_RATE * 2:
            logger.warning("Recording too short (< 2 seconds). Try again.")
            continue

        path = output_dir / f"sample_{take:02d}.wav"
        save_wav(audio, path)

        # Ask for transcript
        transcript = input("  Type what you just said (for reference text, or ENTER to skip): ").strip()
        if transcript:
            transcript_path = path.with_suffix(".txt")
            transcript_path.write_text(transcript, encoding="utf-8")
            logger.info("Saved transcript: %s", transcript_path)

        take += 1
        print()

    recordings = list(output_dir.glob("sample_*.wav"))
    total_duration = sum(
        wave.open(str(f), "rb").getnframes() / SAMPLE_RATE for f in recordings
    )
    print(f"\n  Session complete! {len(recordings)} recording(s), {total_duration:.0f}s total.")
    print(f"  Files saved in: {output_dir.resolve()}\n")


def main() -> None:
    """Entry point for the voice recording script."""
    parser = argparse.ArgumentParser(
        description="Record voice samples for Fish Speech voice cloning",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("recordings"),
        help="Output directory or file path (default: recordings/)",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=60.0,
        help="Max recording duration in seconds (default: 60, Ctrl+C to stop early)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device index (default: system default)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Record a single clip instead of interactive session",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    if args.single:
        audio = record_audio(args.duration, device=args.device)
        path = args.output if args.output.suffix == ".wav" else args.output / "sample_01.wav"
        save_wav(audio, path)
    else:
        output_dir = args.output if args.output.suffix != ".wav" else args.output.parent
        interactive_session(output_dir, device=args.device)


if __name__ == "__main__":
    main()
