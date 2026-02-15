"""Test Fish Speech voice cloning on Vast.ai.

Usage:
    uv run python src/tts_test.py --server http://77.33.143.182:PORT \
        --reference recordings/sample_01.wav \
        --reference-text "Hello, my name is ..." \
        --text "This is a test of voice cloning."

    # Or use a pre-uploaded reference by ID:
    uv run python src/tts_test.py --server http://77.33.143.182:PORT \
        --reference-id my_voice \
        --text "This is a test of voice cloning."

Sends text + reference audio to Fish Speech API, receives synthesized audio,
saves it and optionally plays it back.
"""

import argparse
import base64
import logging
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

logger = logging.getLogger("avatar.tts_test")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def check_health(server_url: str) -> bool:
    """Check if Fish Speech API server is healthy.

    Args:
        server_url: Base URL of the Fish Speech API server.

    Returns:
        True if server is healthy.
    """
    import httpx

    try:
        resp = httpx.get(f"{server_url}/v1/health", timeout=10)
        data = resp.json()
        logger.info("Server health: %s", data)
        return data.get("status") == "ok"
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return False


def upload_reference(
    server_url: str,
    reference_id: str,
    audio_path: Path,
    reference_text: str,
) -> bool:
    """Upload a reference voice to the Fish Speech server.

    Args:
        server_url: Base URL of the Fish Speech API server.
        reference_id: Unique ID for this reference voice.
        audio_path: Path to reference audio WAV file.
        reference_text: Transcript of what was said in the audio.

    Returns:
        True if upload succeeded.
    """
    import httpx

    audio_bytes = audio_path.read_bytes()
    logger.info("Uploading reference '%s' (%d bytes)...", reference_id, len(audio_bytes))

    try:
        resp = httpx.post(
            f"{server_url}/v1/references/add",
            files={"audio": (audio_path.name, audio_bytes, "audio/wav")},
            data={"id": reference_id, "text": reference_text},
            timeout=30,
        )
        if resp.status_code == 200:
            import ormsgpack

            data = ormsgpack.unpackb(resp.content)
            logger.info("Reference uploaded successfully: %s (msg: %s)", reference_id, data.get("message"))
            return True
        else:
            logger.error("Upload failed (%d): %s", resp.status_code, resp.content[:500])
            return False
    except Exception as e:
        logger.error("Upload failed: %s", e)
        return False


def list_references(server_url: str) -> list[str]:
    """List available reference voices on the server.

    Args:
        server_url: Base URL of the Fish Speech API server.

    Returns:
        List of reference IDs.
    """
    import httpx

    try:
        resp = httpx.get(f"{server_url}/v1/references/list", timeout=10)
        import ormsgpack

        data = ormsgpack.unpackb(resp.content)
        ref_ids = data.get("reference_ids", [])
        logger.info("Available references: %s", ref_ids)
        return ref_ids
    except Exception as e:
        logger.error("List references failed: %s", e)
        return []


def synthesize_speech(
    server_url: str,
    text: str,
    reference_audio_path: Path | None = None,
    reference_text: str | None = None,
    reference_id: str | None = None,
    output_format: str = "wav",
) -> bytes | None:
    """Synthesize speech using Fish Speech TTS API.

    Either provide (reference_audio_path + reference_text) for inline reference,
    or reference_id for a pre-uploaded reference.

    Args:
        server_url: Base URL of the Fish Speech API server.
        text: Text to synthesize.
        reference_audio_path: Path to reference audio WAV file (inline mode).
        reference_text: Transcript of reference audio (inline mode).
        reference_id: Pre-uploaded reference voice ID.
        output_format: Output audio format (wav, mp3, pcm).

    Returns:
        Synthesized audio bytes, or None on failure.
    """
    import ormsgpack

    # Build request payload
    payload: dict = {
        "text": text,
        "format": output_format,
        "streaming": False,
        "normalize": True,
        "max_new_tokens": 1024,
        "top_p": 0.8,
        "temperature": 0.8,
        "repetition_penalty": 1.1,
        "chunk_length": 200,
    }

    if reference_id:
        payload["reference_id"] = reference_id
    elif reference_audio_path and reference_text:
        audio_bytes = reference_audio_path.read_bytes()
        payload["references"] = [
            {
                "audio": base64.b64encode(audio_bytes).decode("ascii"),
                "text": reference_text,
            }
        ]
    else:
        logger.warning("No reference provided — using random voice.")

    logger.info("Synthesizing: '%s' (%d chars)...", text[:80], len(text))
    start = time.perf_counter()

    try:
        import httpx

        # Fish Speech API uses ormsgpack for request body
        body = ormsgpack.packb(payload)
        resp = httpx.post(
            f"{server_url}/v1/tts",
            content=body,
            headers={"Content-Type": "application/msgpack"},
            timeout=60,
        )

        elapsed = time.perf_counter() - start

        if resp.status_code == 200:
            audio_data = resp.content
            logger.info(
                "Synthesized %d bytes in %.2fs (%.1f KB)",
                len(audio_data),
                elapsed,
                len(audio_data) / 1024,
            )
            return audio_data
        else:
            logger.error("TTS failed (%d): %s", resp.status_code, resp.text[:500])
            return None
    except Exception as e:
        logger.error("TTS request failed: %s", e)
        return None


def play_audio(audio_data: bytes, sample_rate: int = 44100) -> None:
    """Play WAV audio data through speakers.

    Args:
        audio_data: WAV file bytes.
        sample_rate: Sample rate (auto-detected from WAV header if possible).
    """
    import io
    import wave

    try:
        with wave.open(io.BytesIO(audio_data), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            dtype = np.int16 if wf.getsampwidth() == 2 else np.float32
            audio_array = np.frombuffer(frames, dtype=dtype)
            if channels > 1:
                audio_array = audio_array.reshape(-1, channels)

        logger.info("Playing audio: %.1fs at %dHz...", len(audio_array) / sample_rate, sample_rate)
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()
        logger.info("Playback complete.")
    except Exception as e:
        logger.error("Playback failed: %s", e)


def main() -> None:
    """Entry point for TTS test client."""
    parser = argparse.ArgumentParser(
        description="Test Fish Speech voice cloning",
    )
    parser.add_argument(
        "--server", "-s",
        required=True,
        help="Fish Speech API server URL (e.g., http://77.33.143.182:10349)",
    )
    parser.add_argument(
        "--text", "-t",
        default="Hello! This is a test of voice cloning with Fish Speech. How does it sound?",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--reference", "-r",
        type=Path,
        help="Path to reference audio WAV file (for inline cloning)",
    )
    parser.add_argument(
        "--reference-text", "-rt",
        help="Transcript of the reference audio",
    )
    parser.add_argument(
        "--reference-id", "-rid",
        help="Pre-uploaded reference voice ID",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "output" / "tts_test.wav",
        help="Output file path (default: <project_root>/output/tts_test.wav)",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        default=True,
        help="Play audio after synthesis (default: True)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Don't play audio after synthesis",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload reference audio to server (requires --reference and --reference-text)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available reference voices on server",
    )

    args = parser.parse_args()
    server_url = args.server.rstrip("/")

    # Health check
    if not check_health(server_url):
        logger.error("Server is not healthy. Exiting.")
        sys.exit(1)

    # List references
    if args.list:
        list_references(server_url)
        return

    # Upload reference
    if args.upload:
        if not args.reference or not args.reference_text:
            logger.error("--upload requires --reference and --reference-text")
            sys.exit(1)
        ref_id = args.reference_id or args.reference.stem
        upload_reference(server_url, ref_id, args.reference, args.reference_text)
        return

    # Synthesize
    audio_data = synthesize_speech(
        server_url=server_url,
        text=args.text,
        reference_audio_path=args.reference,
        reference_text=args.reference_text,
        reference_id=args.reference_id,
    )

    if audio_data is None:
        logger.error("Synthesis failed. Exiting.")
        sys.exit(1)

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(audio_data)
    logger.info("Saved synthesized audio: %s", args.output)

    # Play
    if args.play and not args.no_play:
        play_audio(audio_data)


if __name__ == "__main__":
    main()
