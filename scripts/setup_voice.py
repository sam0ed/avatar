# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
# ]
# ///
"""Upload voice recordings to the orchestrator and enable voice cloning.

All files are written to the shared filesystem under one reference_id
via the orchestrator's POST /voice/reference endpoint.  Fish Speech
loads all audio+lab pairs from references/<ref_id>/ at inference time.

Usage:
    uv run scripts/setup_voice.py
    uv run scripts/setup_voice.py --url http://localhost:8000
    uv run scripts/setup_voice.py --ref-id my-voice
    uv run scripts/setup_voice.py --recordings-dir recordings
    uv run scripts/setup_voice.py --disable
    uv run scripts/setup_voice.py --status
"""

import argparse
import logging
import sys
from pathlib import Path

import httpx

logger = logging.getLogger("avatar.setup_voice")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = PROJECT_ROOT / "recordings"
DEFAULT_URL = "http://localhost:8000"
DEFAULT_REF_ID = "my-voice"


def find_recording_pairs(recordings_dir: Path) -> list[tuple[Path, str]]:
    """Find WAV + transcript pairs in the recordings directory.

    Args:
        recordings_dir: Directory containing sample_XX.wav and sample_XX.txt.

    Returns:
        List of (wav_path, transcript_text) tuples, sorted by name.
    """
    pairs = []
    for wav_path in sorted(recordings_dir.glob("sample_*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if txt_path.exists():
            transcript = txt_path.read_text(encoding="utf-8").strip()
            pairs.append((wav_path, transcript))
        else:
            logger.warning("No transcript for %s, skipping", wav_path.name)
    return pairs


def check_health(base_url: str) -> bool:
    """Check orchestrator health.

    Args:
        base_url: Base URL of the orchestrator.

    Returns:
        True if reachable.
    """
    try:
        resp = httpx.get(f"{base_url}/health", timeout=10.0)
        data = resp.json()
        logger.info("Orchestrator health: %s", data)
        return data.get("status") in ("ok", "degraded")
    except Exception as e:
        logger.error("Cannot reach orchestrator at %s: %s", base_url, e)
        return False


def get_voice_status(base_url: str) -> dict | None:
    """Get current voice cloning status from the orchestrator.

    Args:
        base_url: Base URL of the orchestrator.

    Returns:
        Status dict or None on error.
    """
    try:
        resp = httpx.get(f"{base_url}/voice/status", timeout=10.0)
        return resp.json()
    except Exception as e:
        logger.error("Failed to get voice status: %s", e)
        return None


def list_references(base_url: str) -> dict[str, int]:
    """List reference IDs and their audio file counts from the orchestrator.

    Args:
        base_url: Base URL of the orchestrator.

    Returns:
        Dict mapping ref_id to audio count.
    """
    try:
        resp = httpx.get(f"{base_url}/voice/references", timeout=10.0)
        if resp.status_code == 200:
            return resp.json().get("references", {})
        return {}
    except Exception as e:
        logger.error("Failed to list references: %s", e)
        return {}


def upload_reference(base_url: str, wav_path: Path, transcript: str, ref_id: str) -> bool:
    """Upload a voice reference to the orchestrator's shared filesystem.

    Args:
        base_url: Base URL of the orchestrator.
        wav_path: Path to WAV audio file.
        transcript: Text spoken in the audio.
        ref_id: Reference ID (folder name under /app/references/).

    Returns:
        True on success.
    """
    audio_bytes = wav_path.read_bytes()
    size_kb = len(audio_bytes) / 1024

    resp = httpx.post(
        f"{base_url}/voice/reference",
        data={"text": transcript, "ref_id": ref_id},
        files={"audio": (wav_path.name, audio_bytes, "audio/wav")},
        timeout=60.0,
    )
    if resp.status_code == 200:
        data = resp.json()
        logger.info(
            "  Uploaded %s → '%s' (%.0f KB, %d chars) — %d audio(s) in folder",
            wav_path.name, ref_id, size_kb, len(transcript), data.get("audio_count", "?"),
        )
        return True
    else:
        logger.error(
            "  Upload %s failed (%d): %s",
            wav_path.name, resp.status_code, resp.text[:500],
        )
        return False


def enable_voice(base_url: str, ref_id: str) -> bool:
    """Enable voice cloning on the orchestrator and run warmup.

    Args:
        base_url: Base URL of the orchestrator.
        ref_id: Reference ID (must exist on TTS server).

    Returns:
        True if enabled successfully.
    """
    logger.info("Enabling voice cloning with ref_id='%s' + warmup ...", ref_id)
    resp = httpx.post(
        f"{base_url}/voice/enable",
        params={"ref_id": ref_id},
        timeout=180.0,
    )
    data = resp.json()
    if data.get("enabled"):
        logger.info(
            "  Voice cloning ON (ref=%s, warmup=%s)",
            data.get("reference_id", "?"), data.get("warmup", "?"),
        )
        return True
    else:
        logger.error("  Enable failed: %s", data)
        return False


def disable_voice(base_url: str) -> bool:
    """Disable voice cloning on the orchestrator.

    Args:
        base_url: Base URL of the orchestrator.

    Returns:
        True if disabled successfully.
    """
    resp = httpx.post(f"{base_url}/voice/disable", timeout=10.0)
    data = resp.json()
    logger.info("Voice cloning disabled: %s", data)
    return not data.get("enabled", True)


def main() -> None:
    """Upload voice recordings to orchestrator and enable cloning."""
    parser = argparse.ArgumentParser(
        description="Upload voice recordings and enable voice cloning",
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Orchestrator URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--ref-id", default=DEFAULT_REF_ID,
        help=f"Reference ID — all recordings are stored under this ID (default: {DEFAULT_REF_ID})",
    )
    parser.add_argument(
        "--recordings-dir", type=Path, default=RECORDINGS_DIR,
        help="Directory with sample_*.wav/txt pairs",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show voice cloning status and exit",
    )
    parser.add_argument(
        "--disable", action="store_true",
        help="Disable voice cloning and exit",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Skip upload, just enable cloning with existing references",
    )
    args = parser.parse_args()

    # Health check
    if not check_health(args.url):
        sys.exit(1)

    # Status-only mode
    if args.status:
        status = get_voice_status(args.url)
        if status:
            print(f"Voice cloning: {'ON' if status['enabled'] else 'OFF'}")
            print(f"Reference ID: {status.get('reference_id', 'none')}")
        refs = list_references(args.url)
        if refs:
            print(f"Available references: {refs}")
        return

    # Disable mode
    if args.disable:
        disable_voice(args.url)
        return

    ref_id = args.ref_id

    if not args.skip_upload:
        # Find recording pairs
        if not args.recordings_dir.is_dir():
            logger.error("Recordings directory not found: %s", args.recordings_dir)
            sys.exit(1)

        pairs = find_recording_pairs(args.recordings_dir)
        if not pairs:
            logger.error("No sample_*.wav + sample_*.txt pairs in %s", args.recordings_dir)
            sys.exit(1)

        logger.info("Found %d recording pairs in %s", len(pairs), args.recordings_dir)

        # Upload all recordings under a single ref_id
        uploaded = 0
        for wav_path, transcript in pairs:
            if upload_reference(args.url, wav_path, transcript, ref_id):
                uploaded += 1

        if not uploaded:
            logger.error("All uploads failed")
            sys.exit(1)

        logger.info("Uploaded %d/%d recordings under ref_id='%s'", uploaded, len(pairs), ref_id)
    else:
        refs = list_references(args.url)
        if ref_id not in refs:
            logger.error("Reference '%s' not found on server (available: %s)", ref_id, list(refs.keys()))
            sys.exit(1)
        logger.info("Using existing reference '%s' (%d audios)", ref_id, refs[ref_id])

    # Enable voice cloning + warmup on orchestrator
    if not enable_voice(args.url, ref_id=ref_id):
        sys.exit(1)

    logger.info("")
    logger.info("Voice cloning setup complete!")
    logger.info("  Test: cd client && uv run python src/chat_client.py ws://localhost:8000/ws")


if __name__ == "__main__":
    main()
