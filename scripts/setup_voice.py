# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
# ]
# ///
"""Upload voice recordings to the orchestrator and enable voice cloning.

Calls the orchestrator's /voice/* REST endpoints — no SSH, no SCP.
Works through any port mapping (direct or tunnel) as long as port 8000
is reachable.

Usage:
    uv run scripts/setup_voice.py
    uv run scripts/setup_voice.py --url http://localhost:8000
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
    """Get current voice cloning status.

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


def upload_reference(base_url: str, wav_path: Path, transcript: str) -> bool:
    """Upload a single voice reference via the orchestrator.

    Args:
        base_url: Base URL of the orchestrator.
        wav_path: Path to WAV audio file.
        transcript: Text spoken in the audio.

    Returns:
        True if upload succeeded.
    """
    audio_bytes = wav_path.read_bytes()
    size_kb = len(audio_bytes) / 1024

    resp = httpx.post(
        f"{base_url}/voice/reference",
        files={"audio": (wav_path.name, audio_bytes, "audio/wav")},
        data={"text": transcript},
        timeout=60.0,
    )
    if resp.status_code == 200:
        data = resp.json()
        logger.info(
            "  Uploaded %s (%.0f KB, %d chars) — %d total refs",
            wav_path.name, size_kb, len(transcript), data.get("reference_count", "?"),
        )
        return True
    else:
        logger.error(
            "  Upload %s failed (%d): %s",
            wav_path.name, resp.status_code, resp.text[:500],
        )
        return False


def enable_voice(base_url: str) -> bool:
    """Enable voice cloning and run warmup.

    Args:
        base_url: Base URL of the orchestrator.

    Returns:
        True if enabled successfully.
    """
    logger.info("Enabling voice cloning + warmup ...")
    resp = httpx.post(f"{base_url}/voice/enable", timeout=180.0)
    data = resp.json()
    if data.get("enabled"):
        logger.info(
            "  Voice cloning ON (%d refs, warmup=%s)",
            data.get("reference_count", 0), data.get("warmup", "?"),
        )
        return True
    else:
        logger.error("  Enable failed: %s", data)
        return False


def disable_voice(base_url: str) -> bool:
    """Disable voice cloning.

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
    """Upload voice recordings and enable cloning via orchestrator endpoints."""
    parser = argparse.ArgumentParser(
        description="Upload voice recordings and enable voice cloning",
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Orchestrator URL (default: {DEFAULT_URL})",
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
            print(f"References loaded: {status['reference_count']}")
        return

    # Disable mode
    if args.disable:
        disable_voice(args.url)
        return

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

        # Upload each recording
        success_count = 0
        for wav_path, transcript in pairs:
            if upload_reference(args.url, wav_path, transcript):
                success_count += 1

        if success_count == 0:
            logger.error("All uploads failed")
            sys.exit(1)

        logger.info("Uploaded %d/%d references", success_count, len(pairs))

    # Enable voice cloning + warmup
    if not enable_voice(args.url):
        sys.exit(1)

    logger.info("")
    logger.info("Voice cloning setup complete!")
    logger.info("  Test: cd client && uv run python src/chat_client.py ws://localhost:8000/ws")


if __name__ == "__main__":
    main()
