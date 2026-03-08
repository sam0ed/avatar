# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27",
# ]
# ///
"""Upload a reference video and enable face animation on the avatar server.

Sends a video file to the orchestrator's /face/prepare endpoint, then
enables face animation via /face/enable.

Usage:
    uv run scripts/setup_face.py --video path/to/reference.mp4
    uv run scripts/setup_face.py --video path/to/reference.mp4 --url http://HOST:8000
    uv run scripts/setup_face.py --video path/to/reference.mp4 --avatar-id my-avatar
    uv run scripts/setup_face.py --disable
    uv run scripts/setup_face.py --status
    uv run scripts/setup_face.py --list
"""

import argparse
import logging
import sys
from pathlib import Path

import httpx

logger = logging.getLogger("avatar.setup_face")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

DEFAULT_URL = "http://localhost:8000"


def check_health(base_url: str) -> bool:
    """Check orchestrator health and face service availability."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=10.0)
        data = resp.json()
        logger.info("Orchestrator health: %s", data)
        face_status = data.get("face", "unavailable")
        if face_status != "ok":
            logger.warning("Face service status: %s", face_status)
        return data.get("status") in ("ok", "degraded")
    except Exception as e:
        logger.error("Cannot reach orchestrator at %s: %s", base_url, e)
        return False


def get_face_status(base_url: str) -> dict | None:
    """Get current face animation status."""
    try:
        resp = httpx.get(f"{base_url}/face/status", timeout=10.0)
        return resp.json()
    except Exception as e:
        logger.error("Failed to get face status: %s", e)
        return None


def list_avatars(base_url: str) -> list[str]:
    """List available avatar IDs."""
    try:
        resp = httpx.get(f"{base_url}/face/avatars", timeout=10.0)
        data = resp.json()
        return data.get("avatars", [])
    except Exception as e:
        logger.error("Failed to list avatars: %s", e)
        return []


def prepare_avatar(base_url: str, video_path: Path, avatar_id: str | None = None) -> dict | None:
    """Upload a reference video for avatar preparation."""
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        return None

    logger.info("Uploading %s (%.1f MB)...", video_path.name, video_path.stat().st_size / 1e6)

    files = {"video": (video_path.name, video_path.read_bytes(), "video/mp4")}
    data = {}
    if avatar_id:
        data["avatar_id"] = avatar_id

    try:
        resp = httpx.post(
            f"{base_url}/face/prepare",
            files=files,
            data=data,
            timeout=120.0,  # Avatar preparation can take time
        )
        result = resp.json()
        if "error" in result:
            logger.error("Prepare failed: %s", result["error"])
            return None
        logger.info("Avatar prepared: %s", result)
        return result
    except Exception as e:
        logger.error("Failed to prepare avatar: %s", e)
        return None


def enable_face(base_url: str, avatar_id: str) -> bool:
    """Enable face animation with the specified avatar."""
    try:
        resp = httpx.post(f"{base_url}/face/enable", params={"avatar_id": avatar_id}, timeout=10.0)
        data = resp.json()
        if data.get("enabled"):
            logger.info("Face animation enabled with avatar '%s'", avatar_id)
            return True
        logger.error("Enable failed: %s", data.get("error", "unknown"))
        return False
    except Exception as e:
        logger.error("Failed to enable face: %s", e)
        return False


def disable_face(base_url: str) -> bool:
    """Disable face animation."""
    try:
        resp = httpx.post(f"{base_url}/face/disable", timeout=10.0)
        data = resp.json()
        logger.info("Face animation disabled")
        return not data.get("enabled", True)
    except Exception as e:
        logger.error("Failed to disable face: %s", e)
        return False


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Setup face animation on avatar server")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Orchestrator URL (default: {DEFAULT_URL})")
    parser.add_argument("--video", type=Path, help="Reference video file to upload")
    parser.add_argument("--avatar-id", help="Custom avatar ID (auto-generated if omitted)")
    parser.add_argument("--enable", help="Enable face with this avatar ID (skip upload)")
    parser.add_argument("--disable", action="store_true", help="Disable face animation")
    parser.add_argument("--status", action="store_true", help="Show face animation status")
    parser.add_argument("--list", action="store_true", help="List available avatars")
    args = parser.parse_args()

    if not check_health(args.url):
        sys.exit(1)

    if args.status:
        status = get_face_status(args.url)
        if status:
            print(f"Face enabled:    {status.get('face_enabled', False)}")
            print(f"Face available:  {status.get('face_available', False)}")
            print(f"Active avatar:   {status.get('active_avatar_id', 'None')}")
        sys.exit(0)

    if args.list:
        avatars = list_avatars(args.url)
        if avatars:
            print("Available avatars:")
            for a in avatars:
                print(f"  - {a}")
        else:
            print("No avatars prepared yet.")
        sys.exit(0)

    if args.disable:
        disable_face(args.url)
        sys.exit(0)

    if args.enable:
        success = enable_face(args.url, args.enable)
        sys.exit(0 if success else 1)

    if args.video:
        result = prepare_avatar(args.url, args.video, args.avatar_id)
        if result is None:
            sys.exit(1)
        # Auto-enable after successful preparation
        avatar_id = result.get("avatar_id") or args.avatar_id
        if avatar_id:
            enable_face(args.url, avatar_id)
        sys.exit(0)

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
