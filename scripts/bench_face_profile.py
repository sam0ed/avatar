# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "structlog>=24",
# ]
# ///
"""Per-stage MuseTalk throughput profile, run on the server over localhost.

Feeds silent PCM in orchestrator-sized chunks and aggregates the per-stage
timings the face service returns with every /feed response:

    uv run scripts/bench_face_profile.py
    uv run scripts/bench_face_profile.py --chunk-s 1.0 --feeds 8
"""

import argparse
import json
import time
import urllib.request

import structlog

logger = structlog.get_logger("avatar.bench.face")

BASE_URL = "http://localhost:8002"
PCM_SAMPLE_RATE = 44100
BYTES_PER_SAMPLE = 2
REALTIME_FPS = 25


def _post(path: str, body: bytes, content_type: str) -> dict:
    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=body, headers={"Content-Type": content_type}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MuseTalk feed throughput")
    parser.add_argument("--chunk-s", type=float, default=0.3)
    parser.add_argument("--feeds", type=int, default=20)
    parser.add_argument("--avatar-id", default="default")
    args = parser.parse_args()

    chunk = b"\x00" * (int(PCM_SAMPLE_RATE * args.chunk_s) * BYTES_PER_SAMPLE)
    session = _post(
        "/session/start",
        f"avatar_id={args.avatar_id}".encode(),
        "application/x-www-form-urlencoded",
    )
    session_id = session["session_id"]

    stage_totals: dict[str, float] = {}
    frames = 0
    wall_started = time.perf_counter()
    for index in range(args.feeds):
        result = _post(
            f"/session/{session_id}/feed", chunk, "application/octet-stream"
        )
        frames += len(result.get("frames", []))
        for stage, ms in result.get("profile", {}).items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + ms
        logger.info(
            "feed",
            index=index,
            processing_ms=result.get("processing_ms", 0),
            frames=len(result.get("frames", [])),
            **{k: round(v) for k, v in result.get("profile", {}).items()},
        )
    wall_s = time.perf_counter() - wall_started
    _post(f"/session/{session_id}/end", b"", "application/octet-stream")

    staged = sum(stage_totals.values())
    logger.info(
        "summary",
        chunk_s=args.chunk_s,
        feeds=args.feeds,
        audio_s=round(args.feeds * args.chunk_s, 1),
        frames=frames,
        wall_s=round(wall_s, 2),
        fps=round(frames / wall_s, 1),
        fps_needed=REALTIME_FPS,
        attributed_pct=round(100 * staged / (wall_s * 1000), 1),
    )
    for stage, total in sorted(stage_totals.items(), key=lambda item: -item[1]):
        logger.info(
            "stage",
            name=stage,
            total_ms=round(total, 1),
            ms_per_frame=round(total / max(frames, 1), 2),
            share_pct=round(100 * total / staged, 1),
        )


if __name__ == "__main__":
    main()
