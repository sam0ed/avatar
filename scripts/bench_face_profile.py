"""Per-stage MuseTalk throughput profile, run on the server over localhost.

Feeds silent PCM in orchestrator-sized chunks and aggregates the per-stage
timings the face service now returns with every /feed response. Stdlib only,
so it runs with the container's plain python3:

    python3 bench_face_profile.py                    # 20 feeds of 0.3s
    python3 bench_face_profile.py --chunk-s 1.0 --feeds 8
"""

import argparse
import json
import time
import urllib.request

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
        print(
            f"feed {index:2d}: {result.get('processing_ms', 0):7.1f}ms "
            f"frames={len(result.get('frames', []))} "
            + " ".join(f"{k}={v:.0f}" for k, v in sorted(result.get("profile", {}).items()))
        )
    wall_s = time.perf_counter() - wall_started
    _post(f"/session/{session_id}/end", b"", "application/octet-stream")

    audio_s = args.feeds * args.chunk_s
    print("=" * 72)
    print(f"chunk={args.chunk_s}s feeds={args.feeds} audio={audio_s:.1f}s")
    print(f"frames={frames} wall={wall_s:.2f}s -> {frames / wall_s:.1f} fps (need {REALTIME_FPS})")
    staged = sum(stage_totals.values())
    for stage, total in sorted(stage_totals.items(), key=lambda item: -item[1]):
        print(
            f"  {stage:12s} {total:8.1f}ms total  {total / max(frames, 1):6.2f}ms/frame  "
            f"{100 * total / staged:5.1f}% of staged"
        )
    print(f"  staged sum   {staged:8.1f}ms of {wall_s * 1000:.1f}ms wall "
          f"({100 * staged / (wall_s * 1000):.1f}% attributed)")


if __name__ == "__main__":
    main()
