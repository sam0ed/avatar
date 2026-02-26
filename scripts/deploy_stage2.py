# /// script
# requires-python = ">=3.11"
# ///
"""Deploy Stage 2 (LLM + TTS + Orchestrator) to Vast.ai.

Builds a Docker image with all code and deps baked in, pushes to
Docker Hub, then creates a Vast.ai instance.  The onstart-cmd just
runs the entrypoint (model download + supervisord start).

Usage:
    HF_TOKEN=hf_xxx uv run scripts/deploy_stage2.py
    HF_TOKEN=hf_xxx uv run scripts/deploy_stage2.py --offer 12345678
    uv run scripts/deploy_stage2.py --skip-build --offer 12345678

Notes:
    - First build pulls the ~5GB base image (cached afterwards).
    - First boot downloads ~9GB of model weights (LLM + TTS).
    - HF_TOKEN required for gated TTS model (env var or .env file).
    - Ports: 8000 (WebSocket), 8001 (LLM API), 8080 (TTS API).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCKER_IMAGE = "sam0ed/avatar-server:stage2"
DOCKERFILE = "docker/Dockerfile.stage2"
DISK = "80"
BLOCKED_REGIONS = {"CN", "RU"}


def get_hf_token() -> str:
    """Read HF token from env var or .env file."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip().startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip('"').strip("'")
                    break
    return token


def build_and_push() -> bool:
    """Build Docker image and push to Docker Hub.

    Returns:
        True on success, False on failure.
    """
    print(f"Building {DOCKER_IMAGE} ...")
    result = subprocess.run(
        [
            "docker", "build",
            "--provenance=false",
            "-f", DOCKERFILE, "-t", DOCKER_IMAGE, ".",
        ],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print("ERROR: Docker build failed.", file=sys.stderr)
        return False

    print(f"\nPushing {DOCKER_IMAGE} ...")
    result = subprocess.run(["docker", "push", DOCKER_IMAGE])
    if result.returncode != 0:
        print("ERROR: Docker push failed. Run 'docker login' first?", file=sys.stderr)
        return False

    return True


def search_offers() -> str | None:
    """Search for cheapest RTX 4090 offer."""
    result = subprocess.run(
        [
            "vastai", "search", "offers",
            "gpu_name=RTX_4090 num_gpus=1 reliability>0.95 disk_space>=80",
            "-o", "dph+", "--limit", "20", "--raw",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Search failed: {result.stderr}", file=sys.stderr)
        return None

    offers = json.loads(result.stdout)
    offers = [
        o for o in offers
        if o.get("geolocation", "").split(",")[-1].strip() not in BLOCKED_REGIONS
    ]
    if not offers:
        print("No offers found (after filtering blocked regions)!", file=sys.stderr)
        return None

    print("\nAvailable RTX 4090 offers (excluding blocked regions):")
    for o in offers[:5]:
        ram_gb = o.get("cpu_ram", 0) / 1024
        print(
            f"  ID {o['id']:>10}  ${o['dph_total']:.3f}/hr  "
            f"RAM={ram_gb:.0f}GB  "
            f"{o.get('geolocation', 'Unknown'):20s}  "
            f"R={o.get('reliability2', 0) * 100:.0f}%"
        )

    best = offers[0]
    print(f"\nSelected: {best['id']} (${best['dph_total']:.3f}/hr, {best.get('geolocation', '')})")
    return str(best["id"])


def main() -> None:
    """Deploy Stage 2 to Vast.ai."""
    parser = argparse.ArgumentParser(description="Deploy Stage 2 to Vast.ai")
    parser.add_argument("--offer", help="Vast.ai offer ID (skips search)")
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip Docker build+push (use previously pushed image)",
    )
    args = parser.parse_args()

    hf_token = get_hf_token()
    if not hf_token:
        print("ERROR: Set HF_TOKEN env var or add HF_TOKEN=... to .env", file=sys.stderr)
        sys.exit(1)

    # --- Build & push Docker image ---
    if not args.skip_build:
        if not build_and_push():
            sys.exit(1)
    else:
        print(f"Skipping build, using {DOCKER_IMAGE}")

    # --- Find or use offer ---
    offer_id = args.offer or search_offers()
    if not offer_id:
        sys.exit(1)

    # --- Create instance ---
    onstart_cmd = "bash /app/entrypoint_stage2.sh"
    env_flags = f"-e HF_TOKEN={hf_token} -p 8000:8000 -p 8001:8001 -p 8080:8080"

    cmd = [
        "vastai", "create", "instance", offer_id,
        "--image", DOCKER_IMAGE,
        "--disk", DISK,
        "--direct",
        "--env", env_flags,
        "--onstart-cmd", onstart_cmd,
    ]

    print(f"\nCreating instance with image {DOCKER_IMAGE}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode == 0 and "success" in result.stdout.lower():
        print("Instance created. Model download will take ~10-15 min on first boot.")
        print("\nAfter instance starts:")
        print("  vastai show instances")
        print("  vastai ssh-url <INSTANCE_ID>")
        print("  # Health:    curl http://<ip>:8000/health")
        print("  # Chat:      cd client && uv run python src/chat_client.py ws://<ip>:8000/ws")
        print("  # Logs:      ssh -p <port> root@<host> 'supervisorctl status'")
        print("  # Destroy:   vastai destroy instance <INSTANCE_ID>")
    else:
        print("ERROR: Instance creation failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
