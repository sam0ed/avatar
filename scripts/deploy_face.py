# /// script
# requires-python = ">=3.11"
# ///
"""Deploy Stage 4 (LLM + TTS + Face Animation + Orchestrator) to Vast.ai.

Same approach as deploy_stage2.py but with FACE_ENABLED=true, port 8002
for MuseTalk, and extra disk for face models (~7GB) + venv (~3GB).

Uses ghcr.io/sam0ed/avatar-stage4, built by GitHub Actions from
docker/Dockerfile.stage4.  The image bakes system deps, the LLM venv and
the MuseTalk checkout + venv, so the onstart-cmd only clones the repo,
copies configs/code and runs the entrypoint (model download + supervisord).

Usage:
    HF_TOKEN=hf_xxx uv run scripts/deploy_face.py
    HF_TOKEN=hf_xxx uv run scripts/deploy_face.py --offer 12345678

Notes:
    - Repo must be public on GitHub (code is cloned at boot).
    - First boot pulls the image then downloads ~16GB of model weights
      (LLM ~5.8GB + TTS ~3.6GB + MuseTalk ~4GB).  Weights are not baked in;
      hf_transfer/hf_xet make those downloads parallel.
    - HF_TOKEN required for gated TTS model (env var or .env file).
    - Ports: 8000 (WebSocket), 8001 (LLM API), 8080 (TTS API), 8002 (MuseTalk).
    - GPU budget: LLM ~5.5GB + TTS ~5GB + MuseTalk ~2GB = ~12.5GB / 24GB.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE = "ghcr.io/sam0ed/avatar-stage4:latest"
GITHUB_REPO = "https://github.com/sam0ed/avatar.git"
DISK = "120"  # Extra for MuseTalk models + venv
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


def build_onstart_cmd() -> str:
    """Build the onstart command that sets up the container at boot.

    Steps:
        1. Git clone the public repo (shallow, saves time).
        2. Copy configs (supervisord, entrypoint) and orchestrator code.
        3. Install orchestrator Python deps via uv.
        4. Run entrypoint (downloads models, starts supervisord).

    Returns:
        Shell command string (well under Vast.ai 4048 limit).
    """
    steps = [
        # 0. Fix SSH permissions (Vast.ai creates authorized_keys with wrong modes)
        "chmod 700 /root/.ssh 2>/dev/null; chmod 600 /root/.ssh/authorized_keys 2>/dev/null; true",
        # 1. Clone repo (system deps, both venvs and MuseTalk are baked into the image)
        f"git clone --depth 1 {GITHUB_REPO} /tmp/av",
        # 2. Copy configs + code
        "cp /tmp/av/docker/supervisord.conf /etc/supervisor/conf.d/avatar.conf"
        " && cp /tmp/av/docker/entrypoint_stage2.sh /app/"
        " && chmod +x /app/entrypoint_stage2.sh"
        " && mkdir -p /app/orchestrator"
        " && cp -r /tmp/av/server/src /app/orchestrator/"
        " && cp /tmp/av/server/pyproject.toml /app/orchestrator/",
        # 3. Install orchestrator deps
        "cd /app/orchestrator && uv lock && uv sync --no-dev",
        # 4. Run entrypoint (model download + supervisord)
        "cd /app && bash /app/entrypoint_stage2.sh",
    ]
    return " && ".join(steps)


def search_offers(gpus: int) -> str | None:
    """Search for the cheapest RTX 4090 offer with enough disk for face models."""
    result = subprocess.run(
        [
            "vastai", "search", "offers",
            f"gpu_name=RTX_4090 num_gpus={gpus} reliability>0.95 disk_space>=120"
            " inet_down>=700 disk_bw>=2000",
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
            f"disk_bw={o.get('disk_bw', 0):.0f}MB/s  "
            f"{o.get('geolocation', 'Unknown'):20s}  "
            f"R={o.get('reliability2', 0) * 100:.0f}%"
        )

    best = offers[0]
    print(f"\nSelected: {best['id']} (${best['dph_total']:.3f}/hr, {best.get('geolocation', '')})")
    return str(best["id"])


def main() -> None:
    """Deploy Stage 4 (with face animation) to Vast.ai."""
    parser = argparse.ArgumentParser(description="Deploy Stage 4 (Face Animation) to Vast.ai")
    parser.add_argument("--offer", help="Vast.ai offer ID (skips search)")
    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Deploy with FACE_ENABLED=false (skips MuseTalk weights; isolates the audio pipeline)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPUs per instance; the entrypoint auto-assigns services per GPU count",
    )
    args = parser.parse_args()

    hf_token = get_hf_token()
    if not hf_token:
        print("ERROR: Set HF_TOKEN env var or add HF_TOKEN=... to .env", file=sys.stderr)
        sys.exit(1)

    # --- Find or use offer ---
    offer_id = args.offer or search_offers(args.gpus)
    if not offer_id:
        sys.exit(1)

    # --- Create instance ---
    onstart_cmd = build_onstart_cmd()
    face_enabled = "false" if args.no_face else "true"
    env_flags = (
        f"-e HF_TOKEN={hf_token}"
        f" -e FACE_ENABLED={face_enabled}"
        " -p 8000:8000 -p 8001:8001 -p 8080:8080 -p 8002:8002"
    )

    print(f"\nOnstart command ({len(onstart_cmd)} chars, limit 4048):")
    print(onstart_cmd)

    cmd = [
        "vastai", "create", "instance", offer_id,
        "--image", IMAGE,
        "--disk", DISK,
        "--direct",
        "--env", env_flags,
        "--onstart-cmd", onstart_cmd,
    ]

    print(f"\nCreating instance with image {IMAGE} (FACE_ENABLED={face_enabled})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    contract = re.search(r"['\"]new_contract['\"]:\s*(\d+)", result.stdout)
    if result.returncode != 0 or contract is None:
        print("ERROR: Instance creation failed.", file=sys.stderr)
        sys.exit(1)

    instance_id = contract.group(1)
    if not re.search(r"['\"]success['\"]:\s*[Tt]rue", result.stdout):
        print(
            f"WARNING: Vast.ai reported success=False but returned contract {instance_id}. "
            "Verify with 'vastai show instances' before assuming the deploy failed.",
            file=sys.stderr,
        )

    print(f"Instance {instance_id} created. Image pull + model download.")
    print("\nAfter instance starts:")
    print(f"  vastai show instances --raw")
    print(f"  vastai ssh-url {instance_id}")
    print("  # Health:    curl http://<ip>:8000/health")
    print(f"  # Face setup: uv run scripts/setup_face.py --url http://<ip>:8000")
    print("  # Voice:     cd client && uv run python src/face_voice_client.py ws://<ip>:8000/ws")
    print(f"  # Destroy:   vastai destroy instance {instance_id} -y")


if __name__ == "__main__":
    main()
