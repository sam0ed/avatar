# /// script
# requires-python = ">=3.11"
# ///
"""Deploy Stage 2 (LLM + TTS + Orchestrator) to Vast.ai.

Uses the off-the-shelf fishaudio/fish-speech:server-cuda image.
The onstart-cmd git-clones the public repo, installs supervisor +
llama-cpp-python, copies configs/code, then runs the entrypoint
(model download + supervisord start).  No custom Docker image needed.

Usage:
    HF_TOKEN=hf_xxx uv run scripts/deploy_stage2.py
    HF_TOKEN=hf_xxx uv run scripts/deploy_stage2.py --offer 12345678

Notes:
    - Repo must be public on GitHub (code is cloned at boot).
    - First boot installs deps (~2 min) + downloads ~9GB of model weights.
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
IMAGE = "fishaudio/fish-speech:server-cuda"
GITHUB_REPO = "https://github.com/sam0ed/avatar.git"
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


def build_onstart_cmd() -> str:
    """Build the onstart command that sets up the container at boot.

    Steps:
        1. Install supervisor + git via apt.
        2. Create isolated venv for LLM server, install llama-cpp-python.
        3. Git clone the public repo (shallow, saves time).
        4. Copy configs (supervisord, entrypoint) and orchestrator code.
        5. Install orchestrator Python deps via uv.
        6. Run entrypoint (downloads models, starts supervisord).

    Returns:
        Shell command string (~750 chars, well under Vast.ai 4048 limit).
    """
    steps = [
        # 0. Fix SSH permissions (Vast.ai creates authorized_keys with wrong modes)
        "chmod 700 /root/.ssh 2>/dev/null; chmod 600 /root/.ssh/authorized_keys 2>/dev/null; true",
        # 1. System deps
        "apt-get update -qq && apt-get install -y -qq supervisor git",
        # 2. LLM venv + deps (pre-built CUDA wheel, no nvcc needed)
        "uv venv /opt/llm-venv"
        " && uv pip install --python /opt/llm-venv/bin/python"
        " 'llama-cpp-python[server]>=0.3,<1' 'huggingface_hub>=0.25,<1'"
        " --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu126",
        # 3. Clone repo
        f"git clone --depth 1 {GITHUB_REPO} /tmp/av",
        # 4. Copy configs + code
        "cp /tmp/av/docker/supervisord.conf /etc/supervisor/conf.d/avatar.conf"
        " && cp /tmp/av/docker/entrypoint_stage2.sh /app/"
        " && chmod +x /app/entrypoint_stage2.sh"
        " && mkdir -p /app/orchestrator"
        " && cp -r /tmp/av/server/src /app/orchestrator/"
        " && cp /tmp/av/server/pyproject.toml /app/orchestrator/",
        # 5. Install orchestrator deps
        "cd /app/orchestrator && uv lock && uv sync --no-dev",
        # 6. Run entrypoint (model download + supervisord)
        "cd /app && bash /app/entrypoint_stage2.sh",
    ]
    return " && ".join(steps)


def search_offers() -> str | None:
    """Search for cheapest RTX 4090 offer."""
    result = subprocess.run(
        [
            "vastai", "search", "offers",
            "gpu_name=RTX_4090 num_gpus=1 reliability>0.95 disk_space>=80 inet_down>=700",
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
    args = parser.parse_args()

    hf_token = get_hf_token()
    if not hf_token:
        print("ERROR: Set HF_TOKEN env var or add HF_TOKEN=... to .env", file=sys.stderr)
        sys.exit(1)

    # --- Find or use offer ---
    offer_id = args.offer or search_offers()
    if not offer_id:
        sys.exit(1)

    # --- Create instance ---
    onstart_cmd = build_onstart_cmd()
    env_flags = f"-e HF_TOKEN={hf_token} -p 8000:8000 -p 8001:8001 -p 8080:8080"

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

    print(f"\nCreating instance with image {IMAGE}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode == 0 and "success" in result.stdout.lower():
        print("Instance created. Dep install (~2 min) + model download (~10-15 min).")
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
