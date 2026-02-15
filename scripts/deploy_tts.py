# /// script
# requires-python = ">=3.11"
# ///
"""Deploy Fish Speech S1-mini to Vast.ai.

Usage:
    # Search for offers and deploy:
    HF_TOKEN=hf_xxx uv run scripts/deploy_tts.py

    # Deploy to a specific offer:
    HF_TOKEN=hf_xxx uv run scripts/deploy_tts.py --offer 31202338

    # Or put HF_TOKEN in .env file at project root.

Notes:
    - The openaudio-s1-mini model is gated — you must accept the license at
      https://huggingface.co/fishaudio/openaudio-s1-mini and provide a HF token.
    - The Fish Speech Docker image runs as root on Vast.ai (no 'fish' user).
    - Model weights (~3.6GB) are downloaded at startup via onstart-cmd.
    - Server listens on port 8080 inside the container.
"""

import argparse
import json
import os
import subprocess
import sys

IMAGE = "fishaudio/fish-speech:server-cuda"
DISK = "50"
BLOCKED_REGIONS = {"CN", "RU"}


def get_hf_token() -> str:
    """Read HF token from env var or .env file."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_file):
            for line in open(env_file):
                if line.startswith("HF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip('"').strip("'")
                    break
    return token


def search_offers() -> str | None:
    """Search for cheapest RTX 4090 offer."""
    result = subprocess.run(
        [
            "vastai", "search", "offers",
            "gpu_name=RTX_4090 num_gpus=1 reliability>0.95 disk_space>=50",
            "-o", "dph+", "--limit", "20", "--raw",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Search failed: {result.stderr}", file=sys.stderr)
        return None

    offers = json.loads(result.stdout)
    # Filter out blocked regions (China, Russia, etc.)
    offers = [
        o for o in offers
        if o.get("geolocation", "").split(",")[-1].strip() not in BLOCKED_REGIONS
    ]
    if not offers:
        print("No offers found (after filtering blocked regions)!", file=sys.stderr)
        return None

    print("\nAvailable RTX 4090 offers (excluding blocked regions):")
    for o in offers:
        print(
            f"  ID {o['id']:>10}  ${o['dph_total']:.3f}/hr  "
            f"{o.get('geolocation', 'Unknown'):20s}  "
            f"R={o.get('reliability2', 0)*100:.0f}%"
        )

    best = offers[0]
    print(f"\nSelected: {best['id']} (${best['dph_total']:.3f}/hr, {best.get('geolocation', '')})")
    return str(best["id"])


def build_onstart_cmd(hf_token: str) -> str:
    """Build the onstart command for the Vast.ai instance.

    Downloads model weights via huggingface_hub (already in the Fish Speech
    Docker image's uv venv), then starts the API server.
    """
    return (
        f"cd /app && HF_TOKEN={hf_token} uv run python -c \""
        "from huggingface_hub import snapshot_download; "
        "import os; "
        "snapshot_download('fishaudio/openaudio-s1-mini', "
        "local_dir='checkpoints/openaudio-s1-mini', "
        "token=os.environ['HF_TOKEN'])"
        '" && mkdir -p references && bash start_server.sh'
    )


def main() -> None:
    """Deploy Fish Speech S1-mini to Vast.ai."""
    parser = argparse.ArgumentParser(description="Deploy Fish Speech S1-mini to Vast.ai")
    parser.add_argument("--offer", help="Vast.ai offer ID (skips search)")
    args = parser.parse_args()

    hf_token = get_hf_token()
    if not hf_token:
        print("ERROR: Set HF_TOKEN env var or add HF_TOKEN=... to .env", file=sys.stderr)
        sys.exit(1)

    offer_id = args.offer or search_offers()
    if not offer_id:
        sys.exit(1)

    onstart_cmd = build_onstart_cmd(hf_token)

    cmd = [
        "vastai", "create", "instance", offer_id,
        "--image", IMAGE,
        "--disk", DISK,
        "--direct",
        "--env", "-p 8080:8080",
        "--onstart-cmd", onstart_cmd,
    ]

    print(f"\nOnstart command:\n{onstart_cmd}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
