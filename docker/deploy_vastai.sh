#!/usr/bin/env bash
# Deploy avatar server to Vast.ai.
# Prerequisites:
#   - vastai CLI installed: pip install vastai
#   - API key set: vastai set api-key <key>
#   - Docker image pushed to Docker Hub
#
# Usage: ./docker/deploy_vastai.sh <docker_image> [disk_gb]

set -euo pipefail

DOCKER_IMAGE="${1:?Usage: deploy_vastai.sh <docker_image> [disk_gb]}"
DISK_GB="${2:-50}"
PORT=8000

echo "=== Vast.ai Deployment ==="
echo "Image: ${DOCKER_IMAGE}"
echo "Disk:  ${DISK_GB} GB"
echo ""

# Search for a suitable RTX 4090 instance
echo "Searching for RTX 4090 instances..."
vastai search offers \
    'gpu_name=RTX_4090 num_gpus=1 reliability>0.98 inet_down>200 inet_up>200' \
    -o 'dph+' \
    --limit 5

echo ""
echo "To create an instance, run:"
echo "  vastai create instance <INSTANCE_ID> \\"
echo "    --image ${DOCKER_IMAGE} \\"
echo "    --disk ${DISK_GB} \\"
echo "    --env '-p ${PORT}:${PORT}' \\"
echo "    --onstart-cmd 'uv run uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT}'"
echo ""
echo "After instance starts:"
echo "  vastai show instances"
echo "  vastai ssh-url <INSTANCE_ID>"
echo "  # Test: curl http://<instance_ip>:${PORT}/health"
echo "  # Test: uv run python client/src/connectivity_test.py ws://<instance_ip>:${PORT}/ws"
