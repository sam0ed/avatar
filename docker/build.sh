#!/usr/bin/env bash
# Build and push the avatar server Docker image.
# Usage: ./docker/build.sh [tag]
#
# Must be run from the repo root (avatar/).

set -euo pipefail

TAG="${1:-latest}"
IMAGE_NAME="avatar-server"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
docker build \
    -f docker/Dockerfile \
    -t "${IMAGE_NAME}:${TAG}" \
    .

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"
echo ""
echo "To push to Docker Hub:"
echo "  docker tag ${IMAGE_NAME}:${TAG} <your-dockerhub-user>/${IMAGE_NAME}:${TAG}"
echo "  docker push <your-dockerhub-user>/${IMAGE_NAME}:${TAG}"
echo ""
echo "To run locally:"
echo "  docker run --gpus all -p 8000:8000 ${IMAGE_NAME}:${TAG}"
