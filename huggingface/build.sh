#!/bin/bash
set -e

# Configuration
IMAGE_NAME="your-registry/oscar-hf-demo"
TAG="latest"
PLATFORMS="linux/amd64,linux/arm64"

echo "=== Setting up Docker Buildx ==="
# Create a new builder instance for multi-arch builds if it doesn't exist
docker buildx create --name multiarch-builder --use 2>/dev/null || docker buildx use multiarch-builder

# Ensure QEMU emulators are installed for cross-platform builds
docker run --rm --privileged tonistiigi/binfmt:latest --install all

echo "=== Inspecting base image platform support ==="
docker buildx imagetools inspect nvcr.io/nvidia/pytorch:26.04-py3

echo "=== Building multi-arch image ==="
docker buildx build \
    --platform ${PLATFORMS} \
    --tag ${IMAGE_NAME}:${TAG} \
    --push \
    .

echo "=== Build complete ==="
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "Platforms: ${PLATFORMS}"

echo "=== Verifying manifest ==="
docker buildx imagetools inspect ${IMAGE_NAME}:${TAG}
