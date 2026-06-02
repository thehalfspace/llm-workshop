#!/bin/bash
set -e


# This instruction is on how to build the docker image on your local computer where you have root access.
# Once it is pushed, you can pull it on oscar using apptainer pull.

# Configuration
IMAGE_NAME="docker.io/prithworms/hf_phi3_demo_oscar"  #"your-registry/oscar-hf-demo"
TAG="v0.1"
PLATFORMS="linux/amd64" # or linux/arm64

echo "=== Setting up Docker Buildx ==="
# Create a new builder instance for multi-arch builds if it doesn't exist
docker buildx create --name multiarch-builder --use 2>/dev/null || docker buildx use multiarch-builder


echo "=== Building multi-arch image ==="
docker buildx build \
    --platform ${PLATFORMS} \
    --tag ${IMAGE_NAME}:${TAG} \
    -f Dockerfile \
    --push \
    .

echo "=== Build complete ==="
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "Platforms: ${PLATFORMS}"

