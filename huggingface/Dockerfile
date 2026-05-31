# Multi-architecture NGC PyTorch base image
# Supports linux/amd64 and linux/arm64
FROM nvcr.io/nvidia/pytorch:25.08-py3

LABEL maintainer="oscar-llm-workshop"
LABEL description="Multi-arch Phi-3 inference demo with W&B"

# Set working directory
WORKDIR /workspace/oscar-demo

# Install system dependencies (architecture-agnostic)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY phi3_inference.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WANDB_ANONYMOUS=allow

# Default command runs the inference demo
