# syntax=docker/dockerfile:1

# Multi-stage build for opponent-modeling MARL training
#   docker build -t opponent-modeling-marl .
#   docker run --gpus all opponent-modeling-marl python experiments/run_om.py

FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime AS runtime

LABEL org.opencontainers.image.title="opponent-modeling-marl"
LABEL org.opencontainers.image.description="Multi-agent RL with opponent modeling, curriculum learning, and past-self evaluation"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Default command — verify CUDA is available
CMD ["python", "-c", "import torch; print(f'CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"]
