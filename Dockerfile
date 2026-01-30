FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    git ca-certificates curl vim \
    && rm -rf /var/lib/apt/lists/*

# Create user and home dir, then app dir and chown it
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /home/appuser/app && \
    chown -R appuser:appuser /home/appuser

USER appuser
WORKDIR /home/appuser/app

# Now this will succeed
RUN python3 -m venv .venv
ENV PATH="/home/appuser/app/.venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# Use PyTorch 2.5 + CUDA 12.1 wheels
RUN pip install \
    "torch==2.5.1+cu121" \
    "torchvision==0.20.1+cu121" \
    "torchaudio==2.5.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
    "transformers[torch]" \
    accelerate \
    huggingface_hub \
    sentencepiece \
    safetensors \
    einops

RUN python - << 'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA NOT AVAILABLE IN CONTAINER")
EOF

COPY run_llama_cli.py ./run_llama_cli.py

ENTRYPOINT ["python", "run_llama_cli.py"]
