# Local LLM Container

## Requirements
- Docker with NVIDIA GPU support (`nvidia-container-toolkit`) (this could be installed through APT, or whatever package manager do you have)
- NVIDIA GPU with CUDA
- Hugging Face account & token

## Setup
1. Edit `start.sh` and replace the token:
   ```bash
   HF_TOKEN="your_huggingface_token_here"
   ```

2. Run:
   ```bash
   ./start.sh
   ```

## Get HF Token
Create one at: https://huggingface.co/settings/tokens
