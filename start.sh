#!/bin/bash

# Hugging Face token


docker run --rm -it \
  --gpus all \
  -e HF_TOKEN \
  -e HUGGINGFACE_HUB_TOKEN \
  -e LLAMA_MODEL="unsloth/Llama-3.2-3B-Instruct" \
  -v $HOME/.cache/huggingface:/home/appuser/.cache/huggingface \
  local-llm-for-all:latest
