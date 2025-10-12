#!/usr/bin/env bash
set -euo pipefail

echo "==> docker version"
if ! docker --version; then
  echo "docker not found on PATH." >&2
  exit 1
fi

echo "==> nvidia-smi (host)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found; ensure NVIDIA drivers are installed." >&2
fi

echo "==> nvidia-smi inside CUDA container"
if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi; then
  echo "CUDA 12.2 base image test failed; trying CUDA 11.8..." >&2
  docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
fi

echo "==> torch CUDA availability inside vLLM image"
docker run --rm --gpus all vllm/vllm-openai:0.5.5 \
  python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
PY

echo "All checks completed."

