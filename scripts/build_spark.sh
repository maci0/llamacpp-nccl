#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/cuda/bin:${PATH}"
export NCCL_ROOT=/usr
cmake -B build -DGGML_CUDA=ON -DGGML_RPC=ON -DGGML_CUDA_NCCL=ON \
    -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=121 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --config Release -j"$(nproc)" --target llama-cli ggml-rpc-server
echo BUILD_OK
