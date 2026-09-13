#!/usr/bin/env bash
# Two-node tensor-parallel PoC: local CUDA + remote RPC, NCCL over RoCE.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROOT}/build/bin"
GGUF="${QWEN_GGUF:-/home/maci/qwen3.8-flash-next/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
REMOTE="${REMOTE_RPC:-10.0.1.2:50052}"
PROMPT="${PROMPT:-What is 2+2? Answer with just the number.}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "${BIN}/llama-cli" \
    -m "${GGUF}" \
    --rpc "${REMOTE}" \
    --split-mode tensor \
    -ngl 999 \
    -fa on \
    -ctk f16 -ctv f16 \
    --ctx-size "${CTX:-2048}" \
    -n "${N_PREDICT:-32}" \
    --no-warmup \
    -p "${PROMPT}" \
    "$@"
