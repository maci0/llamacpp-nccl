#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
exec "${ROOT}/build/bin/ggml-rpc-server" -H 0.0.0.0 -p 50052 -c
