# llama.cpp work for the DGX Spark pair

Our llama.cpp changes for two NVIDIA GB10 machines (DGX Spark, sm_121), kept as ordered patches
against a pinned upstream commit so they can be reviewed and applied without carrying a fork's
history.

The goal for this work: on this rig, for three models, be faster than the other engines.

| target | llama.cpp surface | bar to beat | state |
|---|---|---|---|
| `qwen3.8-flash-next` | `src/models/qwen4exp.cpp`, `conversion/qwen4exp.py` | vLLM TP2: 31.14 tok/s single-stream, **74.28 tok/s aggregate at 8 concurrent**; SGLang ~21 tok/s | single-Spark GGUF works; **measured 27.05 tok/s generation** (see `bench/`) |
| `deepseek-v4-flash-0731` | two-node NCCL/RPC: `tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `common/speculative.*`, `src/models/deepseek4.cpp`, `src/models/dflash.cpp` | vLLM on this rig: 63.1 / 112.1 / 140.9 / 156.0 tok/s at c1/c3/c5/c6 | in the patches below; needs the 145 GB GGUF over two nodes |
| `glm5.3-flash` | none of ours: upstream PR [ggml-org/llama.cpp#27754](https://github.com/ggml-org/llama.cpp/pull/27754) adds `glm5next` | nothing measured | no weights on the rig yet |

So the honest position: single-Spark llama.cpp already matches vLLM one-to-one and is far behind it
at concurrency, and that is what the two-node tensor-parallel work is for.

## Base

Upstream llama.cpp commit **`304665fe7`** ("Add IQ type handling for MoE (#28476)"). That is the base
the patches apply to. They were also merged onto current master (`790cf51aa`) and built there.

## Contents

```
patches/    our commits, in order, each a normal git patch with authorship
scripts/    start_rpc_spark.sh, run_qwen38_tp2.sh, build_spark.sh
docs/       targets.md - the three-model mission, bars and blockers
bench/      raw llama-bench output, with the machine and flags
```

| patch | what it is |
|---|---|
| `0001` | register the qwen4exp NextN tensors in gguf-py |
| `0002` | the accumulated work: two-node NCCL/RPC transport, DFlash2 draft plumbing, Qwen4Exp model and converter, and the three scripts |
| `0003` | the workspace scope note and the three-model target docs |
| `0004` | the Qwen4Exp line-reconciliation record |

## Applying and building

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 304665fe7
git am /path/to/patches/*.patch

cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-cli llama-server llama-bench
```

## Measured

```
qwen4exp A3B Q4_K - Medium | 103.68 GiB | 176.94 B | CUDA | ngl 99
  pp256   633.26 +- 93.53 t/s
  tg64     27.05 +-  1.33 t/s      (2 reps, 2026-09-13, one GB10)
```

Raw output in `bench/`. Two notes for reproducing it: this build's `llama-bench` **rejects `-c`** and
prints its usage instead, and the `MTP-*` GGUF set is a second copy of the weights, so serving with
the draft head needs both nodes.

## Not in this repo

- **GLM-5-Next support**, which is upstream PR #27754 (Unsloth), not ours. We build with it merged
  locally but do not redistribute it.
- **Weights.** GGUF quants are large; nothing here includes them.
- **The carried branches that cannot be rebased.** `pr27836-qwen4exp-mtp` (`1d8de7c1b`),
  `pr27858-dflash2-tensor` (`3286b033c`), `fix/28003-rdna3-guard` (`41686b31d`) and the 33-commit
  Spark Qwen4Exp series live on a lineage that diverges from upstream: `git rev-list --count
  origin/master..pr27836-qwen4exp-mtp` is 10,667, so a rebase replays ten thousand commits. Their
  content has to be re-expressed as patches on the current line before it can appear here.

## Licence

MIT, the same as llama.cpp, which these changes are derived from. See `LICENSE`.
