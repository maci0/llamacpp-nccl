# Targets

This repo is the llama.cpp home for the DGX Spark performance work. The mission is to beat vLLM,
SGLang and the other engines on the same rig for three models:

| # | Model | llama.cpp surface here | The bar to beat | State |
|---|---|---|---|---|
| 1 | `deepseek-v4-flash-0731` | two-node NCCL/RPC: `tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `common/speculative.*`, `src/models/deepseek4.cpp`, `src/models/dflash.cpp` | vLLM on the same 2 Sparks, guarded, c1/c3/c5/c6 = 63.1 / 112.1 / 140.9 / 156.0 tok/s (the anemll reference), our own vLLM 42.0 / 79.3 / 109.1 / 122.8 | two-node over RPC: pp64 164.02 +- 9.34, tg32 15.08 +- 1.13 tok/s. No speculative decoding yet |
| 2 | `glm5.3-flash` | none of ours: upstream PR [ggml-org/llama.cpp#27754](https://github.com/ggml-org/llama.cpp/pull/27754) adds the `glm5next` arch | SGLang day-0 NVFP4 TP2: 24.7-30.3 tok/s on two Sparks | measured: 19.74 +- 0.12 tok/s one Spark, 17.53 +- 0.03 two-node; the split costs 11 % and buys nothing |
| 3 | `qwen3.8-flash-next` | `src/models/qwen4exp.cpp`, `conversion/qwen4exp.py`; GGUF MTP serving | vLLM TP2 on the same 2 Sparks: 31.14 tok/s single-stream, **74.28 tok/s aggregate at 8 concurrent**; SGLang ~21 tok/s. Our own llama.cpp bar: single-Spark GGUF MTP 4-bit, ~27 to 32 tok/s | GGUF single-Spark serving DONE, measured 27.05 +- 1.33 tok/s; 2-node is the open job |

Read the goal as: for each model, make llama.cpp on this rig faster than the best other engine, measured
the same way. The vLLM numbers above come from the vLLM project's guarded protocol (`drive-median.sh
<tag> 3 512 1 3 5 6`, chat, seed 1234, three passes, median), which lives in the sibling project, not
here.

## Per target

### 1. deepseek-v4-flash-0731

Two-node tensor parallelism over NCCL/RPC is the work. What exists:

- the transport and draft plumbing, committed as `580205750` on `unmerged-prs` and published as
  `patches/0001`-`0005` against upstream `304665fe7`;
- `scripts/start_rpc_spark.sh` for bringing up the peer;
- a working two-node run over RDMA, and no speculative decoding wired into it yet, which is the gap
  to the vLLM bar above.

### 2. glm5.3-flash

Released 2026-08-26, so the research is day-0/1 and the source notes flag unverified claims as such.
320B total / 18B active, 45 layers, hybrid 34 KDA linear-attention + 11 NoPE sparse-MLA layers, MLA KV
(`kv_lora_rank` 512), 1M native context, one MTP draft layer, multimodal.

Measured on one Spark from the Unsloth `UD-IQ1_S` GGUF (86.69 GiB): 19.74 +- 0.12 tok/s generation,
166.59 +- 4.93 tok/s prompt. Two-node is 17.53 +- 0.03, so the split costs 11 % for no gain: the model
fits one node and the default layer split is already even. The GLM feasibility note is not kept here:
it lives locally at `/home/maci/Desktop/glm-5.3-flash/README.md`.

### 3. qwen3.8-flash-next

The furthest along of the three. llama.cpp GGUF MTP on one Spark is already serving and is the bar for
the 2-node work; vLLM TP2 reaches 74.28 tok/s aggregate at 8 concurrent, which single-Spark llama.cpp
does not.

Its source notes are not kept here. The upstream home is the public repo
`maci0/qwen3.8-flash-next-spark`, which holds `plan.md` (Plan A SGLang TP2 done, Plan B GGUF 4-bit
llama.cpp done, single Spark), `research.md`, `session-log.md`, `vllm-perf.md`, `sglang-perf.md` and
`sglang-deployment.md`; refresh from there rather than copying them in.

## Consolidation status

Captured here on 2026-09-13:

- **the Spark-only Qwen4Exp history**, fetched as `refs/remotes/spark-qwen38/*`, and kept as the local
  branch `qwen4exp-spark`. Branch `pr-27742` holds **33 feature commits this repo had never seen**,
  dated 2026-08-26/27, which is the whole Qwen4Exp model port series (gguf arch, hparams, text graph
  with GDN and hyper-connections, PLE n-gram embedding, QSA sparse attention, indexer caches, tensor
  parallel). They were unpushed and existed only on spark1. The branch is carried locally and is not
  part of the published patch set.
- **this repo's accumulated work, now committed** as `580205750` on `unmerged-prs`: 23 files, the
  two-node NCCL/RPC transport, DFlash2 draft plumbing and the Qwen4Exp iteration that was in the
  working tree. It had never been committed on that branch.

### The two Qwen4Exp lines: measured, and not mechanically reconcilable

Attempted as a rebase of the Spark series onto our tip. It stops on the **first** commit:
`conversion/qwen4exp.py` is an **add/add** conflict, because both lines independently wrote that
converter for the same model. Rerun as individual cherry-picks of the eight commits that touch no file
we changed, one applied and seven conflicted anyway, because the Spark fork point is two weeks behind
our master.

What the two lines actually contain:

| | ours (`unmerged-prs`) | the Spark series (`qwen4exp-spark`) |
|---|---|---|
| Q split granularity for TP | **present**, `src/llama-model.cpp:743` | present |
| `indexer_head_size` / indexer cache | present (`llama-hparams.h`, `llama-kv-cache-dsa.cpp`) | present, as its own series of commits |
| quantized KV cache in the QSA path | not found | `0ac4b1802` |
| fused-QKV segmentation for tensor split | not found | `5674c73aa` |
| qwen4exp large-graph node budget | not found | `c52ed2a0b` |
| two-node RPC and DFlash2 | present, ours | not present |

So they are **partial variants of each other**, not one line and its stale copy, and the union needs
per-feature decisions rather than `git rebase`. The mission-critical piece, the tensor-parallel Q
split, is already in ours.

### Baselines: what is on the rig

| target | weights on the rig | state |
|---|---|---|
| `deepseek-v4-flash-0731` | yes, `~/models/ds4-flash-0731-gguf/UD-Q4_K_XL` (145 GB, 5 parts) plus the DSpark draft GGUF (11 GB) | runs two-node with `llama-bench`; 145 GB does not fit one 121 GB Spark, so a two-node `llama-server` run is what remains |
| `qwen3.8-flash-next` | yes, `~/qwen3.8-flash-next/models/unsloth/Qwen3.8-Flash-Next-GGUF` plus an MTP Q8_0 draft | measured single-Spark with `llama-bench` (27.05 +- 1.33 tok/s). The 2-node serving run is the open job |
| `glm5.3-flash` | yes, `~/models/glm53-gguf/UD-IQ1_S` (3 parts, 86.69 GiB) | measured on one and two Sparks; needs the `glm5next` arch from upstream PR #27754 to build |

## Provenance

| Source | How it got here |
|---|---|
| `spark1:~/qwen3.8-flash-next/llama.cpp` branches | `git bundle --branches` in that clone, fetched into `refs/remotes/spark-qwen38/*`. The objects, including file contents, are now in this repo's object store, so the 374 MB bundle was not kept: re-create it with `git bundle create <file> --branches` in that clone if it is ever needed again |
| `/home/maci/Desktop/glm-5.3-flash/README.md` | the GLM feasibility note. Its repo has no remote; it stays local and is not copied here |
| `/home/maci/Desktop/qwen3.8-flash-next-spark/` | the Qwen3.8 source notes. The public repo is the upstream home, so refresh from there rather than editing in place |
