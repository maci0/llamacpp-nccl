# Upstream PRs

Status read from GitHub on 2026-09-13. This is a watch list. Nothing here is submitted, commented on
or updated by automation.

## Ours

| PR | state | what it means here |
|---|---|---|
| [#28003](https://github.com/ggml-org/llama.cpp/pull/28003) ggml-cuda: optimize single-token MMVQ dispatch for RDNA3 architecture | open, **draft**, review required, 1 file / +11 lines, untouched since 2026-08-30 | the same change ships here as `patches/0005`. Draft means no reviewer has it yet |

[#28002](https://github.com/ggml-org/llama.cpp/pull/28002) (cache and reuse quantized src1 activations
across projections in MMVQ) was closed and is not carried here.

## Upstream, carried or watched

| PR | author | state | why it matters here |
|---|---|---|---|
| [#27754](https://github.com/ggml-org/llama.cpp/pull/27754) model: add GLM-5-Next (GLM-5.3-Flash) | danielhanchen | open, **conflicting** with master, updated 2026-09-11 | adds the `glm5next` arch. GLM-5.3 does not build without it. We merge it locally; it is not redistributed here |
| [#27858](https://github.com/ggml-org/llama.cpp/pull/27858) fix assertion DFlash2 with `--split-mode tensor` on CUDA | art-den | open draft, mergeable, blocked | the tensor-split path we need for DeepSeek. The same split modes do not load with an RPC device at all, see below |
| [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) qwen4exp: add NextN/MTP draft head for Qwen3.8-Flash-Next | rmonsurate | open draft, **conflicting**, updated 2026-09-02 | the MTP draft head for the Qwen3.8 target |

## Recorded, not reported

`-sm row` and `-sm tensor` fail to load a model with an RPC device registered. Tested with
`llama-bench -m <glm IQ1_S shard 1> -ngl 99 --rpc 10.0.1.2:50052 -sm row|tensor -ts 1/1`, which returns
`llama_bench: error: failed to load model` after the RDMA link is up. The default `layer` split loads
and runs. Whether the cause is the RPC transport, the IQ1_S quant, or both is not established.

## TurboQuant (asked about 2026-09-14, researched not implemented)

Zandieh, Daliri, Hadian, Mirrokni, "TurboQuant: Online Vector Quantization with Near-optimal Distortion
Rate", arXiv 2504.19874, ICLR 2026 (Google Research, NYU, DeepMind). Random rotation (PolarQuant) plus
per-coordinate Lloyd-Max scalar quantization, with an optional 1-bit QJL residual for unbiased inner
products. It changes KV cache storage and the attention kernel by fusing dequantisation into scoring.
Weights are untouched. Not to be confused with llama.cpp's `TQ1_0`/`TQ2_0`, which are BitNet b1.58.

Upstream state: issue [ggml-org/llama.cpp#20977](https://github.com/ggml-org/llama.cpp/issues/20977) is
open, and PR [#21089](https://github.com/ggml-org/llama.cpp/pull/21089) (CPU `TBQ3_0`/`TBQ4_0`) was closed
unmerged. A CUDA port built on that PR is reported at 9.5 tok/s generation against a 42 baseline,
because the rotation sits on the dequantisation path on every step. Other implementations:
[0xSero/turboquant](https://github.com/0xSero/turboquant) (Triton plus vLLM),
tonbistudio/turboquant-pytorch, sharpner/turboquant-mlx, and Qdrant 1.18 for retrieval only.

Why it is not the lever here: 91 % of this pair's pass is dense and MoE `MUL_MAT`, the KV cache is a few
percent of traffic at the 512-token generations the benchmark uses, and an f16 to `q4_0` KV change on
the real workload measured slightly slower (`bench/deepseek-v4-tp-server-two-node.txt`). The vendor
figures are about attention-logit speed and memory, not decode throughput, and independent replications
are weaker than the claims.

What the same research points at instead, and what the local measurements independently found: vLLM's
`nvfp4_ds_mla` 4-bit MLA cache at 352 bytes per token, which changes attention compute rather than only
bytes; FlashMLA sparse top-k attention; FP4 dense weights on sm_121; and MoE and tensor-parallel
scheduling. The first and third are the byte-shaped levers this work already costed, the requantise
question recorded in `bench/deepseek-v4-op-profile.txt`.
