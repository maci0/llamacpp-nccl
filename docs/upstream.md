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
