# llama.cpp work for the DGX Spark pair

Our llama.cpp changes for two NVIDIA GB10 machines (DGX Spark, sm_121), kept as ordered patches
against a pinned upstream commit so they can be reviewed and applied without carrying a fork's
history.

The goal for this work: on this rig, for three models, be faster than the other engines.

| target | llama.cpp surface | bar to beat | state |
|---|---|---|---|
| `qwen3.8-flash-next` | `src/models/qwen4exp.cpp`, `conversion/qwen4exp.py` | vLLM TP2: 31.14 tok/s single-stream, **74.28 tok/s aggregate at 8 concurrent**; SGLang ~21 tok/s | single-Spark GGUF works; **measured 27.05 tok/s generation** (see `bench/`) |
| `deepseek-v4-flash-0731` | two-node NCCL/RPC: `tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `common/speculative.*`, `src/models/deepseek4.cpp`, `src/models/dflash.cpp` | vLLM on this rig, with DSpark spec decode: 63.1 / 112.1 / 140.9 / 156.0 tok/s at c1/c3/c5/c6 | **on 4-bit weights, two-node tensor parallel + DSpark: 40.4 tok/s at c1 with `-np 2`** (two runs, 40.7 and 40.3, spreads 3-5 %), against 34.3 / 62.4 / 77.6 / 82.2 at c1/c3/c5/c6 with the eight slots the multi-level table needs, sum 256.5. The slot count has to match the concurrency: at two slots c2 gives 56.8 and c3 falls to 47.2 as the third request queues. Our own vLLM run on this pair is 42.0 at c1, so this arm is ~4 % from it and 55 % from the reference image. The TP arm under `llama-cli` is 27.5 tok/s at c1, without the draft 20.6 against 12.8 (`bench/deepseek-v4-tp-server-two-node.txt`, `bench/deepseek-v4-tp-coordination.txt`, `bench/deepseek-v4-op-profile.txt`) |
| `glm5.3-flash` | none of ours: upstream PR [ggml-org/llama.cpp#27754](https://github.com/ggml-org/llama.cpp/pull/27754) adds the `glm5next` arch | SGLang day-0 NVFP4 TP2: 24.7-30.3 tok/s on two Sparks | **runs: 19.74 +- 0.12 tok/s on one Spark** (IQ1_S) |

So the honest position: single-Spark llama.cpp is close to the other engines on single-stream
generation (27.05 against vLLM's 31.14 for qwen4exp, 19.74 against SGLang's 24.7-30.3 for GLM, 38.8
against our own vLLM's 42.0 for DeepSeek, though the anemll reference is 63.1 there) and far behind
them once there is concurrency, and the two-node tensor-parallel work is what would close that.

## Base

Upstream llama.cpp commit **`304665fe7`** ("Add IQ type handling for MoE (#28476)"). That is the base
the patches apply to. They were also merged onto current master (`790cf51aa`) and built there.

## Contents

```
patches/    our commits, in order, each a normal git patch with authorship
scripts/    start_rpc_spark.sh, run_qwen38_tp2.sh, build_spark.sh, bench-spark.py
docs/       targets.md - the three-model mission, bars and blockers
            upstream.md - the PRs we depend on or watch, with their state
bench/      raw llama-bench output, with the machine and flags
```

| patch | what it is |
|---|---|
| `0001` | register the qwen4exp NextN tensors in gguf-py |
| `0002` | the accumulated work: two-node NCCL/RPC transport, DFlash2 draft plumbing, Qwen4Exp model and converter, and the three scripts |
| `0003` | the workspace scope note and the three-model target docs |
| `0004` | the Qwen4Exp line-reconciliation record |
| `0005` | ggml-cuda: single-token MMVQ dispatch optimization (our PR work, RDNA3 path) |

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

glm5next 313B.A17B IQ1_S - 1.5625 bpw | 86.69 GiB | 320.76 B | CUDA | ngl 99
  pp256   240.61 +- 0.00 t/s
  tg64     19.66 +-  0.00 t/s      (1 rep, 2026-09-13, one GB10)
  pp128   166.59 +-  4.93 t/s
  tg32     19.74 +-  0.12 t/s      (3 reps, 2026-09-13, one GB10)
```

`tg64` and `tg32` agree, so generation length does not move this number. An earlier single-rep run in
this workspace recorded `tg32 16.43`; that does not reproduce and was taken with another `llama-bench`
holding the same GPU.

Raw output in `bench/`. The counts that matter are produced by `scripts/bench-spark.py`, which runs
the reference protocol (chat, BST prompt, temp 0.7, seed 1234, thinking off, 512-token generations,
the two greedy gates) with `--reps` passes per level and reports each level as a median with its range,
because single passes on this rig swing by tens of percent. Reproduce a DeepSeek two-node number with:

```sh
ssh spark2 'cd ~/llamacpp-nccl && scripts/start_rpc_spark.sh'          # once
scripts/bench-spark.py --base http://127.0.0.1:8081/v1 --reps 3 --levels 1 3 5 6
```

Notes for reproducing: this build's `llama-bench` **rejects `-c`** and prints its usage instead; the `MTP-*` GGUF set is a second copy of the weights, so serving with the draft head
needs both nodes; and the GLM run needed the `glm5next` arch, which comes from upstream PR #27754
rather than from the patches here.

For GLM-5.3 that number is the whole story so far: **19.74 tok/s on one Spark** against the day-0
SGLang NVFP4 figure of 24.7-30.3 tok/s on two, so llama.cpp on one node is already in the same range
as the other engine on two.

## Two nodes

The two-node path works. `llama-bench --rpc <peer>:50052` brings the model up with backend `CUDA,RPC`
and the link negotiates RDMA over RoCE:

```
RDMA probed:    dev=rocep1s0f1 gid=5 RoCEv2 qpn=10958 inline=316
RDMA activated: qpn=10958->38484 mtu=4096 rx_depth=24
```

At identical flags (`-p 128 -n 32 -r 3`), both arms repeated three times:

| | pp128 | tg32 |
|---|---|---|
| one node, `CUDA` | 166.59 +- 4.93 | 19.74 +- 0.12 |
| two nodes, `CUDA,RPC` | 168.06 +- 3.73 | 17.53 +- 0.03 |

Prompt processing is a wash; generation is 11 % slower over the pair. That is the honest state: a
working mechanism, not yet a faster one. The model already fits on one node (86.69 GiB of 124.6), and
the default layer split is already even, so there is no misconfiguration to fix here. What the default
does, and what `-ts 1/1` does to it, is in `bench/glm53-split-detail.txt`:

```
load_tensors:        CUDA0 model buffer size = 42286.33 MiB
load_tensors: RPC0[10.0.1.2:50052] model buffer size = 43480.40 MiB
```

`-ts 1/1` produces those same two figures. `-sm row` and `-sm tensor` both fail to load the model with
an RPC device present. So making the two-node path pay needs a different mechanism, not a different
split ratio.

### The case the two-node path exists for

DeepSeek-V4-Flash does **not** fit one Spark: UD-Q4_K_XL, 144.44 GiB, 284.33B params (llama-bench
prints its file type as `MXFP4`). It runs across the pair:

```
deepseek4 ?B MXFP4 MoE | 144.44 GiB | 284.33 B | CUDA,RPC | ngl 99
  pp64   164.02 +-  9.34 t/s
  tg32    15.08 +-  1.13 t/s      (3 reps, 2026-09-13, RDMA negotiated qpn=11029->38555)
```

Against vLLM on the same rig, which serves this model with DSpark speculative decoding at 63.1 tok/s
at c1 and 156.0 at c6, that is a large gap, and at that point it was the honest floor: no speculative
decoding here, and a layer split over RPC rather than real tensor parallelism.

### The DSpark draft, wired to the two-node run

The draft head is on the box (`~/models/ds4-flash-0731-gguf/dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf`,
11.3 GiB) and both `-md` and `--spec-type draft-dspark` work over the pair. Measured on the reference
chat workload (chat, BST coding prompt, temp 0.7, seed 1234, thinking off, 512-token generation, one
stream), n_max 7, three passes:

```
pass 1   23.1 tok/s      pass 2   24.8 tok/s      pass 3   25.6 tok/s
median   24.8 tok/s      spread   23.1 - 25.6
```

The same build with no draft gives 14.8 tok/s on that workload, so the draft is worth 1.7x. The
draft-depth sweep, the acceptance counts and the full flags are in `bench/deepseek-v4-dspark-two-node.txt`;
acceptance is not the problem, at 68.6 % with 4.41 tokens per pass against the reference's 4.70.

What is left is per-step cost: 43.1 ms per token here against 15.9 ms for the reference, 2.5x, with
both engines verifying a comparable number of tokens per step. A layer split runs node A's half of the
layers and then node B's half in series, so a token pays one node's full pass over roughly 6 GB of
active weights, and upstream llama.cpp reaches 169 GB/s on this box for that same MXFP4 MoE shape. The
tensor-parallel path that would put both nodes on the same token is implemented in the patches but is
allreduce-latency bound over this RoCE link: 12.8 tok/s without a draft and 21.9 with it, both
`llama-cli` at 2048 ctx, so no better than the layer split.

### Tensor parallelism, and the host serialisation that was hiding in it

The two-node layer split has a floor around 33 tok/s with 4-bit weights, because each node reads its
half of the layers in series and a token costs one node's full pass over 6 GB of active weights. Only
running both nodes on the same token moves that, which is what the tensor-parallel arm does. It has
been losing to the layer split here (12.8 tok/s without a draft, 21.9 with it) because of the cost of
doing so: an RPC session shows ~92 cross-node collectives per token, and the client blocks in each.

The RPC server was doing two things between receiving a subgraph and joining its collective: calling
the synchronous `ggml_backend_graph_compute()` and hence not reading the next request until the
previous subgraph had finished on the GPU. Submitting asynchronously and synchronising only in the
eight handlers that touch backend memory (11 edits in `ggml/src/ggml-rpc/ggml-rpc.cpp`, the allreduce
needs none because it is stream-ordered behind the graph) gives:

| configuration, 4-bit `UD-Q4_K_XL` | before | after |
|---|---|---|
| tensor parallel, no draft | 12.8 tok/s | **20.6 tok/s** |
| tensor parallel + DSpark | 21.9 tok/s | **27.5-35.4 tok/s** over six runs; the uid carry below is a certain code fix but its throughput effect is not separable from the rig's spread at three reps an arm |

Correct output in both cases, and for the first time the tensor-parallel arm beats the layer split at
c1. The same server change is neutral on the layer split (median 24.2 over seven passes against the
published 24.8 and 25.3), which is what the change predicts: there the client needs the server's data
after every subgraph anyway. The "before" column is the recorded number for the same fork and flags,
not a same-day A/B; a 20-minute load each way has not been paid yet, so read the percentages as
directional. Details: `bench/deepseek-v4-tensor-parallel-async.txt`.

A second, smaller defect sat behind the same path: the server reads the graph uid off the wire and
keys its graph cache with it, but never assigned it to the graph, so `cgraph->uid` stayed 0 there and
the CUDA backend's cache fast path (`cgraph->uid != 0 && cgraph->uid == graph->uid`) was unreachable.
Every launch instead walked the graph, memcpying a full `ggml_tensor` per node plus its sources and
memcmping the result. One line (`graph->uid = uid;`) removes that, and four runs put the arm at
27.5 / 29.6 before and 30.9 / 35.4 after.

The 4-bit tensor-parallel path has a measured ceiling rather than a guessed one. `llama-bench` on it
gives pp512 757.60 against 378.95 on one Spark, an exact 2x, so the split costs nothing in bandwidth at
large batches. Decode is 23.23 against 20.51, only +13 %, because one token per pass leaves the matvec
latency-bound at about 79 GB/s instead of the 165 GB/s it reaches at five tokens. At that batch-5 rate
a speculative pass costs about 94 ms of compute per node, which is 50 tok/s; the 74 ms the reference
needs assumes 209 GB/s, the best this rig has shown. So this path needs both the coordination overhead
(a quarter of a pass) removed and the batch-5 bandwidth raised past 200 GB/s.

What is left on that path is measured too. With the collectives made no-ops (wrong output, but the
floor any collective implementation could reach) the same build gives 24.6 against 20.6 tok/s, so the
~87 collectives per token cost about 9 ms of the 48.5 ms a token takes. The rest is per-subgraph host
and launch work: ~88 subgraphs per token, each costing about 0.28 ms once the node's own 15 ms of
compute is accounted for. A thread-per-collective on the client looked like an obvious waste and
measured 19.6 against 20.6, so it was reverted; the thread was overlapping, not wasting.

### Where the time actually goes

Profiled with `nsys` on both nodes (`--cuda-graph-trace=node`, without which the server's capture has
no CUDA data at all), then with thread-CPU timers inside the decode loop; numbers and method in
`bench/deepseek-v4-profile-two-node.txt`.

Per pass (168 ms of wall for 4.41 generated tokens, four graph submissions):

- client GPU ~62 ms and server GPU ~69 ms, strictly serialised by the layer split;
- client host work ~17 ms, of which the target decode is 4.8 ms and the two draft decodes 11.2 ms;
- the rest is handoff and stall.

So the pair is 78 % GPU-busy in total but split between two nodes that cannot run at the same time, and
per token the two halves move ~6 GB of active weights at ~200 GB/s per node, about three quarters of
this part's peak. An earlier reading of this box's CPU counters said the host was saturated; the timers
show that CPU is `cudaStreamSynchronize` spinning while it waits for the stream, which costs power but
not throughput. The binding constraint is the serialised weight traffic of the layer split.

### The fastest configuration measured so far

A layer split does not change the per-token weight traffic (each node reads half the layers, in
series, which is what one node would read anyway), so the lever that does change it is how many bytes
the weights occupy. `UD-Q2_K_XL` is 90.2 GiB and fits one Spark, so it needs no RPC and no split:

```
   -m .../UD-Q2_K_XL/...-00001-of-00003.gguf  -ngl 99  -c 16384 -np 8
   -md .../dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf -ngld 99
   --spec-type draft-dspark --spec-draft-n-max 5
   GGML_CUDA_GRAPH_OPT=1

   c1   38.5 tok/s    c3   64.3    c5   79.1    c6   82.1    sum 264.0
```

`GGML_CUDA_GRAPH_OPT=1` is worth about 2 % on the sum and nothing at c1; the rest of the table below is
the same serve without it.

With the default four slots the same serve gives 40.0 / 46.5 / 60.3 / 55.1, sum 201.9: the earlier
levels were slot-limited, not engine-limited, because a level of 5 or 6 has to queue behind four
slots. Both arms are in the record.

At c1 that is 1.6x the two-node layer split's 24.8, on one machine instead of two, and 1.12x the
two-node tensor-parallel serve measured later (34.3, `bench/deepseek-v4-tp-server-two-node.txt`),
which reads the 4-bit weights and matches it from c3 up. The same serve without
the draft gives 19.8, so the draft is worth 2.0x here. The caveat travels with the number: 2.72 bits
per weight against the reference's roughly 4.5, so it reads about 40 % fewer bytes per token and the
output is lower quality by an unmeasured amount. The gates are trivial and do not speak to that. Full
record: `bench/deepseek-v4-q2-single-node.txt`.

Against the reference it is still behind at every level: 38.8 against 63.1 at c1, 64-80 against
112-156 at c3-c6.

The quant ladder ends here too. `UD-IQ1_S` (76.9 GiB, 15 % fewer bytes per token) measures
38.1 / 61.1 / 77.4 / 81.7, sum 258.3, against 38.8 / 64.2 / 76.5 / 79.9, sum 259.4: the same within
the spread, because the IQ kernel path is slower and the draft acceptance drops from 74.2 % to 55.4 %.
Its gate answers also degrade visibly (` Paris.", "target": "` and `72 (2) 8`), so it is not a usable
configuration. Going smaller buys nothing.

Why no single-Spark configuration can reach the reference: 63.1 tok/s at c1 is 74.3 ms for a pass
which generates 4.69 tokens and verifies five. At the 200-210 GB/s this build reaches, five tokens of
active weights must fit in 2.4-3.0 GB per token, and the two usable quants read 3.3 to 3.87 GB. Both
nodes together would fit (1.65-1.94 GB per node), but the per-layer coordination would have to cost
about 6 ms per token, against the 35-63 ms per token the measured implementations spend. That is
engine work, not a configuration or a quant.

Profiling the single-node arm says the same from the other side (`bench/deepseek-v4-kernel-profile.txt`):
the GPU is 86 % busy there, against 37-41 % on the layer split, and the expert matvecs already run at
about 215 GB/s, which is near this part's 273 GB/s. What is left is thousands of small per-layer
projections that a five-token batch leaves memory-latency bound. The reference is not reading fewer
bytes per token than this arm; it has two machines on the same tokens, with the collectives overlapped.

**Measurement note.** Single-rep runs on this rig are not trustworthy: the earlier
`tg32 16.59` here and `tg32 16.43` for GLM were both single reps and both read high. The
GLM pair and the DeepSeek pair are now 3-rep medians with their spread, and anything
compared across machines is compared at identical `-p`/`-n`. The qwen4exp figure is 2 reps
and the GLM `tg64` figure is 1 rep, so those two carry less weight than the rest.


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
