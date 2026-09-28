# Targets

This repo is the llama.cpp home for the DGX Spark performance work. The mission is to beat vLLM,
SGLang and the other engines on the same rig for three models:

| # | Model | llama.cpp surface here | The bar to beat | State |
|---|---|---|---|---|
| 1 | `deepseek-v4-flash-0731` | two-node NCCL/RPC: `tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `common/speculative.*`, `src/models/deepseek4.cpp`, `src/models/dflash.cpp` | vLLM on the same 2 Sparks, guarded, c1/c3/c5/c6 = 63.1 / 112.1 / 140.9 / 156.0 tok/s (the anemll reference), our own vLLM 42.0 / 79.3 / 109.1 / 122.8 | on 4-bit `UD-Q4_K_XL`, a two-node tensor-parallel `llama-server` with DSpark reaches **40.5 tok/s at c1 with `-np 2` CUDA graphs on** (fifteen rotated rounds inside one load read 24.11 ms a token with the captured graphs against 25.93 launching the nodes directly, both metrics and `mean len = 4.52` in all thirty reps; the earlier reading that graphs cost 7.7 % came from an A/B whose `GGML_CUDA_DISABLE_GRAPHS` arms differed only in the variable's value, which that check reads as presence, so both arms ran eager), and 34.3 / 62.4 / 77.6 / 82.2 at c1/c3/c5/c6 with the eight slots the multi-level table needs. Slot count must match concurrency: at two slots c2 gives 56.8 and c3 falls to 47.2 as the third request queues. That is ~4 % behind our own vLLM run at c1 and 55 % behind the reference image, up from 17 % and 75 % before this. Everything else measured here is recorded as a null or closed with numbers: coordination, the scheduler drain, device syncs, the MMVQ parameter table. `bench/deepseek-v4-tp-server-two-node.txt` |
| 2 | `glm5.3-flash` | none of ours: upstream PR [ggml-org/llama.cpp#27754](https://github.com/ggml-org/llama.cpp/pull/27754) adds the `glm5next` arch | SGLang day-0 NVFP4 TP2: 24.7-30.3 tok/s on two Sparks | measured: 19.74 +- 0.12 tok/s one Spark, 17.53 +- 0.03 two-node; the split costs 11 % and buys nothing |
| 3 | `qwen3.8-flash-next` | `src/models/qwen4exp.cpp`, `conversion/qwen4exp.py`; GGUF MTP serving | vLLM TP2 on the same 2 Sparks: 31.14 tok/s single-stream, **74.28 tok/s aggregate at 8 concurrent**; SGLang ~21 tok/s. Our own llama.cpp bar: single-Spark GGUF MTP 4-bit, ~27 to 32 tok/s | GGUF single-Spark serving DONE, measured 27.05 +- 1.33 tok/s; 2-node is the open job |

Read the goal as: for each model, make llama.cpp on this rig faster than the best other engine, measured
the same way. The vLLM numbers above come from the vLLM project's guarded protocol (`drive-median.sh
<tag> 3 512 1 3 5 6`, chat, seed 1234, three passes, median), which lives in the sibling project, not
here.

## Per target

### 1. deepseek-v4-flash-0731

Two-node tensor parallelism over NCCL/RPC is the work. What exists:

Measured state of the gap, in one paragraph, so it does not have to be reassembled from the bullet
list. A c1 pass is 114 to 124 ms for 4.52 accepted tokens, and the paper under those numbers now comes
from a profiled run rather than from instruments built for the purpose. One c1 rep under nsys, 113
passes, shows 106 ms of GPU kernels in a pass and a kernel resident three quarters of the time, split
as 40.5 ms in the MoE expert matvecs, 24.3 ms in the dense matvecs, 13.4 ms in the NCCL allreduce
kernels and about 28 ms in everything else, which is the first decomposition of this pass that does not
depend on subtracting one instrument's figure from another's. The MoE is the largest term and it issues
160 MB a pass where 89 MB is distinct: a call reads one expert's rows per (token, k-slot) pair, and only
about 20 distinct experts are behind the 36 pairs. That redundancy is **already absorbed by cache** and
not worth chasing: reordering the grid so the six slot blocks of one row range are issued together,
which is a cheap stand-in for the grouped kernel that would remove the redundancy outright, is a
regression of 3 to 16 % in an alternating test, because the original order streams each expert's rows
and the swap trades that locality for reuse the L2 was already providing. The MoE therefore runs at
**152 GB/s, 55 % of this part's peak**, and why a 6-token, 20-expert matvec reaches 55 % where a 35 MB
dense matvec reaches 85 % is the open question, with block shape and issue order both excluded.
Removing the collectives altogether shortens a pass from 124 to
101 ms, so that term is 17 to 19 % and exposed rather than overlapped
(`bench/deepseek-v4-tp-coordination.txt`). The remaining candidates have been measured and
are null or small: the scheduler's cross-device drain, the 47 device drains per pass, CUDA graph
churn, graph re-serialization and the peer's request path, and so are the three things tried last: the MMVQ
parameter table (2-4 %, mixed signs), a 4-bit KV cache (38.9 against 40.4 at c1, because KV is a few
percent of traffic at these context lengths), and kernel fusion (the fusion this backend already does is
worth 10 %, measured by switching it off: 39.5 and 38.2 on against 35.0 off; what is not fused yet is
the elementwise and copy families, about 2400 of the graph's 5208 nodes, and DeepSeek V4's
hyper-connection ops are already fused). TurboQuant was researched rather than implemented and is KV-side; the note with its upstream
state and the reason it cannot explain this gap is in `docs/upstream.md`. The two matmul families are
61 % of the pass's kernel time, `MUL_MAT_ID` at 38 % and `MUL_MAT` at 23 %, with the NCCL allreduce
kernels another 13 %. The MoE's 1.8 times redundant issued traffic is absorbed by cache, proven by the
grid reordering that would exploit it being a regression, so the MoE runs at 152 GB/s of distinct
traffic where one large dense matvec reaches 231; the dense matvecs run at 150 to 165 GB/s. Closing the
rest wants either that 55 % question answered or a step-shaped execution change; no knob in this
configuration moves it, and nothing under 5 % can be resolved here, because one arm repeats a 13 %
spread within a single run. For scale, at the ~40 tok/s c1 this arm holds, the two bars in the table
above sit at different distances: 42.0, our own vLLM run on this pair, is 4 % away, and 63.1, the
anemll reference image, is 58 % away. Both are 4-bit serving of the same weights; the reference image is
the stricter one. The bar is 74 ms a pass and a pass is 114 to 124 ms of which about 59 ms is
byte-bound work, 21 to 23 ms is the exposed allreduce and 17 ms is the draft, so reaching it needs the
engine's small-batch rate raised by about half rather than another knob turned.

- the transport and draft plumbing, committed as `580205750` on `unmerged-prs` and published as
  `patches/0001`-`0005` against upstream `304665fe7`;
- `scripts/start_rpc_spark.sh` for bringing up the peer;
- a working two-node run over RDMA, now in tensor-parallel mode and served: `--rpc <peer>
  --split-mode tensor -md dspark-*.gguf --spec-type draft-dspark --spec-draft-n-max 5` on the
  two-node `llama-server` gives 34.3 / 62.4 / 77.6 / 82.2 tok/s at c1/c3/c5/c6, sum 256.5, on the
  reference chat workload. The same workload on the layer split with the same draft gives 24.8 at c1,
  and the same server without the draft gives 14.8. The draft is not the gap: both engines verify
  about 4.5 tokens per step, the server's own counters put the draft at 20.6 ms of the 131 ms pass,
  and the reference does the whole step in 74.9 ms. What makes this model expensive per token is the
  MoE: at 256 experts and top-6 an extra token in the batch pulls in up to 6 more expert matrices, so
  the marginal cost of a token inside a six-token verify is 13 ms against a 3.4 ms prefill slope.
  Numbers, flags and both accounting sheets: `bench/deepseek-v4-tp-server-two-node.txt`.
- a negative result from the same build: the draft head cannot be pinned to one device
  (`-devd CUDA0`) while tensor parallel sets `output_replicated`, which aborts in
  `ggml_backend_sched_backend_id_from_cur` because `output.weight` sits in a Meta() buffer.
- a profile of the layer-split run on both nodes (`nsys`, one capture each, then thread-CPU timers
  inside the decode loop): the two nodes are serialised by the layer split and together GPU-busy 78 %
  of the wall, but never at the same time, and per token they move ~6 GB of active weights at
  ~200 GB/s per node, about three quarters of the part's peak. The client's high CPU reading turned
  out to be `cudaStreamSynchronize` spinning while it waits for its own stream, so the host is not
  the constraint. `bench/deepseek-v4-profile-two-node.txt`.
- an offline per-op profile, because `perf` and `ptrace` are unavailable here, and **GPU performance
  counters are too**: `nsys profile --gpu-metrics-devices=0 --gpu-metrics-set=gb10x` returns
  `Illegal --gpu-metrics-devices argument: 0. Insufficient privilege, see ERR_NVGPUCTRPERM`. So the
  kernel captures give per-kernel times, and DRAM or L2 utilisation would need an operator with the
  counter permission. `test-export-graph-ops` writes the op shapes of the reserve graphs without
  reading any weights, and `test-backend-ops perf --test-file` times them on CUDA0, so a profile costs
  seconds instead of a 17-minute load; `GGML_TEST_N_RUNS=1` makes an op's weights cold, which matters
  for anything under L2 and changes nothing for the large shapes. First results on the six-token verify graph: large dense projections run at 205 to
  220 GB/s against 232 GB/s measured achievable, the lm head costs 2.4 ms per pass on its own, and
  every op sits on a per-op latency floor near 10 us. It rules the expert matvec kernel out as the
  lever: the two-node arm's incremental slope is 8.75 ms per token against 7.4 ms per node predicted
  for the MoE traffic, so that path is already near 85 % of what the part can move.
  `bench/deepseek-v4-op-profile.txt`.
- what the coordination actually costs, from the RPC server's own per-connection counters: 1.9 to 2.0
  full graph serializations and 6.5 to 11 allreduces per generated token, one collective per subgraph
  boundary. The scheduler's cross-device drain (`ggml_backend_sched_compute_splits`) was the prime
  suspect for the two nodes not overlapping, since it waits on a stream that also holds the collective;
  gating it off with `GGML_SCHED_NO_DRAIN=1` gives 34.7 at c1 against 34.3 and 58.9 at c3 against 62.4,
  gates identical, so it is not the bottleneck and it was reverted. `GGML_CUDA_GRAPH_OPT=1` on both
  nodes is also null (c1 34.7, c3 61.8, inside the band), and the client log from the baseline session
  shows why without an A/B: 154408 uid fast-path reuses against 8875 graph invalidations, about 9 % of
  the client's subgraph launches, worth roughly 0.03 ms per token. What is left of the 86 ms fixed cost
  of a 125 ms pass is not coordination after all: both nodes' GPUs draw 44 to 45 W through a pass
  against ~14 W idle, so both work concurrently and neither is idle-waiting, and `dmon`'s sm% is
  unusable here (it reads 0 % on the peer while the peer executes half the subgraphs). The target is
  the small-batch kernel efficiency instead: the same dense ops run at 79 GB/s at one token per pass
  and 165 GB/s at five, against 205 to 220 GB/s for the large projections.
  `bench/deepseek-v4-tp-coordination.txt`.
- the fastest configuration measured so far, and the reason for it: a layer split does not change the
  per-token weight traffic, only how many bytes the weights occupy does, so a quant that fits one
  Spark and needs no split is the lever. `UD-Q2_K_XL` (90.2 GiB) single-node with the DSpark draft,
  `-np 8` and `GGML_CUDA_GRAPH_OPT=1` gives 38.5 / 64.3 / 79.1 / 82.1 at c1/c3/c5/c6, sum 264.0,
  against 19.8 without the draft.
  With the default four slots it gives 40.0 / 46.5 / 60.3 / 55.1, sum 201.9: above c1 the levels were
  slot-limited, not engine-limited. It is 1.6x the two-node arm at c1 and it is still behind the
  reference everywhere, and at 2.72 bpw it is not a like-for-like quant.
  `bench/deepseek-v4-q2-single-node.txt`.
- the same record closes the smaller-quant option: `UD-IQ1_S` (76.9 GiB, 15 % fewer bytes per token)  measures 38.1 / 61.1 / 77.4 / 81.7, sum 258.3, no better than `UD-Q2_K_XL`, and its gate answers
  degrade visibly. Why no single-Spark configuration can reach the reference is in that file too:
  the bar allows 2.4-3.0 GB of active weights per token at the bandwidth this build reaches, and the
  usable quants read 3.3 to 3.87 GB. The two-node path fits that budget, but its coordination would
  have to fall from the measured 35-63 ms per token to about 6 ms, which is engine work.
- the uid carry in the same request path: the server keyed its graph cache with the uid it read off
  the wire but never put it on the graph, so the CUDA backend's cache fast path was unreachable and
  every launch re-walked the graph node by node. One line fixes it; two runs either side measure
  27.5 / 29.6 before and 30.9 / 35.4 after. `.scratch/patch-server-graph-uid.py`.
- the 4-bit path forward, and the one engine change that has landed: with `UD-Q4_K_XL` weights the
  layer split has a floor near 33 tok/s because each node reads its half of the layers in series, so
  only tensor parallelism can move it. That arm had been losing (12.8 tok/s without a draft, 21.9 with
  it) because the RPC server called the synchronous `ggml_backend_graph_compute()` per subgraph, so it
  read the next request only after the GPU had finished the previous one, while the client blocked in
  ~92 collectives per token. Submitting asynchronously and synchronising only in the eight handlers that
  touch backend memory (11 edits in `ggml/src/ggml-rpc/ggml-rpc.cpp`; the allreduce needs none, being
  stream-ordered behind the graph) gives 20.6 tok/s without the draft against 12.8, and 27.5 with
  DSpark against 21.9, correct output in both. That was the first tensor-parallel result to beat the
  layer split's 24.8; served under the guarded workload the same path now reaches 34.3 at c1 and 82.2
  at c6, see the first bullet. The change lives in the Spark build trees, uncommitted, and is reproducible with
  `.scratch/patch-async-server.py` in the workspace. `bench/deepseek-v4-tensor-parallel-async.txt`.
- a kernel profile of the single-node arm (`bench/deepseek-v4-kernel-profile.txt`): the GPU is 86 %
  busy there against 37-41 % on the layer split, the expert matvecs run at ~215 GB/s against the
  part's 273 GB/s peak, and what is left is thousands of small per-layer projections that a five-token
  batch leaves memory-latency bound. The reference is not reading fewer bytes per token; it has both
  machines on the same tokens with the collectives overlapped. That, and only that, is the remaining
  factor.

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
