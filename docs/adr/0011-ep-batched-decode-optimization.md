# ADR-0011: Optimizing batched EP decode, and why it is bandwidth-bound

**Status:** Accepted
**Date:** 2026-05-29

## Context

Issue #99 lifts the `max_batch_size = 1` clamp under `--ep-size 2` by
multiplexing the head↔worker protocol (the `ATLAS_EP_PROTOCOL=v2` work).
Once the gate is lifted, concurrent requests actually batch instead of
serializing behind a one-slot queue. The motivating workload was a
4-concurrent agent burst whose tail latency spiked to ~605 s under the
batch=1 ceiling.

Lifting the gate is necessary but not the whole story. The batched
multi-sequence decode path it unlocks had two problems worth recording:
a correctness gap (the SSM layers' batched MoE was dead code) and a
performance trap (the generic grouped-GEMM is a net loss at small batch).
This ADR records the decisions made while making batched decode both
correct and fast on 2× GB10, and the larger finding that reframes where
future decode wins can come from.

The reference model throughout is Qwen3.5-122B-A10B-NVFP4 (48 layers: 36
gated-delta-net / SSM, 12 full-attention; 256-expert MoE), EP=2.

## Contents

- [Decision 1 — SSM multi-seq decode: per-seq mixer + batch-dispatched MoE](#decision-1)
- [Decision 2 — Attention multi-seq MoE: per-token at N≥4](#decision-2)
- [Decision 3 — Do not pursue CUDA graphs for EP decode](#decision-3)
- [Decision 4 — Leave the SSM projections BF16 (quantization frontier)](#decision-4)
- [Measurements](#measurements)
- [The binding constraint](#the-binding-constraint)
- [Consequences](#consequences)

## Decision 1 — SSM multi-seq decode: per-seq mixer + batch-dispatched MoE {#decision-1}

`Qwen3SsmLayer::decode_multi_seq_inner` delegated every sequence to the
single-token `decode()` in a loop, with a full batched path sitting
behind an early `return Ok(())` and `#[allow(unreachable_code)]` — dead
since the bug-#6 buffer-aliasing debugging. Per-sequence decode runs N
independent single-token MoE forwards (N × top_k expert GEMVs + N
per-token all-reduces under EP).

The decision: keep the per-sequence SSM **mixer** (conv1d + GDN
recurrence + projections — it carries independent recurrent state, so the
proven single-token kernels stay), but hoist the **MoE** out of the loop
and run it once, dispatched by batch size:

- N=2/3 → the fused `forward_k2`/`forward_k3` expert kernels (one batched
  all-reduce, no per-token launch overhead)
- N≥4 → the per-token MoE loop

Buffer safety (the old bug #6) is structural in the new layout: each
per-seq mixer writes its MoE input to `norm_output[i]` (a distinct
per-seq offset), `ssm_forward` never touches `norm_output`, and the
`ssm_out` it returns is consumed within the same loop iteration — so
nothing survives across sequences and the aliasing cannot recur.

The non-obvious part is the N≥4 fallback. The generic grouped-GEMM
(`forward_prefill`) is built for prefill, where M (tokens) ≫ the number
of active experts. At decode batch sizes the per-expert M is ~1, and the
expert sort / permute / pointer-table overhead — paid once per layer,
across 36 SSM layers — dominates. Measured: the grouped path pushed the
SSM decode step to ~140 ms at N=4 versus ~88 ms for the per-token loop.
So `forward_prefill` is declined for the SSM MoE until a true batched-EP
MoE kernel exists.

## Decision 2 — Attention multi-seq MoE: per-token at N≥4 {#decision-2}

The same grouped-GEMM trap applied to the attention layers' multi-seq
FFN, whose N≥4 branch used `forward_prefill`. Switched to the per-token
MoE loop for N≥4 (N=2/3 keep the fused `forward_k2`/`k3`). Measured at
N=4: the attention decode block dropped from ~40 ms to ~24 ms, ~8% off
the whole step, no regression. This mirrors Decision 1.

## Decision 3 — Do not pursue CUDA graphs for EP decode {#decision-3}

CUDA graphs are disabled under EP (`use_graphs = self.comm.is_none()`)
because the path assumed NCCL all-reduce was not capturable. That
assumption was tested and is false: the 2-rank all-reduce
(`ncclSend`/`ncclRecv` + a local add) runs entirely on one stream, and
the event-based async variant fork-joins a comm stream to the compute
stream — a multi-stream-capturable pattern. A prototype captured the
full n=1 decode step, including the inter-node RoCE collective, into a
graph: clean capture, coherent output, no deadlock.

It still does not help. n=1 throughput was ~40 tok/s with graphs versus
~42 without — a slight regression. The decode step is memory-bandwidth
and inter-node-NCCL bound; CUDA graphs only remove CPU launch overhead,
which is already hidden behind the GPU and network work. The "graphs give
~2×" result holds for launch-bound regimes (smaller models, single host),
not this one. The capability is kept on a side branch for a future
launch-bound model; it is not enabled by default.

Single-host (no EP) was also ruled out as a vehicle: the model is ~76 GB
of weights in memory, and on one GB10 (~109 GB usable) the KV cache gets
zero allocatable blocks at any usable batch/seq-len. EP=2 is the only
topology that leaves room for KV plus batch, which the issue already
observed.

## Decision 4 — Leave the SSM projections BF16 (quantization frontier) {#decision-4}

The largest BF16 weight still loaded every decode step is the SSM mixer
projections — `in_proj_qkv` + `in_proj_z` + `out_proj`, ~6.3 GB across
36 layers. Quantizing them to NVFP4 would cut the most per-step
bandwidth. They are kept BF16 deliberately: the loader already forces
`A_log`/`dt_bias` to FP32 to avoid "exponential error amplification in
the decay gate at 8k+ tokens." The recurrent path is precision-sensitive,
so 4-bit on the projections that feed it is a quality risk that would
pass a short coherence check and degrade at long context. Everything
safely quantizable is already NVFP4 at load (lm_head, the MTP head and
its experts, the routed MoE experts). The model sits at its safe
quantization frontier; this decision is to respect that boundary.

## Measurements {#measurements}

2× GB10, EP=2, Qwen3.5-122B-A10B-NVFP4, MTP speculative on (~77% accept).

- Decode step composition at N=2 (host-sync bucketed): SSM layers ~79%
  (within SSM: MoE ~57% / mixer ~43%), attention ~15%, lm_head ~6%.
- SSM decode step at N=2: 44 ms → 35 ms with the fused `forward_k2`
  (Decision 1), ~15–20% aggregate.
- Attention block at N=4: ~40 ms → ~24 ms (Decision 2).
- Batched vs serialized aggregate throughput at N=2 is ~equal (~32 tok/s
  either way); batching converts first-fast-second-waits into
  both-finish-together. The win is tail latency and admission, not
  aggregate tok/s.
- CUDA graphs under EP at n=1: ~40 tok/s on, ~42 off (Decision 3).

## The binding constraint {#the-binding-constraint}

At decode batch sizes that fit this hardware (N ≤ 8), the step is bound
by weight-load bandwidth and the inter-node all-reduce, not by kernel
launch overhead or arithmetic. Two concurrent tokens share almost no
weight loads — the SSM projections are sequential GEMVs and the MoE
experts at N=2 are mostly disjoint across the 256-expert pool — so
batching does not raise aggregate throughput at low N; it amortizes
admission and launch overhead and flattens the tail.

This explains the pattern across the whole arc: batching gave no
aggregate win at N=2, the grouped-GEMM lost (more work, same bandwidth),
and graphs did nothing (launch is not the bottleneck).

## Consequences {#consequences}

**Better.** Lifting the EP gate removes the tail-latency cliff from the
issue (4-concurrent burst no longer serializes). The MoE-dispatch fixes
make the batched path faster than the grouped-GEMM it replaced (~15–20%
at N=2/3 on SSM, ~8% at N=4 on attention) and delete a large dead-code
block. Output is coherent and cross-sequence-isolated at N=2 and N=4.

**Bounded.** Aggregate decode throughput at low concurrency is set by
memory bandwidth, not by these changes. Future decode wins on this model
must cut bytes-moved or collective latency — quantizing the recurrent
projections (a quality tradeoff, deferred), a true batched-EP MoE kernel
(one grouped expert GEMM tuned for small M plus a single batched
all-reduce), or a faster interconnect. Launch-overhead work (CUDA graphs,
kernel-launch fusion) does not move this workload.

**Kept.** The EP CUDA-graph capability and the decode phase-timing
instrumentation live on side branches, off by default, for a future
launch-bound model and for re-measurement.
