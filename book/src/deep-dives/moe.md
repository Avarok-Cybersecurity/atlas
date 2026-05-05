# MoE Routing & Experts

Mixture-of-Experts is where the Atlas supported-model matrix gets most of its diversity. Qwen3.5 routes 128 experts top-10. MiniMax-M2.7 routes 256 experts top-8 with **sigmoid** gating (not softmax). Gemma-4 has shared experts. Nemotron-H has an MoE-only layer variant (no mixer, just routing + FFN). The engineering problem is making all these shapes share one kernel pipeline without losing per-shape efficiency.

## The generic MoE block

```
residual ─► gate (linear) ─► softmax/sigmoid ─► topk ──► dispatch ──► experts ──► gather ──► weighted sum ──► residual
                                                    │         │
                                                    │         └─ (N different FFN weights, one per expert)
                                                    │
                                                    └─ (tokens routed to ≤ k experts each)
```

Five kernels contribute to one MoE block:

1. **Gate** — linear projection. Uses the same GEMM kernels as attention projections.
2. **Top-k + softmax/sigmoid** — per-token expert selection. From `atlas-reduce::topk`.
3. **Dispatch** — token scatter to expert-local arrangement. Conceptually a permutation.
4. **Expert FFN** — `k` copies of `silu_mul_quant(up(x)) → down(...)`. Grouped GEMM with one batch per expert.
5. **Gather + weighted sum** — reduce expert outputs back to the residual shape. From `atlas-reduce::moe_sum`.

## Prefill: grouped GEMM

For prefill (many tokens per step), the expert FFN is a grouped GEMM: `M` tokens split across `N` expert buckets, each bucket does its own `(B_e, K) × (K, N_out)` matmul.

Atlas's `moe_prefill.cu` kernel takes the "sort tokens by expert, then one GEMM per expert group" approach. The sort is parallel — dispatch tables are computed in a single pre-kernel pass. The GEMM itself uses the same 16×8×16 BF16 MMAs as attention, one CTA per expert-bucket.

This is the kernel that hits the headline **MoE W4A16 256-expert 80-token: 3.87× PyTorch** number in the benchmark table.

## Decode: token-level MoE

For decode (one token per sequence per step), grouped GEMM is the wrong shape — each "group" has one or two tokens, and the launch overhead dominates. Atlas's decode MoE kernels (`moe_expert_relu2_down_shared.cu` and `moe_shared_expert_fused_fp8.cu`) take the opposite approach: one warp per (token, expert-in-topk) pair, compute the expert's up-proj and down-proj in a single kernel.

The shared-expert path for models like Gemma-4-26B fuses the shared expert's compute with the per-token dispatch, avoiding a round-trip through memory.

## The 256-expert case: MiniMax-M2.7

MiniMax-M2.7 is the extreme end of the MoE support matrix: 256 experts, top-8, **sigmoid** routing (not softmax). The sigmoid variant lets multiple experts contribute independently without the softmax normalisation; the tradeoff is that `topk` has to pick from a longer tail of meaningful scores.

Atlas's `minimax_moe` layer uses:

- `norm_topk_prob = true` semantics (topk weights normalised by their sum, not softmax).
- A dedicated `topk_sigmoid` kernel path that reads sigmoid scores rather than post-softmax probabilities.
- An EP=2 token-dispatch pipeline — with 256 experts, a single node is impractical; MiniMax ships only as EP=2.

The wave-17 bug fix that landed the "M2.7-NVFP4 EP=2 full PASS" milestone addressed four subtle issues:

1. **`rms_norm` placement** — the rms_norm output was being reused after the MoE dispatch modified it, causing subtle coherence drift.
2. **`norm_topk_prob`** — initial implementation used softmax-normalised weights; corrected to sum-normalised sigmoid.
3. **FP8-free path** — MiniMax weights are NVFP4 end-to-end; early code had an accidental FP8 upcast in the shared-expert path.
4. **Template-forced thinking detection** — MiniMax's chat template seeds `<think>` differently from Qwen; the detector needed to distinguish.

## Shared experts

Several models (Gemma-4, Qwen3-VL) have a "shared expert" FFN that runs for every token in addition to the `k` routed experts:

```
moe_out = topk_weighted_sum(expert_out) + shared_expert(x)
```

The shared expert is a standard dense FFN run alongside the routed experts. Atlas fuses the shared FFN with the gather step in the decode kernel (`moe_shared_expert_fused_fp8.cu`) to save a memory round-trip.

## Expert parallelism (EP=2)

For 122B, 119B, 229B models, the experts don't fit on one GB10. Split them across two ranks:

- Gate runs on every rank (replicated).
- Token top-k assigns each token to `k` experts; tokens destined for remote experts are sent via `reduce_scatter`.
- Expert FFN runs locally on the owning rank.
- Results come back via `all_gather`.

The collective ops go through `spark-comm::CommBackend`; the dispatch kernel `moe_dispatch.cu` handles the local-vs-remote bucketing. See [spark-comm](../crates/spark-comm.md) and [Multi-GPU & EP=2](../operations/multi-gpu.md).

## Routing edge cases

A handful of bug-sweep wave findings landed in the MoE code:

- **Sibling stride bugs** — the attention and SSM layer code assumed `K=2` MTP strides in the expert outputs; corrected to support `K=1`/`2`/`3` uniformly.
- **Slot-keyed `verify*_graph` caches** — CUDA graph instances for MTP-verify were keyed by batch size alone; needed `(batch, k)` to avoid replaying a K=1 graph on a K=2 step.
- **MoE topk bounds** — a weight-loader edge case where `num_experts_per_token` exceeded `num_experts` for a few experimental checkpoints; added an assertion at load.
- **MoE topk weights == 0 guard** — a numerical edge case where all-zero gate logits produced NaN after normalisation; now clamped.

## Why MoE is fast on GB10

Three design choices stack:

1. **Grouped-GEMM prefill dispatches fewer kernels than per-token** — the fixed overhead per expert group is tiny compared to the math.
2. **Fragment-time dequant** — NVFP4 expert weights are never materialised to BF16; they unpack at the MMA boundary, same as attention (see [NVFP4 deep dive](./nvfp4.md)).
3. **Fused SiLU×Mul + quant** — the up-proj output is never written to memory in BF16; it's SwiGLU'd and re-quantized in one kernel before the down-proj reads it back. Saves the entire BF16 intermediate (a major BW win on MoE, where the intermediate is `batch × topk × moe_intermediate_size`).

The end result on a 256-expert MoE at batch=80: Atlas at 8.43 ms vs PyTorch at 32.65 ms — **3.87×**. That's the biggest single-kernel win in the benchmark table.

## Files to read

- `kernels/gb10/<model>/<quant>/moe_prefill.cu` — grouped-GEMM prefill.
- `kernels/gb10/<model>/<quant>/moe_expert_relu2_down_shared.cu` — token-level decode MoE.
- `kernels/gb10/<model>/<quant>/moe_shared_expert_fused_fp8.cu` — fused shared-expert path.
- `kernels/gb10/minimax-m2-229b/nvfp4/moe_dispatch.cu` — EP=2 token dispatch.
- `crates/spark-model/src/layers/moe.rs`, `moe_prefill.rs`, `moe_shared.rs`, `minimax_moe.rs` — Rust side.
- `docs/design/ep2-token-dispatch-design.md` — EP=2 design note.
- `docs/design/tool-calling-gap-analysis.md` — not directly MoE but covers the MoE+tool-calling interaction.
