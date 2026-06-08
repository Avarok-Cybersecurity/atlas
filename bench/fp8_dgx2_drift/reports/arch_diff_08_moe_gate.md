# Arch-Diff #08 — MoE Gate + Top-K Routing (vLLM vs Atlas)

Scope: Qwen3-Next-80B-A3B-Instruct (Qwen3.6). Per HF `config.json` (DevQuasar
FP8-Dynamic snapshot) **`num_experts=512`, `num_experts_per_tok=10`,
`norm_topk_prob=true`, `hidden_size=2048`, `moe_intermediate_size=512`** —
flagging the task brief's "256 / top-8" as incorrect.

## Cross-engine function map

| Stage | vLLM | Atlas |
|---|---|---|
| Gate Linear ctor | `Qwen3NextSparseMoeBlock.gate = ReplicatedLinear(H, E, bias=False, quant_config=None)` (qwen3_next.py:251-257) — BF16 unquantized | `gate_nvfp4` vs BF16 weight selected at load (forward.rs:88-110) |
| Gate GEMM (decode) | `router_logits, _ = self.gate(hidden_states)` (qwen3_next.py:312) → cuBLAS BF16 | `ops::dense_gemv` (BF16) or `ops::w4a16_gemv` (NVFP4) — forward.rs:99-110 |
| Gate GEMM (prefill) | same `self.gate(hidden_states)` | `dense_gemm` / `w4a16_gemm` — forward_batched.rs:30-54 |
| Scoring + top-K | `FusedTopKRouter._compute_routing → fused_topk(..., scoring_func, renormalize)` (fused_topk_router.py:149-165) | `if correction_bias.is_some() → moe_topk_sigmoid else moe_topk_softmax` (forward.rs:113-148) |
| Softmax + top-K kernel | `vllm::moe::topkGatingSoftmax<NUM_EXPERTS=512>` — fused warp-cooperative softmax-first-then-iterative-argmax (topk_softmax_kernels.cu:230-494) | `moe_topk_softmax` — argmax-first-then-full-softmax (moe_topk.cu:30-176) |
| Indices/weights dtype | int32 / float32 (fused_topk_router.py:81-89) | u32 / float32 (moe_topk.cu:32-33) |
| Permute / sort | `moe_align_block_size` + cutlass/triton grouped MoE | `moe_count_experts` → `moe_sort_by_expert` → `moe_permute_tokens` → grouped GEMM → `moe_unpermute_reduce_indexed` (moe_permute.cu:19-202) |

## Findings

### 1. Gate matmul precision — IDENTICAL

vLLM forces gate BF16 unquantized (qwen3_next.py:255). Atlas calls
`dense_gemv(BF16)` when `gate_nvfp4` is `None` (forward.rs:99-110). For the
FP8-Dynamic checkpoint the gate weight is BF16 in both engines and produces
a BF16 `[N, 512]` `gate_logits`. **Not a drift candidate.** Flag: if
`gate_nvfp4` is populated on the NVFP4 Qwen3-Next variant, Atlas runs
`w4a16_gemv` while vLLM keeps BF16 — that would inject ~1.5% rel-err in
logits and could flip experts. **Out of scope for FP8-Dynamic.**

### 2. Algorithm divergence — softmax-then-topk vs topk-then-softmax

vLLM (topk_softmax_kernels.cu:373-400): full FP32 softmax over E **first**,
then k argmax passes over softmaxed row (:408-480), `renormalize` divides
`selected_sum` after the loop (:482-493).

Atlas (moe_topk.cu:60-106 + 108-176): iterative argmax over raw FP32 logits
**first**, then computes full softmax denominator once and divides each
top-K weight by it (:154-164), then optional renormalize (:166-174).

**Mathematically equivalent**: softmax is monotonic so argmax order on
logits = argmax order on softmax; with `norm_topk_prob=true` both paths
divide top-K weights by their sum, so the softmax magnitude cancels.

**Bit-for-bit not guaranteed**: differing reduction trees (vLLM warp
butterfly across `THREADS_PER_ROW=8` for E=512/VPT=8; Atlas reduces across
256 threads with a different warp+cross-warp partition). FP32 non-associativity
→ ulp-level differences in `exp_sum`. Negligible unless logits near-tied (#4).

### 3. Renormalize semantics — equivalent

vLLM (:486-491) divides recorded `max_val` by `selected_sum` (sum of
softmaxed top-K). Atlas (moe_topk.cu:166-174) divides each weight by their
sum. Intermediate `exp_sum` divisor cancels → identical normalized
weight vector. **Match.**

### 4. Tie-breaking on argmax — **divergence**

vLLM: lower expert index wins on ties (topk_softmax_kernels.cu:440-445,
`if (other_max > max_val || (other_max == max_val && other_expert < expert))`).

Atlas (moe_topk.cu:78-82, :91-99): no tie-break — `if (other_val > local_max)`
keeps whichever expert reached `tid==0`'s warp first; cross-warp picks the
lowest warp_id holding the max (:94-98), so winner depends on `local_idx`
produced by the strided per-thread loop. For exact-tie FP32 softmax the
result is not guaranteed to be lowest-index.

**FP8 drift relevance**: with BF16 gate and FP32 softmax, exact ties are
vanishingly rare on full-precision prompts. But FP8 quantization of upstream
activations (esp. low-magnitude tokens at deep layers) compresses logit
range and produces near-ties; combined with #2's ulp-level FP32 reordering
this can flip 1 of 10 experts — matches the
`project_qwen36_drift_moe_smoking_gun.md` "L24 7/8, L38 3/8" routing
divergence pattern. **Probable drift contributor on FP8 KV path.**

### 5. Bias semantics — N/A

Qwen3-Next has no `e_score_correction_bias`; both engines take the bias-free
softmax path. `moe_topk_sigmoid` (Atlas) and the sigmoid branch of
`fused_topk` (vLLM) are unused here. **Match.**

### 6. Sort / permute / weighted-reduce — match on accumulation

Atlas counting-sort (moe_permute.cu:76-202) builds `sorted_token_ids`,
`expert_offsets`, `token_to_perm`; reduction is FP32-accumulate then BF16
store (moe_permute.cu:106-115). vLLM does FP32-accumulate then BF16 store
through cutlass MoE epilogue. Both engines must dispatch a token to the
same expert set for outputs to align — see #4 for when they don't.

## Bug flags

- **[medium]** moe_topk.cu:67, :78, :95, :225, :233, :248 and
  moe_topk_sigmoid.cu:65, :74, :91 — no deterministic tie-break on argmax.
  Diverges from vLLM (lower-index-wins) when two experts have equal FP32
  softmax / sigmoid+bias. Plausible amplifier of late-layer FP8
  expert-routing drift.
  **Fix**: add `|| (other_val == local_max && other_idx < local_idx)` to
  the warp-shuffle and cross-warp reductions in both kernels.

- **[low]** moe_topk.cu writes `unsigned int` indices while downstream
  permute reads `int` (moe_permute.cu:22, :77, :85). Safe for E<2³¹ but
  worth unifying.

## Conclusion

Gate matmul, scoring choice, renormalize semantics, and FP32-accumulate
permute path **match** under FP8-Dynamic Qwen3-Next config. Two real
divergences: (a) **non-deterministic argmax tie-break** — candidate
amplifier of the FP8-KV expert-routing drift seen at L20/L24/L38; (b)
different reduction trees producing ulp-level softmax differences (benign
in isolation). The argmax-first vs softmax-first ordering is
mathematically equivalent and not a drift source.
