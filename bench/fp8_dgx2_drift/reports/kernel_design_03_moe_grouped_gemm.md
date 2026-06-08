# Kernel Design #03 — MoE FP8 Grouped GEMM (Atlas vs vLLM)

Investigating why MoE output cosine = 0.08 vs vLLM on Qwen3.6-A3B FP8.

---

## 1. `moe_fp8_grouped_gemm.cu` anatomy

Math: `C[M_e,N] = A[M_e,K] (BF16) @ dequant(B_expert[N,K] FP8 E4M3 block-scaled)`.

### Tiles / launch
- `M_TILE=64, N_TILE=64, K_STEP=16, PAD=2` (`:26-29`).
- Grid `(ceil(N/64), max_m_tiles, num_experts)`, block 128 = 4 warps (`:22`).
- Each CTA = one (expert, m-tile, n-tile). Per-expert pointers:
  `B_exp = (u8*)B_weight_ptrs[expert_id]`,
  `S_exp = (bf16*)B_scale_ptrs[expert_id]` (`:186-188`). NULL ⇒ remote EP.

### Routing into the GEMM
`expert_offsets[E+1]` gives row range `[m_start, m_end)` of expanded tokens
firing this expert (`:174-176`). `sorted_token_ids[m_start+m_idx]` indirects
to the original A row (`:238-241`). The down-proj call at
`forward_prefill_fp8.rs:291` passes `DevicePtr(0)` → direct indexing
(`:240`), correct because gate/up already permuted A.

### MMA primitive — BF16×BF16→FP32
`:138-150` is `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
**BF16 inputs, FP32 accumulator.** No FP8 native MMA (silicon limit on
sm_121, see `project_fp4_mma_gb10.md`). No TMA.

### Smem layout
- `smem_A[64][18]` (BF16), `smem_B[16][66]` (BF16).
- B is dequanted **FP8 → FP32 (via `E4M3_LUT_GMOE`) → BF16** before smem
  write (`:262`, `__float2bfloat16(E4M3_LUT_GMOE[byte])`).
- Scale is **NOT** folded here — only LUT value (`:248-250`).

### K_PROMOTE=64 — two-level FP32 promotion
At `:201-210, 273-286`:
- Inner: step K by 16, accumulate FP32 `inner_acc[8][4]`, **no scale**.
- Every K_PROMOTE=64 cols (4 MMAs): fold block scale into `outer_acc`:
  `scale = bf16→f32(S_exp[n_block*k_blocks+k_block]); outer_acc += inner_acc*scale; inner_acc = 0`.
- `n_block = cta_n/FP8_BLOCK` (`:224`). N_TILE=64 < FP8_BLOCK=128, so all
  64 N-cols share one scale per K-block — correct.
- K_PROMOTE=64 divides FP8_BLOCK=128: two folds per 128-K block but both
  use the same scale, so equivalent to one fold at K=128.

### Output
FP32 → BF16 at store (`:300-306`). No epilogue scaling, no output quant.

---

## 2. `moe_shared_expert_fused_fp8.cu` — different scale strategy

GEMV (decode/single-token), structurally different:

- **No MMA.** Pure FP32 dot-product over 16-K-at-a-time uint4 loads
  (`:174-235`).
- Scale folded **per element, eagerly**: `:184-185` loads `sc1, sc2` once
  per 128-K chunk, then `:214-222` does
  `wf1_0 = s_lut[byte] * sc1` before the FMAs.
- Accumulation FP32, final `__float2bfloat16(acc)` at store
  (`:243, :249`).
- SiLU+down variant precomputes `silu(g)*u` into `s_act[1024]` FP32
  (`:328-332`), then same eager-scale GEMV.
- `batch2/batch3` (`moe_shared_expert_fused_fp8_batch2.cu:217-225`)
  identical numerics, more tokens.

### Eager folding is numerically fine here
`Σ a_i*(w_i*s) = s*Σ a_i*w_i` — both in FP32 the same result modulo
rounding. The two-level pattern only matters if the accumulator is BF16
or the scale cast is lossy; neither here. The SSOT "violation" (synthesis
A9) is brittle but **not currently a correctness bug**.

---

## 3. Reconciling agent claims

| Claim | Truth |
|-------|-------|
| Agent #9: grouped GEMM uses K_PROMOTE=64 two-level FP32 promotion | **TRUE** — `:36, :201-210, :273-286` |
| Agent #2: "Atlas dequants FP8→FP32→BF16 in smem before MMA" | **TRUE** for the routed GEMM (`:262`) — but the BF16 round is **lossless** (FP8 E4M3 has 3-bit mantissa, BF16 has 7, see `:204` comment). Agent #2's implication of precision loss is wrong. |

---

## 4. What drives cosine 0.08 (orthogonal)?

Cosine ≈ 0 cannot come from precision drift — even FP4 keeps cosine ≥ 0.95
across an MoE block. Orthogonality means **the set of non-zero output rows
is largely disjoint** between Atlas and vLLM. That's routing / permutation,
not the GEMM kernel.

### Top 3 suspects

1. **Expert routing / permutation mismatch (≈ 65%).**
   `project_qwen36_drift_moe_smoking_gun.md` already documents routing
   diverging 8/8 → 7/8 → 3/8 by L38. The doc `arch_diff_08:60-74`
   claiming Atlas had no tie-break is **stale** — live code
   (`moe_topk.cu:73-84, 96-100`) now has lower-index tie-break. Remaining
   risk: `norm_topk_prob` ordering and Atlas's **argmax-first vs vLLM's
   softmax-first** topk (`arch_diff_08:32-49`). If Atlas's
   `sorted_token_ids` disagree with vLLM's even on 30% of tokens, the
   per-expert output matrices have disjoint row supports → block cosine
   ≈ topk_overlap ratio. 0.08 ≈ 8% expert overlap at late layers.

2. **Missing per-token FP8 activation quant (≈ 25%).**
   `arch_diff_09:53-65`: vLLM does `per_token_group_quant_fp8(A)` before
   the GEMM (`deep_gemm_moe.py:290`). Atlas passes raw BF16 A
   (`forward_prefill_fp8.rs:238`). Mathematically swaps `A_BF16` for
   `dequant(quant_fp8(A))`. Per-token per-128 scales across 8 routed
   experts compound differently across 4096 tokens — won't hit 0.08
   alone but amplifies #1.

3. **Shared vs routed engine SSOT drift (≈ 8%).**
   FP32-equivalent on paper. Worth excluding with a unit test feeding
   the same (a, W, s) to both kernels.

**Action**: instrument `(token_id, expert_id)` end-to-end. If overlap
with vLLM ≈ 0.08, suspect #1 confirmed and the grouped GEMM is innocent.

---

## 5. Replacing with vLLM-equivalent W8A8 + FP32 epilogue — effort

Closest vLLM equivalent on non-Hopper: Triton `fused_moe_kernel` block-FP8
(`fused_moe.py:316-516`).

- **New kernel** `moe_fp8_grouped_gemm_w8a8.cu` (~450 LoC): same geometry,
  add per-token FP8 quant of A into scratch (~80 LoC, mirrors
  `dense_gemv_fp8w.cu`). FP8 native MMA still unavailable on sm_121, so
  smem dequant + BF16 MMA stays; the win is per-token A scale folded into
  the epilogue: `outer_acc += (a_scale*b_scale) * inner_acc`.
- **Shared expert mirror** (~150 LoC) to keep SSOT.
- **Rust callsite**: `gemm_quant.rs::moe_fp8_grouped_gemm` gains
  `a_scale_ptr`; `forward_prefill_fp8.rs` (~50 LoC) for pre-step quant +
  scratch alloc.
- **Total**: ~700 LoC across 4 files, **3–5 days impl** + 1 day to gate
  behind `ATLAS_FP8_MOE_W8A8=1` and A/B bench.

---

## Ranked bug list

| # | Suspect | Confidence | Evidence |
|---|---------|------------|----------|
| 1 | Expert routing / permutation mismatch | **65%** | cosine ≈ 0.08 ≈ expert-overlap floor; `project_qwen36_drift_moe_smoking_gun.md`; `arch_diff_08:32-49` argmax-first vs softmax-first |
| 2 | Missing per-token FP8 activation quant (A_BF16 vs A_FP8) | **25%** | `arch_diff_09:53-65`; `forward_prefill_fp8.rs:238` passes BF16 A; vLLM `deep_gemm_moe.py:290` quantises |
| 3 | Shared-vs-routed engine numeric drift (eager vs late fold) | **8%** | `arch_diff_09:14, 95-98`; FP32-equivalent but brittle |
| 4 | K_PROMOTE granularity / BF16 cast in dequant | **2%** | `:262` cast lossless per `:204`; K_PROMOTE divides FP8_BLOCK cleanly |

**Recommendation**: instrument `(token_id, expert_id)` end-to-end before
touching the GEMM. If routing overlap with vLLM ≥ 0.95, return here for
#2. If ≤ 0.5, the GEMM is innocent — fix lives in `moe_topk*.cu` and
`forward.rs:113-148` gate path.
