# SBR Phase-0 — Validation kill-gate results

## (A) Exactness of operator-transport — PASS (synthetic, decisive)

Per-token GDN update is affine in the state: `h_t = A_t h_{t-1} + B_t`, with
`A_t = exp(g_t) I − β_t k_t k_tᵀ` (diagonal-plus-rank-1) and `B_t = β_t k_t v_tᵀ`,
derived directly from the kernel recurrence in `gated_delta_rule.cu`. Per-chunk
affine operators therefore compose associatively.

`research/sbr/sbr_phase0_synthetic.py` (CPU, fp32, realistic decay∈[0.95,0.9999],
β=sigmoid(N), L2-normalized keys) compares balanced **segment-tree reassociated**
chunk-operator compose against **sequential per-token replay**:

| depth (tokens) | chunks | transport cosine | max-abs-err | rel-L2 |
|---|---|---|---|---|
| 512   | 8   | 1.00000024 | 1.97e-6 | 8.9e-7 |
| 2048  | 32  | 1.00000000 | 1.43e-6 | 8.5e-7 |
| 8192  | 128 | 0.99999994 | 1.61e-6 | 8.7e-7 |
| 16384 | 256 | 1.00000000 | 1.61e-6 | 8.8e-7 |

Worst transport cosine across depths = **0.99999994** ≫ 0.999 gate. Reassociation
error sits at the fp32 rounding floor; depth does not degrade it. **Transport is
numerically sound — proceed to implementation.**

## (B) Invertibility / group-subtraction eligibility — PASS under nominal dist.

`A_t` eigenvalues: `exp(g)` (mult K−1) and `exp(g) − β‖k‖²` (along k). Under nominal
distribution: **100% eligible** (|λ_min|>1e-3), median condition number 2.04, p99 16,
max 436. Group-subtraction (Phase 3) is broadly applicable; FiBA backbone covers the
ill-conditioned tail.

> **Caveat / TODO before Phase 3:** confirm against REAL Qwen3-Next-NVFP4 weights on
> dgx2 that no head cluster sits at β≈1 / `exp(g)≈β‖k‖²` (near-singular). This gates
> the eviction design only — transport (Phase 2) is invertibility-independent.

## Sequencing
Phase 0(A) green → Phase 1 (persist chunk operators + path-product) → Phase 2
(transport) → real-weight 0(B) audit on dgx2 → Phase 3 (group-subtraction eviction).
