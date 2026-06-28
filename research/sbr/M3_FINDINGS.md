# M3 — 2-D (position × layer) sheaf L₀ reconciliation: findings

Synthetic CPU prototype (`M3_sheaf_prototype.py`, fp32 sim / fp64 solve). Tests
the one regime where sheaf cohomology is non-vacuous for SBR: a 2-D cell complex
`(token-position × layer)`, where the four restriction maps around a plaquette
need not commute (H¹≠0 / nonzero curvature). The load-bearing object is the **L₀
sheaf-Laplacian least-squares reconciliation**: cross-layer × cross-position
redundancy over-determines the multi-layer SSM state, so it *might* denoise cheap
lossy per-layer (contractive-window) reconstructions.

## Setup
- L=48 layers, H=4 heads, KD=VD=16 → stalk dim **D=1024** per cell.
- Affine-GDN per-token update reused from Phase-1: `h_t = (exp(g_t)I − β_t k_t k_tᵀ)h_{t−1} + β_t k_t v_tᵀ`.
- Heavy-tailed decay: 76% fast heads ∈[0.95,0.999], **24% slow** ∈[0.9999,1.0]
  (measured 50/192 = 26% slow; 36/48 layers carry ≥1 slow head).
- Layers coupled: layer ℓ+1's streams are driven by a projection of layer ℓ's
  per-token state (low noise) → consecutive-layer states are genuinely related,
  so the **vertical restriction maps are meaningful** (not noise).
- **Horizontal edges** = EXACT per-chunk GDN affine operators `Γ` with the exact
  (materialized-input) drive `c`. **Vertical edges** = ridge-fit linear map
  `V_ℓ: stateℓ → stateℓ+1` (+ bias).
- **Lossy input** = per-layer contractive-window replay from a zero state over the
  last τ tokens. Small τ → exact-ish for fast heads, degraded for slow heads.
- Solve: `min_x [ w_d·‖x−lossy‖² + w_h·‖x_v−Γx_u−c‖²(horiz) + w_v·‖x_v−Vx_u−b‖²(vert) ]`
  with a subset of **layers pinned exact**; CG over the free cells.
- Metric: cosine vs the exact reference, over **free (unpinned) cells**, reported
  full-cell and split into **slow-head** vs **fast-head** sub-blocks.

Diagnostics: vertical ridge fit cosine **0.978** (min 0.971); **mean plaquette
curvature (relative H/V non-commutation) = 0.479** — the fitted 2-D sheaf is
strongly non-flat.

## Main table — τ=48, pin 12/48 layers exact (every 4th), free cells
| method | min | mean | median | ≥0.999 | slow-sub | fast-sub |
|---|---|---|---|---|---|---|
| lossy input | 0.547 | 0.964 | 0.973 | 0.000 | 0.916 | 0.994 |
| **horiz-only** (no cross-layer) | 0.613 | **0.995** | 0.9998 | **0.861** | 0.989 | 0.999 |
| FULL 2-D sheaf (w_v=3) | 0.954 | 0.992 | 0.997 | 0.022 | 0.994 | 0.991 |

Reading: horizontal-only (exact GDN recurrence + lossy data + exact pins) already
does the heavy lifting — lossy 0.964 → 0.995, and 86% of free cells clear the
0.999 gate. The FULL sheaf with a fitted map at w_v=3 **raises the floor** (min
0.61→0.95, slow-sub 0.989→0.994) but **drags the bulk down** (fast-sub 0.999→0.991,
≥0.999 collapses 0.861→0.022). The vertical term behaves as a regularizer toward
the vertical map's own ~0.99 accuracy ceiling: it helps the worst slow cells and
hurts the many cells horizontal already fixed.

## (a) τ sweep — mean cell cosine / slow-subspace cosine (pin 12/48)
| τ | lossy | horiz-only | FULL (w_v=3) |
|---|---|---|---|
| 24 | 0.894 / 0.791 | 0.987 / 0.974 | **0.992 / 0.993** |
| 48 | 0.964 / 0.916 | **0.995** / 0.989 | 0.992 / 0.994 |
| 96 | 0.989 / 0.978 | **0.998** / 0.996 | 0.993 / 0.995 |

A clean **crossover**: FULL beats horiz-only only at τ=24 (lossy so bad that
horizontal+pins cannot recover it and the vertical prediction is more accurate).
At τ≥48 horiz-only wins on mean; the vertical map's ~0.99 ceiling caps FULL.

## (b) pin-set size sweep (τ=48) — mean cell cosine over free
| pinned | lossy | horiz | full | full−horiz |
|---|---|---|---|---|
| 24/48 | 0.958 | 0.994 | 0.991 | −0.0028 |
| 16/48 | 0.963 | 0.995 | 0.992 | −0.0025 |
| 12/48 | 0.964 | 0.995 | 0.992 | −0.0027 |
| 6/48 | 0.964 | 0.995 | 0.993 | −0.0025 |
| 3/48 | 0.965 | 0.995 | 0.992 | −0.0029 |

The cross-layer term is **uniformly slightly negative** vs horizontal-only at this
τ regardless of anchor density. The win in horiz-only comes from the exact
horizontal recurrence + anchors, **not** from cross-layer structure.

## (c) vertical weight w_v sweep (τ=48, pin 12/48); w_v=0 == horiz-only
| w_v | cell-mean | slow-sub | ≥0.999 |
|---|---|---|---|
| 0.0 | 0.9951 | 0.9887 | 0.861 |
| **0.3** | **0.9966** | 0.9933 | 0.861 |
| 1.0 | 0.9964 | 0.9957 | 0.725 |
| 3.0 | 0.9924 | 0.9944 | 0.022 |
| 10.0 | 0.9786 | 0.9856 | 0.003 |

There **is** a tuned sweet spot (w_v≈0.3) where the fitted-map sheaf beats
horiz-only on both mean (+0.0015) and slow-subspace (+0.0046) **without** losing
gate fraction. But it does not *raise* the gate fraction (0.861 either way): it
improves slow-cell fidelity, not the count of cells clearing 0.999.

## (d) vertical-map fit quality + ORACLE ceiling (τ=48, pin 12/48, w_v=3)
| fit | fit-cos | full-mean | full-slow | vs horiz |
|---|---|---|---|---|
| ridge λ=1e−3 (overfit) | 1.000 | 0.990 | 0.993 | −0.0048 |
| ridge λ=1e−1 | 0.978 | 0.992 | 0.994 | −0.0027 |
| ridge λ=1e+1 | 0.820 | 0.993 | 0.993 | −0.0021 |
| ridge λ=1e+3 | 0.775 | 0.990 | 0.984 | −0.0054 |
| **ORACLE-V w_v=1** | 1.000 | 0.9988 | 0.9974 | **+0.0037** |
| **ORACLE-V w_v=3** | 1.000 | 0.9996 | 0.9992 | **+0.0045** |
| **ORACLE-V w_v=10** | 1.000 | **0.9999** | **0.9998** | **+0.0048** |

ORACLE-V = best linear ℓ→ℓ+1 map fit directly on the **exact** grid states
(curvature driven to **0.0001**, a nearly flat sheaf). This is the decisive
ablation: with an accurate vertical map the FULL 2-D sheaf **beats horiz-only by
+0.0048**, lifts the slow-subspace to **0.9998**, and reaches **mean 0.9999** —
clearing the coherence gate that horizontal-only (0.995, 86%) cannot.

## Cost
At τ=48, pin 12/48: solve is 331,776 free DOF (CG ~30 matvecs). Lossy windows cost
~15.5k token-updates vs ~147k for exact full replay of the unpinned layers (~9×
cheaper) — but only horiz-only converts that cheap lossy input into gate-passing
states; the cross-layer sheaf does not improve on it with a fitted map.

## VERDICT (brutally honest)

**The 2-D sheaf does genuine, non-trivial work in principle — but the realistic
version is dominated by the cheaper non-sheaf baseline, so it is NOT a usable SBR
lever as-is.**

1. **The concept is real, not vacuous.** With an accurate vertical map (ORACLE,
   curvature→0), the cross-layer L₀ solve reaches mean **0.9999** / slow-sub
   **0.9998** and beats the no-cross-layer baseline by **+0.0048**, clearing the
   gate that horizontal-only plateaus below (0.995 / 86%). The extra fidelity is
   exactly the cross-layer information — this is **not** something a per-layer
   (horizontal-only) least-squares can match. So the "H¹≠0 / 2-D redundancy"
   premise is validated at the ceiling.

2. **But with a fittable vertical map it adds nothing exploitable.** A ridge map
   on activations tops out at cosine ~0.978 / plaquette curvature ~0.48. That
   accuracy caps the reconciled state at ~0.99, so at practical τ (≥48) the FULL
   sheaf is **net-negative vs horizontal-only** (−0.002 to −0.003 mean) and
   destroys the gate fraction (0.861→0.022 at w_v=3). It only helps (a) at extreme
   lossiness τ=24, or (b) at a hand-tuned tiny w_v≈0.3 where the gain is +0.0015
   mean / +0.005 slow and **does not raise the gate fraction**.

3. **The win that exists is horizontal + anchors, which is exactly M1.** The
   honesty check from the plan is answered: horizontal-only (exact GDN recurrence
   + exact pinned layers) — a plain per-layer least-squares with **no sheaf
   cross-layer coupling** — already lifts the lossy input from 0.964 to 0.995 and
   86% gate-passing. That is the M1 lever (exact recurrence from near anchors), not
   sheaf topology.

4. **The bottleneck is precisely measured: vertical-map quality (plaquette
   curvature), not the sheaf math.** Closing the gap requires an inter-layer
   restriction map far more accurate than ridge-on-activations — but a static
   linear ℓ→ℓ+1 map is fundamentally limited by each layer's own recurrence memory
   (the source of curvature 0.48), and the only map that reaches curvature ≈0 is
   fit on the exact states you are trying to recover (circular — if you had them
   you would not need reconciliation).

**Practical conclusion for Atlas/SBR:** ship the exact horizontal lever (M1:
eviction/tail-pin + checkpoint replay over known inputs). The 2-D sheaf L₀
reconciliation is a real but currently *un-actionable* idea: it would only pay off
given a high-accuracy learned inter-layer state map (curvature ≪ 0.1), which we do
not have and cannot get cheaply for GDN. Recommend **not** pursuing real-model
(dgx2) validation unless/until such a map is demonstrated — the synthetic ceiling
already bounds the upside and the fitted-map regime is a negative.
