# SBR Phase-1 — architectural findings (synthetic, fp32) → refines the method

## What the measurements decided

### 1. Dense operator-transport: DEAD for GDN (FLOP-negative)
Composing dense per-chunk affine operators `(A 128×128, B 128×128)` costs
`O(n_chunks · d³)`; sequential replay costs `O(n_chunks · 64 · d²)`. Since `d=128 > 64`,
**compose ≈ 2× the cost of replay**, and the operator is not more compact than the
state (`A` is diagonal + rank-≤64 correction ⇒ effectively full at chunk=64). The
O(log n) segment-tree story does not pay off for GDN. (Exact for Mamba2 scalar gates,
but Qwen3-Next is GDN/DPLR.)

### 2. Contractive-window reconstruction: WORKS — the real glocal lever
GDN is contractive (decay<1), so `h_M` depends only on the last ~τ tokens before M;
older tokens' contributions decay away. Local replay from a **zero** state over the
last τ tokens (mixed decay: 75% heads ∈[0.95,0.999], 25% weak-forget ∈[0.9995,1.0]):

| τ (tokens) | min cosine | mean | median |
|---|---|---|---|
| 256  | 0.8818 | 0.9724 | 1.0000 |
| 512  | 0.9759 | 0.9944 | 1.0000 |
| 1024 | 0.9989 | 0.9997 | 1.0000 |
| **2048** | **0.999997** | 1.000000 | 1.000000 |

**τ ≈ 2048 reconstructs ALL heads to cosine ≥ 0.999 from zero state, independent of
(M−C).** This bounds warm-hit replay regardless of depth/eviction — flat TTFT.

### 3. Low-rank "slow-mode global summary": DEAD
Weak-forget head far-field state is near-full-rank: **86/128 modes for 90% energy,
115/128 for 99%**. You cannot summarize the long-range part in few modes. The glocal
split is therefore: **local-τ window (approx, gated) + exact checkpoint** (NOT low-rank
compression of the global part).

## Refined SBR method (honest, for GDN / Qwen3-Next)

- **EXACT default = eviction + checkpoint-density fix** (the dominant lever, matches the
  adversarial critique's #1). Marconi already checkpoints every 64 tokens but LRU
  **evicts the active conversation's recent checkpoints**, stranding it far from an
  anchor (the user's sibling-session symptom). Fix: **per-conversation tail-pin** +
  decouple KV-leaf vs SSM-any-node LRU so M−C stays ~one turn. Exact, cheap.
- **GLOCAL "transport" = contractive-window cap.** When no usable near checkpoint
  exists (cold / heavy eviction), reconstruct `h_M` by replaying only the last
  τ≈2048 tokens — bounded cost, cosine-gated. This is the local section staying
  connected to the whole via the decaying restriction maps. **Approximate ⇒ behind a
  coherence gate, not the exact default.**
- **fp32 operator application** for the replay window (not bf16 FLA) — fixes the
  documented warm-hit corruption (`trait_prefill_recur.rs:80-90`, "token-stutter
  corruption 2026-06-10") that currently forces full O(M−C) WY4 replay.

## REAL-MODEL decay envelope (dgx2, A_log + dt_bias, 36 GDN layers × 32 heads = 1152)

Extracted `decay0 = exp(-exp(A_log)·softplus(dt_bias))` (baseline dt; real dt is
input-modulated, so this is an envelope):

| metric | value |
|---|---|
| per-head decay0 | min 0.0, median **0.958**, max 1.0 |
| weak-forget heads (decay>0.9995) | **5.9%** (68/1152) |
| horizon τ (tokens to ε=1e-3) | median **163**, p90 9164, p99 53798 |
| τ≤512 / τ≤2048 / τ≤8192 | **64% / 76% / 88%** |

**Heavy-tailed.** Most heads forget within a few hundred tokens (window-friendly), but
**~12% have τ>8192** — long-range heads, the ones that matter most for instruction
following.

### Decisive consequence: contractive window cannot cover slow heads
A slow head does not forget old tokens but **still integrates new ones**, so its state at
M depends on the *entire* [C,M] span — a last-τ window drops the contributions of
[C, M−τ] that it has NOT forgotten. Therefore:

- **Fast heads (~76%, τ≤2048):** local-τ window replay — cheap, flat in (M−C). ✓
- **Slow heads (~24%):** require exact replay over the full [C,M]; the ONLY way to make
  that cheap is a **near checkpoint** ⇒ the **eviction/tail-pin fix is mandatory and
  primary** (matches the adversarial critique's #1). Low-rank compression of these
  heads is dead (near-full-rank).

### Reordered priorities (data-driven)
1. **EVICTION / tail-pin (exact, primary)** — bounds (M−C) so the un-truncatable slow
   heads replay cheaply. This is the real cure for 1s→21s.
2. **Contractive-window (glocal, secondary)** — cuts the fast-head (76%) cost and gives a
   bounded approximate fallback when no near checkpoint exists; coherence-gated.
3. Optionally replay only the slow-head subset over long spans when cold (0.24× cost).

The sheaf/glocality framing now maps cleanly and *honestly*: the "global section" is the
slow-head exact state (kept via tail-pinned checkpoints); the "local sections" are the
fast-head windows; they glue because the fast heads' dependence on the far past has
decayed to zero (the restriction maps are contractive).

## Authoritative gate still required (dgx2, real Qwen3-Next-NVFP4)
- Real **per-head horizon under actual inference dt** (not just baseline) — needs an
  instrumented prefill on dgx2; sets the true fast/slow partition and τ.
1. **Real τ**: synthetic τ≈2048 assumes a decay mix; the real per-head decay sets the
   actual safe window. Measure per-head decay during a real prefill.
2. **Coherence safety**: confirm window reconstruction at the real τ holds
   cosine/argmax/KL under the coherence-parity gate (the model already sits near an
   FP8 content floor — 0.999 may need headroom).
3. **Baseline curve**: reproduce the warm-hit TTFT-vs-(M−C) 1s→21s curve to define the
   number we must beat.
