# M3 — 2-D (position × layer) sheaf reconciliation (research, background)

## Why this and not the 1-D sheaf
On a causal 1-D path (a conversation) a cellular sheaf has H¹=0, so "gluing"
collapses to the exact associative prefix-product — vacuous (M1 already exploits
it without topology). The **only** place sheaf cohomology does real work here is a
**2-D cell complex** `(token-position × layer)`: on a 2-D complex H¹ can be ≠0, and
the four restriction maps around each plaquette need not commute → genuine
curvature. The honest framing (per the topology scan): the load-bearing object is
the **L₀ sheaf-Laplacian least-squares reconciliation** (the "H¹≠0" label is
decorative); its value is that cross-layer × cross-position **redundancy
over-determines** the state, denoising cheap *lossy* per-layer reconstructions.

## Hypothesis
A cheap APPROXIMATE per-layer SSM-state reconstruction (e.g. contractive-window
replay with small τ — exact for the ~76% fast heads, lossy for the ~24% slow/
weak-forget heads) can be **reconciled** by an L₀ harmonic solve over the
position×layer sheaf into a globally-consistent multi-layer state with
**materially higher fidelity than the lossy input** — ideally enough to clear the
coherence gate (cosine ≥ 0.999) at a fraction of exact-replay cost.

If true → an approximate fast-reconstruction mode (window + sheaf glue) and a
genuine novel contribution. If false → an honest negative result; M1 (exact)
stands as the win.

## Construction
- Base complex: rectangle, cells `(p, ℓ)`, p ∈ chunk boundaries, ℓ ∈ 0..L (48).
- Vertex stalk `F(p,ℓ)` = layer ℓ's SSM state at position p (full, or low-rank embed).
- **Horizontal** edge `(p,ℓ)→(p+1,ℓ)`: restriction = layer ℓ's per-chunk GDN affine
  operator `(Γ,C)` (position recurrence — already materialized by the FLA kernel).
- **Vertical** edge `(p,ℓ)→(p,ℓ+1)`: restriction = inter-layer map (how layer ℓ's
  output conditions layer ℓ+1's SSM-state evolution). Approximate as a linear map
  fit from activations (ridge regression) or use the actual input projections.
- Plaquette curvature = non-commutation of the 4 maps (nonzero under approximation).
- L₀ = δ⁰ᵀ δ⁰ (sheaf Laplacian). Reconcile by Dirichlet harmonic extension: fix
  TRUSTED cells (exact anchors — e.g. attention layers' KV-derived state, or a few
  exact layers), solve `min_x xᵀ L₀ x` for the rest (sparse SPD solve / CG).

## Prototype (numpy/torch, CPU — no dgx GPU contention)
`research/sbr/M3_sheaf_prototype.py`:
1. Synthetic 48-layer GDN states at a match point M; exact reference via full
   recurrence (reuse sbr_phase1 machinery + a fitted vertical map).
2. Lossy input: contractive-window reconstruction per layer (small τ) → degrades
   slow heads (per real heavy-tailed A_log: ~24% need long memory).
3. Build δ⁰ (horizontal GDN ops + vertical fitted maps), L₀.
4. Harmonic solve with a subset of layers/positions pinned exact.
5. Metric: per-layer cosine(reconciled, ref) vs cosine(lossy, ref) — does
   reconciliation lift the slow-head layers toward ≥0.999? Cost vs exact replay.
6. Ablations: pin set size, vertical-map fit quality, low-rank stalk dim.

## Gate / honesty
- Compare against the trivial baseline: does the L₀ solve beat just using the
  lossy states (and beat plain per-layer smoothing without the sheaf structure)?
- If the win comes only from pinning exact anchors (not the cross-layer coupling),
  say so — that would mean the sheaf adds nothing beyond denser anchoring.
- Real-model validation (dgx2) only after the synthetic prototype shows a clear,
  non-trivial lift; otherwise stop and report the negative result.

## Status
Background research thread (started 2026-06-27). Independent of M1 (shipped,
exact). Does NOT touch dgx2 GPU (CPU prototype first). See [[project_sheaf_based_replaying_2026_06_27]].
