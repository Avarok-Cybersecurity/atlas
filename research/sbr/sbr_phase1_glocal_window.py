#!/usr/bin/env python3
"""SBR Phase-1 — contractive-window / glocal-split measurement (CPU, fp32).

Decides the implementation primitive for GDN. Dense operator-transport loses to
replay for GDN (compose is O(chunks·d^3) vs replay O(chunks·64·d^2)), and GDN
operators are not more compact than the state. The real lever is GLOCALITY via
CONTRACTIVITY:

  h_M = A_{C->M} h_C + sum_j A_{j+1->M} (contribution_j)

Each token j's contribution is weighted by the decay product A_{j+1->M} ~ exp(sum g).
For contractive heads (decay<1) tokens far before M contribute ~0, so:

  * LOCAL: reconstruct h_M by replaying only the last tau tokens before M from a
    ZERO state — error = dropped far-field, bounded by cumulative decay. NO anchor
    near M needed -> warm-hit cost O(tau), FLAT in (M-C).
  * GLOBAL (slow modes): weak-forget heads (decay~1) keep long memory; they need a
    cheap always-maintained coarse summary (few slow modes) = the "global section".

This script measures, on a realistic MIXED decay population:
  (1) per-head reconstruction cosine vs window tau (local replay from zero),
  (2) what fraction of heads are "fast" (tau<=512 reaches 0.999) vs "slow"
      (need the global summary),
  (3) the slow-mode count: how few modes carry the long-range info.

Run: python3 sbr_phase1_glocal_window.py
"""
import torch

torch.manual_seed(1)
DT = torch.float32
K_DIM = 128
V_DIM = 128
N_HEADS = 32
N_TOK = 16384  # deep conversation; "match point" M = end


def mixed_decay(n_tok, n_heads, weak_frac=0.25):
    """Per-head, per-token decay a_t=exp(g_t). Most heads forget; a weak_frac
    minority are near-1 (long-range)."""
    a = torch.empty(n_heads, n_tok, dtype=DT)
    n_weak = int(round(weak_frac * n_heads))
    # fast-forgetting heads: decay in [0.95, 0.999]
    a[: n_heads - n_weak] = torch.empty(n_heads - n_weak, n_tok, dtype=DT).uniform_(0.95, 0.999)
    # weak-forget (long-range) heads: decay in [0.9995, 1.0]
    a[n_heads - n_weak :] = torch.empty(n_weak, n_tok, dtype=DT).uniform_(0.9995, 1.0)
    return a, n_weak


def head_stream(n_tok, decay_row):
    g = torch.log(decay_row)
    beta = torch.sigmoid(torch.randn(n_tok, dtype=DT))
    k = torch.randn(n_tok, K_DIM, dtype=DT)
    k = k / k.norm(dim=1, keepdim=True).clamp_min(1e-6)
    v = torch.randn(n_tok, V_DIM, dtype=DT)
    return g, beta, k, v


def scan(h0, g, beta, k, v, lo, hi):
    h = h0.clone()
    for t in range(lo, hi):
        a_t = h.transpose(0, 1) @ k[t]
        vp = (v[t] - a_t) * beta[t]
        h = torch.exp(g[t]) * h + torch.outer(k[t], vp)
    return h


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def main():
    print("=== SBR Phase-1 contractive-window / glocal-split (CPU fp32) ===")
    decay, n_weak = mixed_decay(N_TOK, N_HEADS)
    taus = [64, 128, 256, 512, 1024, 2048, 4096]
    fast_heads = 0
    slow_heads = []
    # aggregate cosine per tau across heads
    agg = {t: [] for t in taus}
    h_full_norm_slow = []

    for hd in range(N_HEADS):
        g, beta, k, v = head_stream(N_TOK, decay[hd])
        h0 = torch.zeros(K_DIM, V_DIM, dtype=DT)
        h_full = scan(h0, g, beta, k, v, 0, N_TOK)  # exact, from true start
        per_tau_cos = {}
        for tau in taus:
            lo = max(0, N_TOK - tau)
            h_loc = scan(torch.zeros(K_DIM, V_DIM, dtype=DT), g, beta, k, v, lo, N_TOK)
            c = cos(h_full, h_loc)
            agg[tau].append(c)
            per_tau_cos[tau] = c
        # classify: fast if tau<=512 reaches 0.999
        if per_tau_cos[512] >= 0.999:
            fast_heads += 1
        else:
            slow_heads.append((hd, per_tau_cos[512], per_tau_cos[4096]))

    print(f"\nHeads: {N_HEADS} ({n_weak} seeded weak-forget). "
          f"FAST (tau<=512 -> cos>=0.999): {fast_heads}/{N_HEADS} "
          f"({100*fast_heads/N_HEADS:.0f}%)")
    print("\nAggregate reconstruction cosine (local replay from zero, last-tau):")
    print("  tau   |  min      mean     median")
    for t in taus:
        col = torch.tensor(agg[t])
        print(f"  {t:5d} | {col.min():.6f}  {col.mean():.6f}  {col.median():.6f}")

    if slow_heads:
        print(f"\nSLOW heads needing global summary ({len(slow_heads)}): "
              f"[head, cos@512, cos@4096]")
        for hd, c5, c4 in slow_heads[:12]:
            print(f"    head {hd:2d}: {c5:.5f} -> {c4:.5f}")

    # slow-mode rank: for one weak-forget head, how many SVD modes of the
    # far-field state carry the long-range info?
    hd = N_HEADS - 1  # a weak-forget head
    g, beta, k, v = head_stream(N_TOK, decay[hd])
    h_far = scan(torch.zeros(K_DIM, V_DIM, dtype=DT), g, beta, k, v, 0, N_TOK - 512)
    sv = torch.linalg.svdvals(h_far)
    energy = (sv.cumsum(0) / sv.sum())
    r90 = int((energy < 0.90).sum()) + 1
    r99 = int((energy < 0.99).sum()) + 1
    print(f"\nSlow-mode rank (weak-forget head {hd}, far-field h before last-512):")
    print(f"    rank for 90% energy: {r90}/{K_DIM}   99% energy: {r99}/{K_DIM}")
    print(f"    -> global summary needs ~{r90}-{r99} modes, not full {K_DIM}x{V_DIM}")

    print("\nINTERPRETATION: if FAST% is high and slow-mode rank is low, the glocal "
          "split (local-tau replay + small slow-mode global summary) gives FLAT-in-(M-C) "
          "warm-hit cost. This is the SBR primitive to implement, not dense segment-tree.")


if __name__ == "__main__":
    main()
