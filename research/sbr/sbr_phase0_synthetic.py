#!/usr/bin/env python3
"""SBR Phase-0 synthetic kill-test (CPU, fp32).

Validates the core Sheaf-Based-Replaying claim BEFORE any kernel work:
  the GDN per-token update is affine in the state h,
      h_t = A_t h_{t-1} + B_t,   A_t = exp(g_t) I - beta_t k_t k_t^T,  B_t = beta_t k_t v_t^T
  so per-chunk affine operators (A_chunk, B_chunk) compose associatively and an
  operator-transport reconstruction of h_M (segment-tree compose of chunk ops,
  applied once to the anchor state h_C) must equal sequential token replay.

Decisive questions:
  (1) Does the segment-tree (balanced, reassociated) compose hold cosine >= 0.999
      and small max-abs-err vs the sequential per-token scan, at realistic depth?
  (2) Sanity: does the affine operator form reproduce the reference recurrence bit-closely?
  (3) Invertibility audit: fraction of per-token A_t that are well-conditioned
      (eligible for O(1) group-subtraction on evict) under realistic gate/beta.

Run: python3 sbr_phase0_synthetic.py
"""
import torch

torch.manual_seed(0)
DEV = "cpu"  # never touch dgx1 GPU
DT = torch.float32

# Qwen3-Next GDN per-head dims
K_DIM = 128
V_DIM = 128
CHUNK = 64


def make_token_stream(n_tok, decay_lo=0.95, decay_hi=0.9999, beta_mode="sigmoid"):
    """Realistic per-token GDN inputs for one head."""
    # decay a_t = exp(g_t) in [decay_lo, decay_hi] (close to 1, per GDN behaviour)
    a = torch.empty(n_tok, dtype=DT).uniform_(decay_lo, decay_hi)
    g = torch.log(a)
    # beta_t in (0,1): sigmoid of a normal logit (matches BA projection -> sigmoid)
    if beta_mode == "sigmoid":
        beta = torch.sigmoid(torch.randn(n_tok, dtype=DT))
    else:
        beta = torch.empty(n_tok, dtype=DT).uniform_(0.0, 1.0)
    # keys L2-normalized (GDN normalizes k); values ~ N(0,1)
    k = torch.randn(n_tok, K_DIM, dtype=DT)
    k = k / k.norm(dim=1, keepdim=True).clamp_min(1e-6)
    v = torch.randn(n_tok, V_DIM, dtype=DT)
    return g, beta, k, v


def reference_scan(h0, g, beta, k, v):
    """Exact per-token recurrence straight from the kernel comment."""
    h = h0.clone()
    for t in range(g.shape[0]):
        a_t = (h.transpose(0, 1) @ k[t])          # [V_DIM]  = h^T k
        vp = (v[t] - a_t) * beta[t]               # v'_t     [V_DIM]
        h = torch.exp(g[t]) * h + torch.outer(k[t], vp)  # k ⊗ v'
    return h


def token_affine(g_t, beta_t, k_t, v_t):
    """Per-token affine operator (A_t, B_t) with h_t = A_t h_{t-1} + B_t."""
    A = torch.exp(g_t) * torch.eye(K_DIM, dtype=DT) - beta_t * torch.outer(k_t, k_t)
    B = beta_t * torch.outer(k_t, v_t)            # [K_DIM, V_DIM]
    return A, B


def compose(op2, op1):
    """(A2,B2) ∘ (A1,B1) = (A2 A1, A2 B1 + B2). Applies op1 first, then op2."""
    A1, B1 = op1
    A2, B2 = op2
    return (A2 @ A1, A2 @ B1 + B2)


def chunk_operator(g, beta, k, v, lo, hi):
    """Compose per-token ops over [lo, hi) sequentially -> one chunk affine op."""
    A = torch.eye(K_DIM, dtype=DT)
    B = torch.zeros(K_DIM, V_DIM, dtype=DT)
    for t in range(lo, hi):
        At, Bt = token_affine(g[t], beta[t], k[t], v[t])
        A, B = compose((At, Bt), (A, B))
    return (A, B)


def segtree_compose(ops):
    """Balanced (reassociated) reduction of a list of affine ops, left-to-right order."""
    # pairwise balanced tree to maximize reassociation difference vs linear fold
    cur = ops
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(compose(cur[i + 1], cur[i]))  # op[i] first, then op[i+1]
        if len(cur) % 2 == 1:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def run_case(n_tok, label):
    g, beta, k, v = make_token_stream(n_tok)
    h0 = torch.randn(K_DIM, V_DIM, dtype=DT) * 0.1  # anchor state h_C

    h_ref = reference_scan(h0, g, beta, k, v)

    # sanity: per-token affine fold reproduces reference
    A = torch.eye(K_DIM, dtype=DT); B = torch.zeros(K_DIM, V_DIM, dtype=DT)
    for t in range(n_tok):
        At, Bt = token_affine(g[t], beta[t], k[t], v[t])
        A, B = compose((At, Bt), (A, B))
    h_affine = A @ h0 + B
    cos_affine = cos(h_ref, h_affine)
    mae_affine = (h_ref - h_affine).abs().max().item()

    # transport: chunk ops -> balanced segment-tree compose -> apply to anchor
    n_chunks = (n_tok + CHUNK - 1) // CHUNK
    ops = [chunk_operator(g, beta, k, v, c * CHUNK, min((c + 1) * CHUNK, n_tok))
           for c in range(n_chunks)]
    A_tot, B_tot = segtree_compose(ops)
    h_transport = A_tot @ h0 + B_tot
    cos_tr = cos(h_ref, h_transport)
    mae_tr = (h_ref - h_transport).abs().max().item()
    rel_tr = ((h_ref - h_transport).norm() / h_ref.norm().clamp_min(1e-9)).item()

    print(f"[{label}] n_tok={n_tok} chunks={n_chunks} |h_ref|={h_ref.norm():.2f}")
    print(f"    affine-fold   : cos={cos_affine:.8f}  maxabs={mae_affine:.3e}")
    print(f"    segtree-transp: cos={cos_tr:.8f}  maxabs={mae_tr:.3e}  rel_l2={rel_tr:.3e}")
    return cos_tr


def invertibility_audit(n_tok=20000):
    """Fraction of per-token A_t eligible for O(1) group-subtraction.
    A_t = exp(g) I - beta k k^T has eigenvalues {exp(g) (mult K-1), exp(g)-beta|k|^2}.
    Well-conditioned iff exp(g)-beta|k|^2 bounded away from 0 (|k|=1 here)."""
    g, beta, k, v = make_token_stream(n_tok)
    a = torch.exp(g)                       # exp(g)
    lam_min = a - beta                     # |k|^2 = 1
    cond = a / lam_min.abs().clamp_min(1e-12)
    eligible = (lam_min.abs() > 1e-3).float().mean().item()
    near_sing = (lam_min.abs() <= 1e-3).float().mean().item()
    print("\n[invertibility audit]  (per-token A_t = exp(g)I - beta kk^T)")
    print(f"    decay exp(g): min={a.min():.4f} max={a.max():.4f}")
    print(f"    beta        : min={beta.min():.4f} max={beta.max():.4f}")
    print(f"    eligible (|lam_min|>1e-3): {eligible*100:.1f}%   near-singular: {near_sing*100:.2f}%")
    print(f"    condition number: median={cond.median():.2f} p99={cond.quantile(0.99):.2f} max={cond.max():.2f}")


if __name__ == "__main__":
    print("=== SBR Phase-0 synthetic kill-test (CPU fp32) ===")
    worst = 1.0
    for n in (512, 2048, 8192, 16384):
        worst = min(worst, run_case(n, f"depth"))
    invertibility_audit()
    print(f"\nGATE: worst transport cosine across depths = {worst:.8f}  "
          f"({'PASS' if worst >= 0.999 else 'FAIL'} @ 0.999)")
