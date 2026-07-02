#!/usr/bin/env python3
"""M3 - 2-D (position x layer) cellular-sheaf L0 reconciliation (CPU, fp32/fp64).

Tests the ONLY regime where sheaf cohomology is non-vacuous for SBR: a 2-D cell
complex (token-position x layer). On a 2-D complex H^1 can be != 0 -> the four
restriction maps around a plaquette need not commute (genuine curvature). The
load-bearing object is the L0 sheaf-Laplacian *least-squares reconciliation*:
cross-layer x cross-position redundancy over-determines the multi-layer SSM
state, so in principle it can denoise CHEAP LOSSY per-layer (contractive-window)
reconstructions.

HYPOTHESIS: a sheaf L0 harmonic solve (lossy data term + exact horizontal GDN
recurrence edges + fitted vertical inter-layer edges + a few pinned exact layers)
recovers free (unpinned) layers with materially higher cosine than the lossy
input AND beats the no-vertical-coupling baseline. If the win comes only from
horizontal structure / exact anchors, the cross-layer sheaf adds nothing beyond
per-layer smoothing that a non-sheaf least-squares matches -> honest negative.

Synthetic, self-contained. Reuses the affine-GDN per-token update
  h_t = A_t h_{t-1} + B_t,   A_t = exp(g_t) I - beta_t k_t k_t^T,  B_t = beta_t k_t v_t^T
from sbr_phase1_glocal_window.py, with a heavy-tailed decay mix
(~76% fast heads in [0.95,0.999], ~24% slow heads in [0.9999,1.0]).

CPU ONLY. Run: python3 M3_sheaf_prototype.py
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

torch.manual_seed(7)
np.random.seed(7)
torch.set_num_threads(min(8, os.cpu_count() or 1))
DEV = torch.device("cpu")
F32 = torch.float32

# ---- geometry -------------------------------------------------------------
L = 48           # layers
H = 4            # heads per layer
KD = 16          # key dim per head
VD = 16          # value dim per head
D = H * KD * VD  # stalk dim (flattened multi-head SSM state) = 1024
N_TOK = 4096     # sequence length; match point M = end
P_GRID = 9       # grid positions (chunk boundaries) over the tail
DELTA = 256      # token spacing between grid positions
M_LATENT = 96    # cross-layer coupling signal dim
SLOW_FRAC = 0.24
SUB = 8          # subsample stride for vertical-map fit data
TAU_MAIN = 48    # representative degraded contractive window
TAUS = [24, 48, 96]


def make_decay():
    lo = torch.empty(L, H, dtype=F32)
    hi = torch.empty(L, H, dtype=F32)
    slow = torch.rand(L, H) < SLOW_FRAC
    lo[slow], hi[slow] = 0.9999, 1.0
    lo[~slow], hi[~slow] = 0.95, 0.999
    return lo, hi, slow


def per_token_decay(lo_h, hi_h):
    u = torch.rand(N_TOK, H, dtype=F32)
    return lo_h.unsqueeze(0) + u * (hi_h - lo_h).unsqueeze(0)


def head_scan(h0, g, beta, k, v, lo, hi, keep=False):
    """Affine-GDN recurrence over tokens [lo,hi). g,beta:(T,H) k,v:(T,H,*)."""
    h = h0.clone()
    states = [] if keep else None
    for t in range(lo, hi):
        a = torch.einsum("hkv,hk->hv", h, k[t])
        vp = (v[t] - a) * beta[t].unsqueeze(-1)
        h = torch.exp(g[t]).view(-1, 1, 1) * h + torch.einsum("hk,hv->hkv", k[t], vp)
        if keep:
            states.append(h.reshape(-1).clone())
    if keep:
        return h, torch.stack(states)
    return h


def gen_layer0():
    k = torch.randn(N_TOK, H, KD, dtype=F32)
    k = k / k.norm(dim=2, keepdim=True).clamp_min(1e-6)
    v = torch.randn(N_TOK, H, VD, dtype=F32)
    beta = torch.sigmoid(torch.randn(N_TOK, H, dtype=F32))
    return k, v, beta


def gen_coupled(prev_s, Wk, Wv, Wb):
    """Layer l+1 streams driven by layer l output signal prev_s (T,M_LATENT).
    Low coupling noise -> the inter-layer relation is mostly linear; residual
    error comes from layer l+1's OWN recurrence memory (genuine curvature)."""
    k = (prev_s @ Wk).reshape(N_TOK, H, KD) + 0.05 * torch.randn(N_TOK, H, KD)
    k = k / k.norm(dim=2, keepdim=True).clamp_min(1e-6)
    v = (prev_s @ Wv).reshape(N_TOK, H, VD) + 0.05 * torch.randn(N_TOK, H, VD)
    beta = torch.sigmoid((prev_s @ Wb).reshape(N_TOK, H) + 0.1 * torch.randn(N_TOK, H))
    return k, v, beta


def build_forward():
    lo_h, hi_h, slow = make_decay()
    g_idx = [N_TOK - 1 - (P_GRID - 1 - i) * DELTA for i in range(P_GRID)]
    P_o = torch.randn(D, M_LATENT, dtype=F32) / np.sqrt(D)
    streams, grid_states, sub_states = [], [], []
    prev_s = None
    for l in range(L):
        g = torch.log(per_token_decay(lo_h[l], hi_h[l]))
        if l == 0:
            k, v, beta = gen_layer0()
        else:
            Wk = torch.randn(M_LATENT, H * KD, dtype=F32) / np.sqrt(M_LATENT)
            Wv = torch.randn(M_LATENT, H * VD, dtype=F32) / np.sqrt(M_LATENT)
            Wb = torch.randn(M_LATENT, H, dtype=F32) / np.sqrt(M_LATENT)
            k, v, beta = gen_coupled(prev_s, Wk, Wv, Wb)
        h0 = torch.zeros(H, KD, VD, dtype=F32)
        _, states_t = head_scan(h0, g, beta, k, v, 0, N_TOK, keep=True)
        grid_states.append(states_t[g_idx])
        sub_states.append(states_t[::SUB])
        prev_s = states_t @ P_o
        streams.append((g, beta, k, v))
    return streams, torch.stack(grid_states), torch.stack(sub_states), slow, g_idx


# ---- horizontal (position) GDN chunk operators ----------------------------
def chunk_gamma(g, beta, k, lo, hi):
    G = torch.eye(KD, dtype=F32).unsqueeze(0).repeat(H, 1, 1)
    eyeH = torch.eye(KD).unsqueeze(0)
    for t in range(lo + 1, hi + 1):
        A = torch.exp(g[t]).view(-1, 1, 1) * eyeH \
            - beta[t].view(-1, 1, 1) * torch.einsum("hi,hj->hij", k[t], k[t])
        G = torch.einsum("hij,hjk->hik", A, G)
    return G


def gamma_apply(G, x):
    return torch.einsum("hij,hjv->hiv", G, x.reshape(H, KD, VD)).reshape(-1)


def gamma_apply_T(G, x):
    return torch.einsum("hji,hjv->hiv", G, x.reshape(H, KD, VD)).reshape(-1)


# ---- vertical inter-layer maps (ridge regression) -------------------------
def fit_vertical(states_per_layer, lam):
    """Fit W_l (D,D), b_l (D): state_{l+1} ~ W_l^T state_l + b_l. Returns
    (W,b) list and per-layer fit cosine. states_per_layer: (L, n, D)."""
    S = states_per_layer.double()
    Ws, qual = [], []
    I = torch.eye(D, dtype=torch.float64)
    for l in range(L - 1):
        X, Y = S[l], S[l + 1]
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        W = torch.linalg.solve(Xc.T @ Xc + lam * I, Xc.T @ Yc)
        b = my - W.T @ mx
        Y_hat = X @ W + b
        q = torch.nn.functional.cosine_similarity(
            (Y - my).flatten(), (Y_hat - my).flatten(), dim=0).item()
        Ws.append((W.float(), b.float()))
        qual.append(q)
    return Ws, qual


# ---- lossy contractive-window reconstruction ------------------------------
def lossy_states(streams, g_idx, tau):
    out = torch.zeros(L, P_GRID, D, dtype=F32)
    z = torch.zeros(H, KD, VD, dtype=F32)
    for l in range(L):
        g, beta, k, v = streams[l]
        for i, gp in enumerate(g_idx):
            lo = max(0, gp + 1 - tau)
            out[l, i] = head_scan(z, g, beta, k, v, lo, gp + 1).reshape(-1)
    return out


# ---- sheaf assembly --------------------------------------------------------
def cid(i, l):
    return i * L + l


def build_h_edges(streams, grid_states, g_idx):
    """Exact horizontal GDN edges: (u, v, Gamma, drive). Drive is EXACT (the
    materialized inter-position input is known in the SBR setting)."""
    edges = []
    for l in range(L):
        g, beta, k, _ = streams[l]
        for i in range(P_GRID - 1):
            G = chunk_gamma(g, beta, k, g_idx[i], g_idx[i + 1])
            c = grid_states[l, i + 1] - gamma_apply(G, grid_states[l, i])
            edges.append((cid(i, l), cid(i + 1, l), G, c))
    return edges


def build_v_edges(Ws):
    edges = []
    for i in range(P_GRID):
        for l in range(L - 1):
            W, b = Ws[l]
            edges.append((cid(i, l), cid(i, l + 1), W, b))
    return edges


def grad_full(x, h_edges, v_edges, lossy, free, wd, wh, wv, use_offset):
    gr = torch.zeros_like(x)
    if use_offset:
        gr[free] += wd * (x[free] - lossy[free])
    else:
        gr[free] += wd * x[free]
    for (u, vtx, G, c) in h_edges:
        r = x[vtx] - gamma_apply(G, x[u])
        if use_offset:
            r = r - c
        gr[vtx] += wh * r
        gr[u] -= wh * gamma_apply_T(G, r)
    for (u, vtx, W, b) in v_edges:
        r = x[vtx] - (W.T @ x[u])
        if use_offset:
            r = r - b
        gr[vtx] += wv * r
        gr[u] -= wv * (W @ r)
    return gr


def cg_solve(applyH, rhs, free, max_iter=300, tol=1e-7):
    x = torch.zeros_like(rhs)
    r = rhs.clone()
    r[~free] = 0.0
    p = r.clone()
    rs = (r * r).sum()
    rs0 = rs.clone()
    it = 0
    for it in range(1, max_iter + 1):
        Ap = applyH(p)
        Ap[~free] = 0.0
        alpha = rs / (p * Ap).sum().clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        r[~free] = 0.0
        rs_new = (r * r).sum()
        if rs_new <= tol * tol * rs0:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, it


def solve_sheaf(h_edges, v_edges, lossy, exact, pinned, wd, wh, wv):
    free = ~pinned
    he = [(u, v, G.double(), c.double()) for (u, v, G, c) in h_edges]
    ve = [(u, v, W.double(), b.double()) for (u, v, W, b) in v_edges]
    lossy_d = lossy.double()
    x0 = torch.zeros_like(lossy_d)
    x0[pinned] = exact.double()[pinned]
    rhs = -grad_full(x0, he, ve, lossy_d, free, wd, wh, wv, True)
    rhs[~free] = 0.0

    def applyH(x):
        xx = x.clone()
        xx[~free] = 0.0
        out = grad_full(xx, he, ve, lossy_d, free, wd, wh, wv, False)
        out[~free] = 0.0
        return out

    dx, it = cg_solve(applyH, rhs, free)
    x = x0.clone()
    x[free] = x0[free] + dx[free]
    return x.float(), it


# ---- metrics --------------------------------------------------------------
def cell_cos(states, exact):
    return torch.nn.functional.cosine_similarity(states, exact, dim=1)


def slow_dim_mask(slow):
    """(L, D) bool: dims belonging to slow heads of each layer."""
    m = torch.zeros(L, D, dtype=torch.bool)
    blk = KD * VD
    for l in range(L):
        for hh in range(H):
            if slow[l, hh]:
                m[l, hh * blk:(hh + 1) * blk] = True
    return m


def subspace_cos(states, exact, layer_of, sdmask, want_slow):
    """Mean cosine over slow- (or fast-) head dims, per cell, averaged."""
    vals = []
    for c in range(states.shape[0]):
        l = layer_of[c].item()
        mask = sdmask[l] if want_slow else ~sdmask[l]
        if mask.sum() == 0:
            continue
        a, b = states[c][mask], exact[c][mask]
        vals.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    return float(np.mean(vals)) if vals else float("nan")


def pin_mask(pin_layers):
    m = torch.zeros(P_GRID * L, dtype=torch.bool)
    for i in range(P_GRID):
        for l in pin_layers:
            m[cid(i, l)] = True
    return m


def plaquette_curvature(h_edges, v_edges, exact):
    hmap = {(u, v): (G, c) for (u, v, G, c) in h_edges}
    vmap = {(u, v): (W, b) for (u, v, W, b) in v_edges}
    rel = []
    for i in range(P_GRID - 1):
        for l in range(L - 1):
            x = exact[cid(i, l)]
            G1, c1 = hmap[(cid(i, l), cid(i + 1, l))]
            W1, b1 = vmap[(cid(i, l), cid(i, l + 1))]
            G2, c2 = hmap[(cid(i, l + 1), cid(i + 1, l + 1))]
            W2, b2 = vmap[(cid(i + 1, l), cid(i + 1, l + 1))]
            a = W2.T @ (gamma_apply(G1, x) + c1) + b2        # H then V
            bb = gamma_apply(G2, (W1.T @ x + b1)) + c2       # V then H
            rel.append(((a - bb).norm() / a.norm().clamp_min(1e-6)).item())
    return float(np.mean(rel))


def report(name, c, free, layer_of, sdmask, states, exact):
    cf = c[free]
    sl = subspace_cos(states[free], exact[free], layer_of[free], sdmask, True)
    fa = subspace_cos(states[free], exact[free], layer_of[free], sdmask, False)
    print(f"{name:<26}{cf.min():>9.4f}{cf.mean():>9.4f}{cf.median():>9.4f}"
          f"{(cf >= 0.999).float().mean():>9.3f}{sl:>11.4f}{fa:>11.4f}")
    return cf.mean().item(), sl


def main():
    print("=== M3 2-D sheaf L0 reconciliation prototype (CPU) ===")
    print(f"L={L} H={H} KD=VD={KD} D={D} N_TOK={N_TOK} P_GRID={P_GRID} "
          f"DELTA={DELTA} slow_frac={SLOW_FRAC}")
    streams, grid_states, sub_states, slow, g_idx = build_forward()
    exact = grid_states.permute(1, 0, 2).reshape(-1, D)
    layer_of = torch.tensor([l for i in range(P_GRID) for l in range(L)])
    sdmask = slow_dim_mask(slow)
    print(f"slow heads total: {int(slow.sum())}/{L * H} "
          f"({100 * slow.float().mean():.0f}%); layers with >=1 slow head: "
          f"{int(slow.any(1).sum())}/{L}")

    Ws, qual = fit_vertical(sub_states, lam=1e-1)
    print(f"vertical ridge fit cosine: mean={np.mean(qual):.4f} "
          f"min={np.min(qual):.4f} max={np.max(qual):.4f}")
    h_edges = build_h_edges(streams, grid_states, g_idx)
    v_edges = build_v_edges(Ws)
    print(f"mean plaquette curvature (rel. H/V non-commutation): "
          f"{plaquette_curvature(h_edges, v_edges, exact):.4f}")

    pin_layers = list(range(0, L, 4))
    pinned = pin_mask(pin_layers)
    free = ~pinned
    hdr = (f"{'method':<26}{'min':>9}{'mean':>9}{'median':>9}{'>=.999':>9}"
           f"{'slow-sub':>11}{'fast-sub':>11}")
    print(f"\n### MAIN: tau={TAU_MAIN}, pin {len(pin_layers)}/{L} layers exact "
          f"(every 4th). Metrics over FREE (unpinned) cells.")
    print(hdr)
    lossy = lossy_states(streams, g_idx, TAU_MAIN)
    lossy_f = lossy.permute(1, 0, 2).reshape(-1, D)
    report("lossy input", cell_cos(lossy_f, exact), free, layer_of, sdmask, lossy_f, exact)
    xh, ih = solve_sheaf(h_edges, v_edges, lossy_f, exact, pinned, 1.0, 10.0, 0.0)
    report("horiz-only (no x-layer)", cell_cos(xh, exact), free, layer_of, sdmask, xh, exact)
    xf, iff = solve_sheaf(h_edges, v_edges, lossy_f, exact, pinned, 1.0, 10.0, 3.0)
    report("FULL 2-D sheaf", cell_cos(xf, exact), free, layer_of, sdmask, xf, exact)
    print(f"CG iters: horiz={ih} full={iff}")
    nfl = L - len(pin_layers)
    print(f"cost: free DOF={int(free.sum()) * D}  | exact full-replay of unpinned "
          f"layers={nfl * N_TOK} tok-updates  (lossy window={nfl * P_GRID * TAU_MAIN})")

    print("\n=== ABLATIONS ===")
    print(f"(a) tau sweep (mean cell cosine / slow-subspace cosine over free), pin 12/48")
    print(f"{'tau':<8}{'lossy':>18}{'horiz':>18}{'full':>18}")
    for tau in TAUS:
        lz = lossy_states(streams, g_idx, tau).permute(1, 0, 2).reshape(-1, D)
        cl = cell_cos(lz, exact)[free].mean().item()
        sl_l = subspace_cos(lz[free], exact[free], layer_of[free], sdmask, True)
        xhh, _ = solve_sheaf(h_edges, v_edges, lz, exact, pinned, 1.0, 10.0, 0.0)
        xff, _ = solve_sheaf(h_edges, v_edges, lz, exact, pinned, 1.0, 10.0, 3.0)
        ch = cell_cos(xhh, exact)[free].mean().item()
        sh = subspace_cos(xhh[free], exact[free], layer_of[free], sdmask, True)
        cf = cell_cos(xff, exact)[free].mean().item()
        sf = subspace_cos(xff[free], exact[free], layer_of[free], sdmask, True)
        print(f"{tau:<8}{cl:>9.4f}/{sl_l:<8.4f}{ch:>9.4f}/{sh:<8.4f}{cf:>9.4f}/{sf:<8.4f}")

    print(f"\n(b) pin-set size sweep (tau={TAU_MAIN}); mean cell cosine over free")
    print(f"{'pinned':<12}{'lossy':>9}{'horiz':>9}{'full':>9}{'full-horiz':>12}")
    lz = lossy_states(streams, g_idx, TAU_MAIN).permute(1, 0, 2).reshape(-1, D)
    for step in [2, 3, 4, 8, 16]:
        pl = list(range(0, L, step))
        pm = pin_mask(pl)
        fr = ~pm
        cl = cell_cos(lz, exact)[fr].mean().item()
        xhh, _ = solve_sheaf(h_edges, v_edges, lz, exact, pm, 1.0, 10.0, 0.0)
        xff, _ = solve_sheaf(h_edges, v_edges, lz, exact, pm, 1.0, 10.0, 3.0)
        ch = cell_cos(xhh, exact)[fr].mean().item()
        cf = cell_cos(xff, exact)[fr].mean().item()
        print(f"{len(pl):>3}/{L:<8}{cl:>9.4f}{ch:>9.4f}{cf:>9.4f}{cf - ch:>12.4f}")

    print(f"\n(c) vertical weight w_v sweep (tau={TAU_MAIN}, pin 12/48); w_v=0 == horiz-only")
    print(f"{'w_v':<8}{'cell-mean':>11}{'slow-sub':>11}{'>=.999':>9}")
    for wv in [0.0, 0.3, 1.0, 3.0, 10.0]:
        xff, _ = solve_sheaf(h_edges, v_edges, lz, exact, pinned, 1.0, 10.0, wv)
        cc = cell_cos(xff, exact)
        ss = subspace_cos(xff[free], exact[free], layer_of[free], sdmask, True)
        print(f"{wv:<8.1f}{cc[free].mean():>11.4f}{ss:>11.4f}"
              f"{(cc[free] >= 0.999).float().mean():>9.3f}")

    print(f"\n(d) vertical-map fit quality sweep (ridge lambda; tau={TAU_MAIN}, pin 12/48, w_v=3)")
    print(f"   incl. ORACLE-V = best linear l->l+1 map fit on EXACT grid states (ceiling)")
    print(f"{'fit':<14}{'fitcos':>9}{'full-mean':>11}{'full-slow':>11}{'vs horiz':>11}")
    base_h = cell_cos(xh, exact)[free].mean().item()
    for lam in [1e-3, 1e-1, 1e1, 1e3]:
        Ws2, q2 = fit_vertical(sub_states, lam=lam)
        ve2 = build_v_edges(Ws2)
        xff, _ = solve_sheaf(h_edges, ve2, lz, exact, pinned, 1.0, 10.0, 3.0)
        cc = cell_cos(xff, exact)[free].mean().item()
        ss = subspace_cos(xff[free], exact[free], layer_of[free], sdmask, True)
        print(f"ridge {lam:<8.0e}{np.mean(q2):>9.4f}{cc:>11.4f}{ss:>11.4f}{cc - base_h:>11.4f}")
    # oracle: fit on the exact grid states themselves (upper bound)
    grid_layer_first = exact.reshape(P_GRID, L, D).permute(1, 0, 2)  # (L,P_GRID,D)
    Wso, qo = fit_vertical(grid_layer_first, lam=1e-4)
    veo = build_v_edges(Wso)
    print(f"oracle plaquette curvature: {plaquette_curvature(h_edges, veo, exact):.4f}")
    for wv in [1.0, 3.0, 10.0]:
        xff, _ = solve_sheaf(h_edges, veo, lz, exact, pinned, 1.0, 10.0, wv)
        cc = cell_cos(xff, exact)[free].mean().item()
        ss = subspace_cos(xff[free], exact[free], layer_of[free], sdmask, True)
        print(f"oracle w_v={wv:<5.1f}{np.mean(qo):>9.4f}{cc:>11.4f}{ss:>11.4f}{cc - base_h:>11.4f}")

    print("\nVERDICT logic: FULL must beat BOTH 'lossy' and 'horiz-only' by a real "
          "margin (esp. slow-subspace) for the 2-D sheaf to do non-trivial work. "
          "If full<=horiz, the cross-layer sheaf adds nothing beyond per-layer "
          "smoothing a non-sheaf LS already gives. ORACLE-V isolates whether the "
          "blocker is vertical-map quality vs the sheaf concept itself.")
    print("DONE")


if __name__ == "__main__":
    main()
