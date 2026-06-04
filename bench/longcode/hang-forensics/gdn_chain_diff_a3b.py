#!/usr/bin/env python3
"""GDN layer-0 chain diff for Qwen3.6-35B-A3B-FP8.

Adapted from gdn_chain_diff2.py with A3B shapes:
  q=2048 (16 k-heads * 128) ; k=2048 (16 k-heads * 128) ; v=4096 (32 v-heads * 128)
  conv on [q|k|v] = 8192 (head-major within each segment)
  recurrence + gated-RMSNorm in value space 4096 (32 v-heads * 128, HEAD-MAJOR)

Usage:  python3 gdn_chain_diff_a3b.py /workspace/dumps/atlas-gdnsub-a3b /home/claude/gdnref_a3b
"""
import sys, pathlib
import numpy as np

A, H = sys.argv[1], sys.argv[2]


def rd(p):
    p = pathlib.Path(p)
    if not p.exists(): return None
    return (np.frombuffer(p.read_bytes(), "<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def cos_rel(a, b):
    m = min(len(a), len(b)); a, b = a[:m], b[:m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    c = float(a @ b / (na * nb + 1e-12))
    r = float(np.linalg.norm(a - b) / (nb + 1e-12))
    return c, r


def per_head_cos(a, b, hd):
    nh = len(a) // hd
    A2 = a[:nh * hd].reshape(nh, hd); B2 = b[:nh * hd].reshape(nh, hd)
    na = np.linalg.norm(A2, axis=1); nb = np.linalg.norm(B2, axis=1)
    pc = (A2 * B2).sum(1) / (na * nb + 1e-12)
    return pc, A2, B2, na, nb


def best_match_cos(A2, B2):
    nh = A2.shape[0]
    na = np.linalg.norm(A2, axis=1) + 1e-12
    nb = np.linalg.norm(B2, axis=1) + 1e-12
    S = (A2 @ B2.T) / np.outer(na, nb)
    used = set(); matched = []
    order = np.argsort(-S.max(axis=1))
    for i in order:
        row = S[i].copy()
        for j in used: row[j] = -2.0
        j = int(np.argmax(row)); used.add(j); matched.append((int(i), j, float(S[i, j])))
    cvals = np.array([c for _, _, c in matched])
    ident = np.array([1 if i == j else 0 for i, j, _ in matched]).mean()
    return cvals.mean(), cvals.min(), float((cvals > 0.9).mean()), float(ident)


def report(name, a, b, segs, head_dim):
    print(f"\n--- {name} ---")
    if a is None or b is None:
        print(f"  MISSING  atlas={a is not None}  hf={b is not None}")
        return
    print(f"  atlas_n={len(a)}  hf_n={len(b)}")
    if len(a) != len(b):
        c, r = cos_rel(a, b)
        print(f"  SIZE MISMATCH -> trunc flat cos={c:+.5f} relL2={r:.4f}")
        return
    c, r = cos_rel(a, b)
    print(f"  flat                 cos={c:+.5f}  relL2={r:.4f}")
    off = 0
    for sn, sl in segs:
        if off + sl <= len(a):
            cs, rs = cos_rel(a[off:off + sl], b[off:off + sl])
            pc, A2, B2, na_arr, nb_arr = per_head_cos(a[off:off + sl], b[off:off + sl], head_dim)
            bm_mean, bm_min, bm_frac, ident = best_match_cos(A2, B2)
            # Per-head magnitude statistics — methodology §5 fingerprint
            ratios = na_arr / (nb_arr + 1e-12)
            print(f"  seg {sn:<3}({sl:5d})    cos={cs:+.5f} relL2={rs:.4f}  "
                  f"| aligned-head cos mean={pc.mean():+.4f} min={pc.min():+.4f} "
                  f">0.9:{int((pc > 0.9).sum())}/{len(pc)}")
            print(f"  {'':14s}            best-match-head cos mean={bm_mean:+.4f} "
                  f"min={bm_min:+.4f} >0.9:{bm_frac:.2f} identity-frac={ident:.2f}")
            print(f"  {'':14s}            per-head |A|/|B| ratio  mean={ratios.mean():+.4f} "
                  f"std={ratios.std():.4f} min={ratios.min():+.4f} max={ratios.max():+.4f}")
        off += sl


print("=== GDN layer-0 chain diff (A3B; Atlas vs source-grounded HF oracle) ===")
print(f"atlas={A}")
print(f"hf   ={H}")

# A3B: conv segments q|k|v = 2048|2048|4096 = 8192 total
# recurrence/gnorm value-space single v segment = 4096
report("conv1d   (Atlas conv  ~ HF conv1d post-silu)",
       rd(f"{A}/gdnsub_step0_L0_conv.bin"), rd(f"{H}/gdnref_L0_conv1d.bin"),
       [("q", 2048), ("k", 2048), ("v", 4096)], 128)
report("l2norm   (Atlas l2 (post-l2norm of q,k)  ~ HF n/a — Atlas-only sanity)",
       rd(f"{A}/gdnsub_step0_L0_l2.bin"), rd(f"{H}/gdnref_L0_conv1d.bin"),
       [("q", 2048), ("k", 2048), ("v", 4096)], 128)
report("recur    (Atlas gdn   ~ HF recur_in/pre-norm)",
       rd(f"{A}/gdnsub_step0_L0_gdn.bin"), rd(f"{H}/gdnref_L0_recur_in.bin"),
       [("v", 4096)], 128)
report("gnorm    (Atlas gnorm ~ HF norm gated-rmsnorm)",
       rd(f"{A}/gdnsub_step0_L0_gnorm.bin"), rd(f"{H}/gdnref_L0_norm.bin"),
       [("v", 4096)], 128)

print("\n--- norms (context) ---")
for lbl, p in [
    ("atlas conv ", f"{A}/gdnsub_step0_L0_conv.bin"),
    ("atlas l2   ", f"{A}/gdnsub_step0_L0_l2.bin"),
    ("atlas gdn  ", f"{A}/gdnsub_step0_L0_gdn.bin"),
    ("atlas gnorm", f"{A}/gdnsub_step0_L0_gnorm.bin"),
    ("hf in_proj_qkv", f"{H}/gdnref_L0_in_proj_qkv.bin"),
    ("hf conv1d  ", f"{H}/gdnref_L0_conv1d.bin"),
    ("hf recur_in", f"{H}/gdnref_L0_recur_in.bin"),
    ("hf norm    ", f"{H}/gdnref_L0_norm.bin"),
    ("hf out_proj", f"{H}/gdnref_L0_out_proj.bin"),
]:
    v = rd(p)
    nv = None if v is None else round(float(np.linalg.norm(v)), 4)
    ln = None if v is None else len(v)
    print(f"  |{lbl}| = {nv}  (n={ln})")
