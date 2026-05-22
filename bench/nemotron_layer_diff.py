#!/usr/bin/env python3
"""Per-layer divergence comparator: Atlas (ATLAS_NEMO_DUMP) vs HF oracle.

Both sides write headerless little-endian f32 .bin of the LAST token's
post-layer residual-stream hidden vector. This computes flat cosine,
relative L2, and max-abs-diff per layer, flags the first divergent layer,
and characterizes the growth curve.

Usage: nemotron_layer_diff.py <atlas_dir> <hf_dir>
"""
import json
import pathlib
import sys

import numpy as np

A = pathlib.Path(sys.argv[1])
H = pathlib.Path(sys.argv[2])

# hybrid_override_pattern for Nano-30B (52 layers).
PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
KIND = {"M": "mamba", "E": "moe", "*": "attn"}


def rd(p):
    p = pathlib.Path(p)
    if not p.exists():
        return None
    return np.frombuffer(p.read_bytes(), "<f4").astype(np.float64)


def metrics(a, b):
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb + 1e-12))
    rel = float(np.linalg.norm(a - b) / (nb + 1e-12))
    maxd = float(np.max(np.abs(a - b)))
    return cos, rel, maxd, na, nb


def main():
    nlayers = len(PATTERN)
    print(f"{'layer':>6} {'kind':>6} {'cosine':>10} {'relL2':>10} "
          f"{'maxdiff':>10} {'|atlas|':>10} {'|hf|':>10}")
    print("-" * 70)
    first_div = None
    rows = []
    # NOTE: HF only exposes layer outputs L0 .. L{nlayers-2}. The last
    # layer's pre-norm residual is not in hidden_states (HF appends
    # post-norm_f as the final entry instead), so hf_L{nlayers-1}.bin is
    # intentionally absent -- compare the final-norm output instead.
    for i in range(nlayers):
        a = rd(A / f"atlas_L{i}.bin")
        b = rd(H / f"hf_L{i}.bin")
        if a is None or b is None:
            note = ("  (HF exposes no pre-norm residual for the last layer "
                    "-- see final_norm row)" if i == nlayers - 1 else "")
            print(f"{i:>6} {KIND.get(PATTERN[i],'?'):>6}   MISSING "
                  f"atlas={a is not None} hf={b is not None}{note}")
            continue
        cos, rel, maxd, na, nb = metrics(a, b)
        k = KIND.get(PATTERN[i], "?")
        flag = ""
        if cos < 0.999 and first_div is None:
            first_div = i
            flag = "  <-- FIRST DIVERGENCE"
        print(f"{i:>6} {k:>6} {cos:>10.6f} {rel:>10.6f} "
              f"{maxd:>10.4f} {na:>10.2f} {nb:>10.2f}{flag}")
        rows.append((i, k, cos, rel))

    print("-" * 70)
    # final norm + logits
    for name in ("final_norm", "logits"):
        a = rd(A / f"atlas_{name}.bin")
        b = rd(H / f"hf_{name}.bin")
        if a is not None and b is not None:
            cos, rel, maxd, na, nb = metrics(a, b)
            print(f"{name:>13} cos={cos:.6f} relL2={rel:.6f} "
                  f"maxdiff={maxd:.4f} |atlas|={na:.2f} |hf|={nb:.2f}")

    # top token agreement
    af = A / "atlas_logits.bin"
    hf = H / "hf_logits.bin"
    if af.exists() and hf.exists():
        al = rd(af)
        hl = rd(hf)
        at = np.argsort(-al)[:10]
        ht = np.argsort(-hl)[:10]
        print(f"\natlas top-10 token ids: {at.tolist()}")
        print(f"hf    top-10 token ids: {ht.tolist()}")
        print(f"argmax match: atlas={int(at[0])} hf={int(ht[0])} "
              f"-> {'SAME' if at[0]==ht[0] else 'DIFFERENT'}")
        overlap = len(set(at.tolist()) & set(ht.tolist()))
        print(f"top-10 overlap: {overlap}/10")

    print()
    if first_div is None:
        print("VERDICT: no layer drops below cosine 0.999 -> numerically "
              "faithful within NVFP4 quant noise.")
    else:
        print(f"VERDICT: first divergence at layer {first_div} "
              f"({KIND.get(PATTERN[first_div],'?')}).")
        # growth characterization
        cosines = [c for (_, _, c, _) in rows if _ is not None]


if __name__ == "__main__":
    main()
