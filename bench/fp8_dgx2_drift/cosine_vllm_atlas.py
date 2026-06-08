#!/usr/bin/env python3
"""Apples-to-apples per-layer cosine: Atlas-FP8 vs vLLM-FP8.

Both engines run the SAME FP8 model (Qwen3.6-35B-A3B-FP8) on the SAME ~10378-token
prompt. vLLM passes the opencode harness 10/10; Atlas drifts. This finds the layer
where Atlas's residual stream first diverges from vLLM's (the FP8 implementation gap,
isolated from FP8 quant noise itself — both engines have the same quant noise).

Inputs in /workspace/atlas-dumps/fp8native_dgx2/:
  vllm_L{0..39}.bin   - vLLM-FP8 per-layer last-token residual (f32 LE, 2048)
  atlas_L{0..39}.bin  - Atlas-FP8 per-layer last-token residual (f32 LE, 2048)
  (optional) vllm_logits.bin / atlas_logits.bin - final logits over vocab
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

OUT = pathlib.Path("/workspace/atlas-dumps/fp8native_dgx2")
N_LAYERS = 40
# Qwen3.6: layer types — 0-indexed. 10 full-attention layers, rest SSM/GDN.
# (best-effort label; adjust if config differs)
ATTN_LAYERS = set(range(3, 40, 4))  # heuristic; refine from config if needed

def load(p):
    b = p.read_bytes()
    return np.frombuffer(b, dtype="<f4").astype(np.float64)

def cmp(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return (float(a @ b / (na*nb + 1e-30)),
            float(np.linalg.norm(a-b)/(nb+1e-30)),
            float(na), float(nb))

def main():
    have = lambda pre: all((OUT/f"{pre}_L{i}.bin").exists() for i in range(N_LAYERS))
    if not have("vllm"):
        print("MISSING vllm_L*.bin — run the vLLM dump first"); sys.exit(1)
    if not have("atlas"):
        print("MISSING atlas_L*.bin — run the Atlas dump (ATLAS_NEMO_DUMP) next"); sys.exit(1)
    print(f"{'layer':>5} {'type':>5} {'cos':>9} {'rel_l2':>9} {'|vllm|':>9} {'|atlas|':>9}")
    print("-"*52)
    onset = None
    rows = []
    for i in range(N_LAYERS):
        a = load(OUT/f"atlas_L{i}.bin"); v = load(OUT/f"vllm_L{i}.bin")
        if a.size != v.size:
            print(f"{i:>5}  SIZE MISMATCH atlas={a.size} vllm={v.size}"); continue
        cos, rel, nv, na = cmp(a, v)
        typ = "attn" if i in ATTN_LAYERS else "ssm"
        rows.append((i, typ, cos, rel, nv, na))
        flag = ""
        if cos < 0.999 and onset is None:
            onset = i; flag = "  <-- divergence onset (cos<0.999)"
        print(f"{i:>5} {typ:>5} {cos:>9.5f} {rel:>9.5f} {nv:>9.1f} {na:>9.1f}{flag}")
    print("-"*52)
    worst = min(rows, key=lambda r: r[2])
    print(f"worst layer: L{worst[0]} ({worst[1]}) cos={worst[2]:.5f} rel_l2={worst[3]:.4f}")
    print(f"final-layer L39 cos={rows[-1][2]:.5f}")
    if onset is not None:
        print(f"DIVERGENCE ONSET: L{onset} — inspect this layer's ops (attn/SSM/MoE/norm) next")
    else:
        print("No layer below cos 0.999 — Atlas-FP8 matches vLLM-FP8; gap is NON-numerical (sampler/parser/scheduler)")
    # final logits / argmax overlap
    if (OUT/"vllm_logits.bin").exists() and (OUT/"atlas_logits.bin").exists():
        vl = load(OUT/"vllm_logits.bin"); al = load(OUT/"atlas_logits.bin")
        if vl.size == al.size:
            cos,rel,_,_ = cmp(al, vl)
            k=20
            vtop=set(np.argsort(vl)[-k:]); atop=set(np.argsort(al)[-k:])
            print(f"\nfinal logits: cos={cos:.5f} rel_l2={rel:.4f} top1_match={np.argmax(vl)==np.argmax(al)} top{k}_overlap={len(vtop&atop)}/{k}")

if __name__ == "__main__":
    main()
