#!/usr/bin/env python3
"""Three-way cosine analysis adapted for the dgx2 study.

A: HF[FP8->BF16] vs HF[BF16-unquant]   -> FP8 ceiling
B: Atlas[FP8-native] vs HF[BF16-unquant] -> total drift
C: Atlas[FP8-native] vs HF[FP8->BF16]    -> Atlas compute drift

Inputs (all in /workspace/atlas-dumps/fp8native_dgx2/):
  atlas_L{0..39}.bin     - Atlas FP8-native dump (today, 2026-05-25)
  hf_L{0..39}.bin        - HF[FP8->BF16] reference (hf_dequant_forward.py output)
  hf_bf16_L{0..39}.bin   - HF[BF16-unquant] reference (hf_forward_bf16_unquant.py output)

If hf_bf16_*.bin is missing, runs only the C comparison (Atlas vs HF[FP8->BF16]).
"""
from __future__ import annotations

import pathlib

import numpy as np

OUT = pathlib.Path("/workspace/atlas-dumps/fp8native_dgx2")
N_LAYERS = 40


def load(p):
    return np.frombuffer(p.read_bytes(), dtype="<f4")


def cmp(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return {
        "cos": float(a @ b / (na * nb + 1e-30)),
        "rel_l2": float(np.linalg.norm(a - b) / (nb + 1e-30)),
    }


def main():
    have_bf16 = all((OUT / f"hf_bf16_L{i}.bin").exists() for i in range(N_LAYERS))
    have_fp8dq = all((OUT / f"hf_L{i}.bin").exists() for i in range(N_LAYERS))
    have_atlas = all((OUT / f"atlas_L{i}.bin").exists() for i in range(N_LAYERS))
    print(f"have_atlas={have_atlas} have_hf_fp8dq={have_fp8dq} have_hf_bf16={have_bf16}")

    rows = []
    A, B, C = [], [], []
    for i in range(N_LAYERS):
        try:
            atlas = load(OUT / f"atlas_L{i}.bin")
        except Exception:
            atlas = None
        try:
            hf_fp8 = load(OUT / f"hf_L{i}.bin")
        except Exception:
            hf_fp8 = None
        try:
            hf_bf16 = load(OUT / f"hf_bf16_L{i}.bin")
        except Exception:
            hf_bf16 = None
        row = {"i": i}
        if hf_fp8 is not None and hf_bf16 is not None:
            row["A"] = cmp(hf_fp8, hf_bf16)
            A.append(row["A"]["cos"])
        if atlas is not None and hf_bf16 is not None:
            row["B"] = cmp(atlas, hf_bf16)
            B.append(row["B"]["cos"])
        if atlas is not None and hf_fp8 is not None:
            row["C"] = cmp(atlas, hf_fp8)
            C.append(row["C"]["cos"])
        rows.append(row)

    print()
    hdr = f"{'L':>3}  "
    if A:
        hdr += f"{'A_cos':>8} {'A_relL2':>8}  "
    if B:
        hdr += f"{'B_cos':>8} {'B_relL2':>8}  "
    if C:
        hdr += f"{'C_cos':>8} {'C_relL2':>8}"
    print(hdr)

    for r in rows:
        line = f"L{r['i']:>2d}  "
        if "A" in r:
            line += f"{r['A']['cos']:>8.5f} {r['A']['rel_l2']:>8.4f}  "
        if "B" in r:
            line += f"{r['B']['cos']:>8.5f} {r['B']['rel_l2']:>8.4f}  "
        if "C" in r:
            flag = " *" if r["C"]["cos"] < 0.99 else ""
            line += f"{r['C']['cos']:>8.5f} {r['C']['rel_l2']:>8.4f}{flag}"
        print(line)

    def stats(name, xs):
        if not xs:
            print(f"  {name}: (no data)")
            return None
        arr = np.array(xs)
        imin = int(np.argmin(arr))
        print(
            f"  {name}: mean={arr.mean():.6f}  min={arr.min():.6f} (L{imin})  max={arr.max():.6f}  n={len(arr)}"
        )
        return {"mean": float(arr.mean()), "min": float(arr.min()), "min_layer": imin}

    print(f"\n=== Summary ===")
    sA = stats("A (HF[FP8->BF16] vs HF[BF16-unquant])", A)
    sB = stats("B (Atlas[FP8-native] vs HF[BF16-unquant])", B)
    sC = stats("C (Atlas[FP8-native] vs HF[FP8->BF16])", C)

    if sA and sC:
        delta = sA["mean"] - sC["mean"]
        print(f"\nHeadroom (A_mean - C_mean) = {delta:+.6f}")
        if delta < 0.001:
            verdict = "Atlas FP8-native is AT the FP8 ceiling — drift NOT in SSM dispatch."
        elif delta < 0.01:
            verdict = f"Atlas has minor compute headroom of {delta:.4f} below the FP8 ceiling."
        else:
            verdict = f"Atlas has substantial compute headroom ({delta:.4f}) — kernel-level drift remains."
        print("Verdict:", verdict)


if __name__ == "__main__":
    main()
