#!/usr/bin/env python3
"""Per-layer cosine analysis for the dgx2 FP8-native SSM drift study.

Compares the freshly-dumped Atlas[FP8-native] hidden states on dgx2 against
the existing HF[FP8->BF16] reference and the (older) HF[BF16-unquant]
reference from /workspace/atlas-dumps/numdrift.

Layouts:
  ATLAS_DIR  /workspace/atlas-dumps/fp8native_dgx2/atlas_L{0..39}.bin
  HF_FP8_DIR /workspace/atlas-dumps/fp8dequant/hf_L{0..39}.bin       (HF[FP8->BF16])
  HF_BF16_DIR /workspace/atlas-dumps/numdrift/hf_L{0..39}.bin        (HF[BF16-unquant])

Reports:
  A  HF[FP8->BF16] vs HF[BF16-unquant]   -> FP8 ceiling
  B  Atlas[FP8-native] vs HF[BF16-unquant] -> total drift
  C  Atlas[FP8-native] vs HF[FP8->BF16]    -> Atlas-side compute error
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ATLAS = pathlib.Path("/workspace/atlas-dumps/fp8native_dgx2")
HF_FP8 = pathlib.Path("/workspace/atlas-dumps/fp8dequant")
HF_BF16 = pathlib.Path("/workspace/atlas-dumps/numdrift")
N_LAYERS = 40


def load(p: pathlib.Path) -> np.ndarray:
    return np.frombuffer(p.read_bytes(), dtype="<f4")


def cmp(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "cos": float(a @ b / (na * nb + 1e-30)),
        "rel_l2": float(np.linalg.norm(a - b) / (nb + 1e-30)),
    }


def main(write_md: pathlib.Path | None = None) -> None:
    rows = []
    A, B, C = [], [], []
    for i in range(N_LAYERS):
        atlas_p = ATLAS / f"atlas_L{i}.bin"
        hf_fp8_p = HF_FP8 / f"hf_L{i}.bin"
        hf_bf16_p = HF_BF16 / f"hf_L{i}.bin"
        missing = []
        if not atlas_p.exists():
            missing.append("atlas")
        if not hf_fp8_p.exists():
            missing.append("hf_fp8")
        if not hf_bf16_p.exists():
            missing.append("hf_bf16")
        if missing:
            print(f"L{i:2d}: MISSING {missing}")
            continue
        atlas = load(atlas_p)
        hf_fp8 = load(hf_fp8_p)
        hf_bf16 = load(hf_bf16_p)
        rA = cmp(hf_fp8, hf_bf16)
        rB = cmp(atlas, hf_bf16)
        rC = cmp(atlas, hf_fp8)
        A.append(rA["cos"])
        B.append(rB["cos"])
        C.append(rC["cos"])
        rows.append({"i": i, "A": rA, "B": rB, "C": rC})

    print(
        f"{'L':>3}  {'A_cos':>8} {'A_relL2':>7}  "
        f"{'B_cos':>8} {'B_relL2':>7}  "
        f"{'C_cos':>8} {'C_relL2':>7}"
    )
    for r in rows:
        flag = " *" if r["C"]["cos"] < 0.99 else "  "
        print(
            f"L{r['i']:>2d}  "
            f"{r['A']['cos']:>8.5f} {r['A']['rel_l2']:>7.4f}  "
            f"{r['B']['cos']:>8.5f} {r['B']['rel_l2']:>7.4f}  "
            f"{r['C']['cos']:>8.5f} {r['C']['rel_l2']:>7.4f}{flag}"
        )

    def stats(lbl, xs):
        if not xs:
            print(f"  {lbl}: (no data)")
            return None
        arr = np.array(xs)
        idx_min = int(np.argmin(arr))
        return {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "min_layer": idx_min,
            "n": len(xs),
        }

    sA, sB, sC = stats("A", A), stats("B", B), stats("C", C)
    print(f"\n=== Summary (n={sA['n'] if sA else 0} layers) ===")
    for lbl, s in [
        ("A (HF[FP8->BF16] vs HF[BF16])", sA),
        ("B (Atlas[FP8-native] vs HF[BF16])", sB),
        ("C (Atlas[FP8-native] vs HF[FP8->BF16])", sC),
    ]:
        if s:
            print(
                f"  {lbl}: mean={s['mean']:.6f} min={s['min']:.6f} (L{s['min_layer']}) max={s['max']:.6f}"
            )

    if sA and sC:
        headroom = sA["mean"] - sC["mean"]
        print(f"\nHeadroom (A_mean - C_mean) = {headroom:+.6f}")
        if abs(headroom) < 0.001:
            verdict = "AT the FP8 ceiling — drift is NOT in Atlas SSM dispatch."
        elif headroom > 0:
            verdict = f"Atlas compute drift of {headroom:.4f} cos remains below ceiling — investigate kernels."
        else:
            verdict = f"Atlas BEATS the ceiling by {-headroom:.4f} cos (unexpected; check refs)."
        print("Verdict:", verdict)

    if write_md and sA:
        with write_md.open("w") as f:
            f.write("# dgx2 FP8-native SSM per-layer drift — three-way comparison\n\n")
            f.write(f"- A mean cos (FP8 ceiling) = `{sA['mean']:.6f}` (min L{sA['min_layer']}: {sA['min']:.6f})\n")
            f.write(f"- B mean cos (Atlas vs HF[BF16]) = `{sB['mean']:.6f}` (min L{sB['min_layer']}: {sB['min']:.6f})\n")
            f.write(f"- C mean cos (Atlas vs HF[FP8->BF16]) = `{sC['mean']:.6f}` (min L{sC['min_layer']}: {sC['min']:.6f})\n")
            f.write(f"- Headroom A−C = `{sA['mean'] - sC['mean']:+.6f}`\n\n")
            f.write("## Per-layer table\n\n")
            f.write("| L | A.cos | A.relL2 | B.cos | B.relL2 | C.cos | C.relL2 |\n")
            f.write("|--:|------:|--------:|------:|--------:|------:|--------:|\n")
            for r in rows:
                f.write(
                    f"| L{r['i']:2d} | "
                    f"{r['A']['cos']:.5f} | {r['A']['rel_l2']:.4f} | "
                    f"{r['B']['cos']:.5f} | {r['B']['rel_l2']:.4f} | "
                    f"{r['C']['cos']:.5f} | {r['C']['rel_l2']:.4f} |\n"
                )


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(out)
