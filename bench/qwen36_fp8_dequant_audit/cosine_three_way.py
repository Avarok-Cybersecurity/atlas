#!/usr/bin/env python3
"""Phase 2a three-way cosine comparison.

Decomposes the 0.967 Atlas-vs-HF[BF16-unquant] gap into:

  A. HF[FP8->BF16] vs HF[BF16-unquant]  -> inherent FP8 quant loss (the CEILING)
  B. Atlas[FP8]    vs HF[BF16-unquant]  -> total drift (already known, ~0.967)
  C. Atlas[FP8]    vs HF[FP8->BF16]     -> Atlas compute-side fidelity

The exit gate:

  If A mean ~= 0.99x and matches B closely => Atlas is at the ceiling, work the
       symptoms downstream (Phase 2d).
  If A mean >> B (e.g. A=0.998, B=0.967) => Atlas has Compute headroom; close
       it via Phase 2b BF16 rounding patch.

Reuses /tmp/cosine_compare.py's cmp_pair() math. Writes a final markdown
verdict under /workspace/atlas-dumps/fp8dequant/PHASE2A_VERDICT.md.
"""
from __future__ import annotations

import pathlib

import numpy as np

NUMDRIFT = pathlib.Path("/workspace/atlas-dumps/numdrift")
DEQUANT = pathlib.Path("/workspace/atlas-dumps/fp8dequant")
N_LAYERS = 40


def load(p: pathlib.Path) -> np.ndarray:
    return np.frombuffer(p.read_bytes(), dtype="<f4")


def cmp_pair(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb + 1e-30))
    return {
        "cos": cos,
        "max_abs": float(np.max(np.abs(a - b))),
        "rel_l2": float(np.linalg.norm(a - b) / (nb + 1e-30)),
    }


def summarize(name: str, cosines: list[float]) -> dict:
    if not cosines:
        return {"name": name, "mean": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "name": name,
        "mean": float(np.mean(cosines)),
        "min": float(np.min(cosines)),
        "max": float(np.max(cosines)),
        "min_layer": int(np.argmin(cosines)),
        "n": len(cosines),
    }


def main() -> None:
    rows: list[dict] = []
    cosines_A: list[float] = []
    cosines_B: list[float] = []
    cosines_C: list[float] = []

    for i in range(N_LAYERS):
        atlas_p = NUMDRIFT / f"atlas_L{i}.bin"
        hf_unquant_p = NUMDRIFT / f"hf_L{i}.bin"
        hf_fp8dq_p = DEQUANT / f"hf_L{i}.bin"
        missing: list[str] = []
        if not atlas_p.exists():
            missing.append("atlas")
        if not hf_unquant_p.exists():
            missing.append("hf[unquant]")
        if not hf_fp8dq_p.exists():
            missing.append("hf[FP8->BF16]")
        if missing:
            print(f"L{i:2d}: MISSING {missing}")
            continue
        atlas = load(atlas_p)
        hf_unquant = load(hf_unquant_p)
        hf_fp8dq = load(hf_fp8dq_p)
        rA = cmp_pair(hf_fp8dq, hf_unquant)
        rB = cmp_pair(atlas, hf_unquant)
        rC = cmp_pair(atlas, hf_fp8dq)
        cosines_A.append(rA["cos"])
        cosines_B.append(rB["cos"])
        cosines_C.append(rC["cos"])
        rows.append({"layer": i, "A": rA, "B": rB, "C": rC})

    print()
    print(
        f"{'L':>3}  "
        f"{'A_cos':>8} {'A_relL2':>8}  "
        f"{'B_cos':>8} {'B_relL2':>8}  "
        f"{'C_cos':>8} {'C_relL2':>8}"
    )
    for r in rows:
        marker_C = " *" if r["C"]["cos"] < 0.998 else "  "
        print(
            f"L{r['layer']:>2d}  "
            f"{r['A']['cos']:>8.5f} {r['A']['rel_l2']:>8.4f}  "
            f"{r['B']['cos']:>8.5f} {r['B']['rel_l2']:>8.4f}  "
            f"{r['C']['cos']:>8.5f} {r['C']['rel_l2']:>8.4f}{marker_C}"
        )

    sA = summarize("A: HF[FP8->BF16] vs HF[unquant]", cosines_A)
    sB = summarize("B: Atlas vs HF[unquant]        ", cosines_B)
    sC = summarize("C: Atlas vs HF[FP8->BF16]      ", cosines_C)
    print()
    print(f"=== Summary (n={sA['n']} layers) ===")
    for s in (sA, sB, sC):
        print(
            f"  {s['name']}:  mean={s['mean']:.6f}  min={s['min']:.6f}  max={s['max']:.6f}"
        )

    # Verdict
    print()
    ceiling = sA["mean"]
    actual = sC["mean"]
    delta = ceiling - actual
    print(
        f"=== Verdict (ceiling=A mean={ceiling:.5f}, "
        f"Atlas compute=C mean={actual:.5f}, headroom={delta:+.5f}) ==="
    )
    if delta < 0.001:
        verdict = (
            "Atlas is AT the inherent FP8 ceiling. No fixable compute drift; "
            "proceed to Phase 2d (sampling-side symptom hardening)."
        )
    elif delta < 0.01:
        verdict = (
            "Atlas is close to the ceiling but has minor headroom. Phase 2b "
            "rounding patch worth trying; expect modest gain."
        )
    else:
        verdict = (
            f"Atlas has substantial compute headroom ({delta:.4f} cos). "
            "Phase 2b BF16 round-to-nearest-even patch is the right next step; "
            "expect Atlas-vs-ceiling gap to close significantly."
        )
    print(verdict)

    # Write verdict file
    md_path = DEQUANT / "PHASE2A_VERDICT.md"
    with md_path.open("w") as f:
        f.write("# Phase 2a Verdict — FP8 Quantization-Loss Ceiling\n\n")
        f.write(f"- **A** mean cosine `HF[FP8->BF16]  vs HF[unquant]` = `{sA['mean']:.6f}` (min L{sA.get('min_layer')}: {sA['min']:.6f})\n")
        f.write(f"- **B** mean cosine `Atlas[FP8]    vs HF[unquant]`  = `{sB['mean']:.6f}` (min L{sB.get('min_layer')}: {sB['min']:.6f})\n")
        f.write(f"- **C** mean cosine `Atlas[FP8]    vs HF[FP8->BF16]` = `{sC['mean']:.6f}` (min L{sC.get('min_layer')}: {sC['min']:.6f})\n")
        f.write(f"- **Headroom** (A − C) = `{delta:+.6f}`\n\n")
        f.write("## Verdict\n\n")
        f.write(verdict + "\n\n")
        f.write("## Per-layer table\n\n")
        f.write("| Layer | A.cos | A.relL2 | B.cos | B.relL2 | C.cos | C.relL2 |\n")
        f.write("|------:|------:|--------:|------:|--------:|------:|--------:|\n")
        for r in rows:
            f.write(
                f"| L{r['layer']:2d} | "
                f"{r['A']['cos']:.5f} | {r['A']['rel_l2']:.4f} | "
                f"{r['B']['cos']:.5f} | {r['B']['rel_l2']:.4f} | "
                f"{r['C']['cos']:.5f} | {r['C']['rel_l2']:.4f} |\n"
            )
    print(f"\nVerdict written to {md_path}")


if __name__ == "__main__":
    main()
