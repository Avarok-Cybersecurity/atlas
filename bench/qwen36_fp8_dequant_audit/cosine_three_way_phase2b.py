#!/usr/bin/env python3
"""Phase 2b cosine compare: post-RNE Atlas image vs both references.

Reads:
  - /workspace/atlas-dumps/numdrift/rne/atlas_L{0..39}.bin    (NEW post-RNE)
  - /workspace/atlas-dumps/numdrift/atlas_L{0..39}.bin        (OLD truncating)
  - /workspace/atlas-dumps/numdrift/hf_L{0..39}.bin           (unquant BF16 ref)
  - /workspace/atlas-dumps/fp8dequant/hf_L{0..39}.bin         (FP8->BF16 ref)

Reports four series:
  B_old: Atlas[truncating] vs HF[unquant]      -- Phase α baseline
  B_new: Atlas[RNE]        vs HF[unquant]      -- Phase 2b result
  C_old: Atlas[truncating] vs HF[FP8->BF16]    -- Phase 2a Atlas-fidelity baseline
  C_new: Atlas[RNE]        vs HF[FP8->BF16]    -- Phase 2b Atlas-fidelity result

Plus the ceiling (A) from Phase 2a for reference.

Target: C_new mean >= 0.997. B_new mean approaches A mean (0.989).
"""
from __future__ import annotations

import pathlib

import numpy as np

NUMDRIFT = pathlib.Path("/workspace/atlas-dumps/numdrift")
DEQUANT = pathlib.Path("/workspace/atlas-dumps/fp8dequant")
RNE = pathlib.Path("/workspace/atlas-dumps/numdrift/rne")
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
    cosines_B_old: list[float] = []
    cosines_B_new: list[float] = []
    cosines_C_old: list[float] = []
    cosines_C_new: list[float] = []

    for i in range(N_LAYERS):
        atlas_old_p = NUMDRIFT / f"atlas_L{i}.bin"
        atlas_new_p = RNE / f"atlas_L{i}.bin"
        hf_unquant_p = NUMDRIFT / f"hf_L{i}.bin"
        hf_fp8dq_p = DEQUANT / f"hf_L{i}.bin"
        missing: list[str] = []
        for p, label in [
            (atlas_old_p, "atlas_old"),
            (atlas_new_p, "atlas_new"),
            (hf_unquant_p, "hf_unquant"),
            (hf_fp8dq_p, "hf_fp8dq"),
        ]:
            if not p.exists():
                missing.append(label)
        if missing:
            print(f"L{i:2d}: MISSING {missing}")
            continue
        atlas_old = load(atlas_old_p)
        atlas_new = load(atlas_new_p)
        hf_unquant = load(hf_unquant_p)
        hf_fp8dq = load(hf_fp8dq_p)
        rA = cmp_pair(hf_fp8dq, hf_unquant)
        rB_old = cmp_pair(atlas_old, hf_unquant)
        rB_new = cmp_pair(atlas_new, hf_unquant)
        rC_old = cmp_pair(atlas_old, hf_fp8dq)
        rC_new = cmp_pair(atlas_new, hf_fp8dq)
        cosines_A.append(rA["cos"])
        cosines_B_old.append(rB_old["cos"])
        cosines_B_new.append(rB_new["cos"])
        cosines_C_old.append(rC_old["cos"])
        cosines_C_new.append(rC_new["cos"])
        rows.append({"layer": i, "A": rA, "B_old": rB_old, "B_new": rB_new, "C_old": rC_old, "C_new": rC_new})

    print()
    print(
        f"{'L':>3}  "
        f"{'A':>8}  "
        f"{'B_old':>8} {'B_new':>8} {'dB':>7}  "
        f"{'C_old':>8} {'C_new':>8} {'dC':>7}"
    )
    for r in rows:
        dB = r["B_new"]["cos"] - r["B_old"]["cos"]
        dC = r["C_new"]["cos"] - r["C_old"]["cos"]
        print(
            f"L{r['layer']:>2d}  "
            f"{r['A']['cos']:>8.5f}  "
            f"{r['B_old']['cos']:>8.5f} {r['B_new']['cos']:>8.5f} {dB:>+7.4f}  "
            f"{r['C_old']['cos']:>8.5f} {r['C_new']['cos']:>8.5f} {dC:>+7.4f}"
        )

    sA = summarize("A: HF[FP8->BF16] vs HF[unquant] (ceiling)", cosines_A)
    sB_old = summarize("B_old: Atlas[trunc] vs HF[unquant]       ", cosines_B_old)
    sB_new = summarize("B_new: Atlas[RNE]   vs HF[unquant]       ", cosines_B_new)
    sC_old = summarize("C_old: Atlas[trunc] vs HF[FP8->BF16]      ", cosines_C_old)
    sC_new = summarize("C_new: Atlas[RNE]   vs HF[FP8->BF16]      ", cosines_C_new)

    print()
    print(f"=== Summary (n={sA['n']} layers) ===")
    for s in (sA, sB_old, sB_new, sC_old, sC_new):
        print(f"  {s['name']}:  mean={s['mean']:.6f}  min={s['min']:.6f}  max={s['max']:.6f}")

    print()
    print("=== Phase 2b verdict ===")
    print(f"  Ceiling (A):        mean={sA['mean']:.5f}")
    print(f"  C improvement:      {sC_old['mean']:.5f} -> {sC_new['mean']:.5f}  (+{sC_new['mean']-sC_old['mean']:.5f})")
    print(f"  B improvement:      {sB_old['mean']:.5f} -> {sB_new['mean']:.5f}  (+{sB_new['mean']-sB_old['mean']:.5f})")
    print(f"  Gap to ceiling now: A - C_new = {sA['mean']-sC_new['mean']:+.5f}")
    print()
    if sC_new["mean"] >= 0.997:
        print("RESULT: TARGET MET. C >= 0.997, Atlas compute path at ceiling.")
    elif sC_new["mean"] - sC_old["mean"] >= 0.01:
        print("RESULT: SIGNIFICANT IMPROVEMENT. Some headroom remains; investigate other compute paths.")
    elif sC_new["mean"] - sC_old["mean"] >= 0.001:
        print("RESULT: MARGINAL IMPROVEMENT. RNE alone is not sufficient; need MMA / kernel deep-dive.")
    else:
        print("RESULT: NO IMPROVEMENT. The truncation-vs-RNE hypothesis was wrong; need new lead.")


if __name__ == "__main__":
    main()
