#!/usr/bin/env python3
"""Compare per-layer cosine across multiple phase2c configs.

Usage:
    python3 compare-configs.py [config_name1] [config_name2] ...

If no args, compares all phase2c-* dirs against HF reference.
With args, focuses the comparison on those configs and shows per-layer deltas.
"""
from __future__ import annotations
import pathlib
import sys
import numpy as np

ND = pathlib.Path("/workspace/atlas-dumps/numdrift")


def load(p: pathlib.Path) -> np.ndarray:
    return np.frombuffer(p.read_bytes(), dtype="<f4")


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def per_layer(config_dir: pathlib.Path) -> list[float]:
    hf_dir = ND
    out: list[float] = []
    for i in range(40):
        ap = config_dir / f"atlas_L{i}.bin"
        hp = hf_dir / f"hf_L{i}.bin"
        if ap.exists() and hp.exists():
            a = load(ap)
            h = load(hp)
            if a.size != h.size:
                out.append(float("nan"))
                continue
            if np.isnan(a).any():
                out.append(float("nan"))
                continue
            out.append(cos(a, h))
        else:
            out.append(float("nan"))
    return out


def fmt(x: float) -> str:
    if x != x:  # NaN
        return "  nan "
    return f"{x:.4f}"


def main() -> None:
    if len(sys.argv) > 1:
        names = sys.argv[1:]
        dirs = [ND / f"phase2c-{n}" if not (ND / n).exists() else ND / n for n in names]
    else:
        dirs = sorted(p for p in ND.iterdir() if p.is_dir() and p.name.startswith("phase2c-"))

    # Load each
    data: dict[str, list[float]] = {}
    for d in dirs:
        if not d.exists():
            print(f"# WARN: {d} missing", file=sys.stderr)
            continue
        data[d.name] = per_layer(d)

    # Header
    headers = list(data.keys())
    print(f"{'layer':<8} " + " ".join(f"{h[:13]:>13}" for h in headers))
    print("-" * (8 + 14 * len(headers)))

    # Per-layer
    for i in range(40):
        row = f"L{i:02} ".ljust(8)
        for h in headers:
            row += f"{fmt(data[h][i]):>13} "
        print(row)

    # Stats
    print()
    print(f"{'mean':<8} " + " ".join(f"{fmt(float(np.nanmean(data[h]))):>13}" for h in headers))
    print(f"{'min':<8} " + " ".join(f"{fmt(float(np.nanmin(data[h]))):>13}" for h in headers))

    # Delta from first config (if multi)
    if len(headers) >= 2:
        base = data[headers[0]]
        print()
        print("Per-layer delta vs " + headers[0] + ":")
        for h in headers[1:]:
            deltas = [data[h][i] - base[i] for i in range(40)]
            mean_d = float(np.nanmean(deltas))
            max_d = max(deltas, key=lambda x: abs(x) if x == x else 0)
            print(f"  {h:25}  mean_delta={mean_d:+.4f}  max_abs_delta={max_d:+.4f}")


if __name__ == "__main__":
    main()
