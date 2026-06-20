#!/usr/bin/env python3
"""Compare two tiers via per-metric Mann-Whitney U test.

Reads per-run scores for two tiers from
`bench/fp8_dgx2_drift/harness/runs/run_<tier>_*.json` and produces a
ranked table of which metrics differ significantly.

Stats:
  - Mann-Whitney U (one-sided, B vs A) for each metric
  - Effect size: median(B) - median(A)
  - Bonferroni-adjusted p-values (n_metrics tests)

Usage:
    python3 compare_tiers.py --base TIER_A --candidate TIER_B [--alpha 0.05]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from typing import Any

from aggregate import METRICS, jpath

try:
    from scipy.stats import mannwhitneyu  # type: ignore
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def mannwhitney_u_self(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U with normal approximation, when scipy
    isn't available. Returns (U_stat, p_value).
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return (0.0, 1.0)
    combined = [(v, "a") for v in a] + [(v, "b") for v in b]
    combined.sort(key=lambda x: x[0])
    # Average-rank assignment for ties
    ranks: dict[int, float] = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-indexed ranks
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    R_a = sum(ranks[k] for k, (_, lab) in enumerate(combined) if lab == "a")
    U_a = R_a - n_a * (n_a + 1) / 2.0
    U_b = n_a * n_b - U_a
    U = min(U_a, U_b)
    # Normal approximation
    mu = n_a * n_b / 2.0
    sigma = (n_a * n_b * (n_a + n_b + 1) / 12.0) ** 0.5
    if sigma == 0:
        return (U, 1.0)
    z = (U - mu) / sigma
    # Two-sided p
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2.0))
    return (U, p)


def load_runs(runs_dir: pathlib.Path, tier: str) -> list[dict[str, Any]]:
    """Strict tier match — file name must be exactly `run_<tier>_<N>.json`
    with `<N>` a positive integer. Prevents glob collisions between
    prefix-related tier names (e.g. `sm1` vs `sm1_a2ao`)."""
    out = []
    prefix = f"run_{tier}_"
    for f in sorted(runs_dir.iterdir()):
        if not f.is_file() or not f.name.startswith(prefix) or not f.name.endswith(".json"):
            continue
        suffix = f.name[len(prefix):-len(".json")]
        if not suffix.isdigit():
            continue
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out


def values_for(runs: list[dict[str, Any]], metric_path: str) -> list[float]:
    out: list[float] = []
    for r in runs:
        v = jpath(r, metric_path)
        if v is None:
            continue
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        try:
            out.append(float(v))
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    here = pathlib.Path(__file__).resolve().parent
    ap.add_argument("--runs-dir", type=pathlib.Path, default=here / "runs")
    ap.add_argument("--reports-dir", type=pathlib.Path, default=here / "reports")
    ap.add_argument("--base", required=True, help="Tier A (baseline)")
    ap.add_argument("--candidate", required=True, help="Tier B (proposed)")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    A = load_runs(args.runs_dir, args.base)
    B = load_runs(args.runs_dir, args.candidate)
    if not A or not B:
        print(f"ERROR: missing runs (A={len(A)}, B={len(B)})", file=sys.stderr)
        return 1
    print(f"Comparing {args.base} (n={len(A)}) vs {args.candidate} (n={len(B)})")
    print(f"Using scipy: {HAVE_SCIPY}")
    print()

    rows: list[tuple[str, float, float, float, float, float]] = []
    n_tests = len(METRICS)
    for path, name, _kind in METRICS:
        a_vals = values_for(A, path)
        b_vals = values_for(B, path)
        if not a_vals or not b_vals:
            rows.append((name, 0.0, 0.0, 0.0, 1.0, 1.0))
            continue
        median_a = statistics.median(a_vals) if a_vals else 0.0
        median_b = statistics.median(b_vals) if b_vals else 0.0
        delta = median_b - median_a
        if HAVE_SCIPY:
            try:
                _, p = mannwhitneyu(b_vals, a_vals, alternative="two-sided")
            except Exception:
                _, p = mannwhitney_u_self(a_vals, b_vals)
        else:
            _, p = mannwhitney_u_self(a_vals, b_vals)
        p_bonf = min(1.0, p * n_tests)
        rows.append((name, median_a, median_b, delta, p, p_bonf))

    rows.sort(key=lambda r: r[5])  # by adjusted p

    print(f"{'metric':<38} {'med(A)':>10} {'med(B)':>10} {'delta':>10} {'p':>8} {'p_bonf':>8}")
    print("-" * 90)
    for name, ma, mb, delta, p, pb in rows:
        sig = " *" if pb < args.alpha else ""
        print(f"{name:<38} {ma:>10.3f} {mb:>10.3f} {delta:>+10.3f} {p:>8.4f} {pb:>8.4f}{sig}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report = args.reports_dir / f"compare_{args.base}_vs_{args.candidate}.md"
    lines = [
        f"# Tier comparison — `{args.base}` (n={len(A)}) vs `{args.candidate}` (n={len(B)})",
        "",
        f"alpha={args.alpha} (Bonferroni-adjusted across {n_tests} metrics).",
        "Significant rows are starred (*).",
        "",
        f"| metric | median({args.base}) | median({args.candidate}) | Δ | p | p_bonf | sig |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, ma, mb, delta, p, pb in rows:
        sig = "**\\***" if pb < args.alpha else ""
        lines.append(
            f"| {name} | {ma:.3f} | {mb:.3f} | {delta:+.3f} | {p:.4f} | {pb:.4f} | {sig} |"
        )
    report.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
