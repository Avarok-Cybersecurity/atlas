#!/usr/bin/env python3
"""Aggregate per-run scores into per-tier summaries.

Reads bench/fp8_dgx2_drift/harness/runs/run_<tier>_*.json and writes
per-tier statistics to bench/fp8_dgx2_drift/harness/reports/<tier>.{md,csv}.

Stats per metric:
  - mean, std
  - p50, p90
  - bootstrap 95% CI on the mean (10k resamples)
  - count of non-zero observations (for rare-event metrics)

Usage:
    python3 aggregate.py [--runs-dir PATH] [--reports-dir PATH] [--tier TIER]

If --tier is omitted, aggregates every tier found in runs-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
import statistics
import sys
from collections import defaultdict
from typing import Any

# Metrics to aggregate. Tuple of (json-path, friendly-name, "rate" or "count").
# Rate = boolean-ish (was it 0/1?); count = integer count of events.
METRICS = [
    ("filesystem.files_count", "files_written", "count"),
    ("cargo.cargo_toml_valid", "cargo_toml_valid", "rate"),
    ("cargo.cargo_toml_present", "cargo_toml_present", "rate"),
    ("webserver.webserver_ok", "webserver_ok", "rate"),
    ("followed_directions.followed_directions", "followed_directions", "rate"),
    ("followed_directions.steps_completed", "fd_steps_completed", "count"),
    ("tool_calls.total", "tool_calls_total", "count"),
    ("drift.write_calls", "write_calls", "count"),
    ("drift.write_empty_path", "drift_empty_path", "count"),
    ("drift.write_path_drift_from_target", "drift_path_outside_target", "count"),
    ("drift.write_path_has_literal_space", "drift_path_literal_space", "count"),
    ("drift.write_content_starts_with_lean", "drift_lean_prefix", "count"),
    ("drift.write_content_is_bash_command", "drift_bash_as_content", "count"),
    ("drift.write_content_xml_attr_leak", "drift_xml_attr_leak", "count"),
    ("drift.write_content_newlines_collapsed_toml", "drift_toml_newlines_collapsed", "count"),
    ("atlas.ws1_mask_active_fires", "atlas_ws1_mask_fires", "count"),
    ("atlas.b1_drift_gauge_fires", "atlas_b1_drift_fires", "count"),
    ("atlas.tier_5c_retries", "atlas_tier5c_retries", "count"),
    ("atlas.a2_fuzzy_repair_fires", "atlas_a2_fuzzy_fires", "count"),
    ("atlas.tool_call_lines", "atlas_tool_call_lines", "count"),
    ("wall_time_s", "wall_time_s", "count"),
]


def jpath(d: dict[str, Any], path: str) -> Any:
    """Read 'a.b.c' from nested dict. Returns None on missing."""
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def bootstrap_ci(values: list[float], n_boot: int = 10000, ci: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap 95% CI on the mean. Returns (lo, hi)."""
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rnd = random.Random(0xA71A5)  # ~"ATLAS" — deterministic, doesn't matter
    means: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = int(n_boot * (1 - ci) / 2)
    hi_idx = int(n_boot * (1 - (1 - ci) / 2))
    return (means[lo_idx], means[hi_idx])


def summarize(values: list[float], kind: str) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    n = len(values)
    mean = sum(values) / n
    std = statistics.pstdev(values) if n > 1 else 0.0
    sorted_v = sorted(values)
    p50 = sorted_v[n // 2]
    p90 = sorted_v[min(n - 1, int(0.9 * n))]
    ci_lo, ci_hi = bootstrap_ci(values)
    nonzero = sum(1 for v in values if v != 0)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "p50": p50,
        "p90": p90,
        "ci_lo_95": ci_lo,
        "ci_hi_95": ci_hi,
        "nonzero_runs": nonzero,
        "kind": kind,
    }


def aggregate_tier(runs_dir: pathlib.Path, tier: str) -> dict[str, Any]:
    """Strict tier match — file name must be exactly `run_<tier>_<N>.json`
    with `<N>` a positive integer. Prevents glob collisions between
    prefix-related tier names (e.g. `sm1` vs `sm1_a2ao`)."""
    runs: list[dict[str, Any]] = []
    prefix = f"run_{tier}_"
    for f in sorted(runs_dir.iterdir()):
        if not f.is_file() or not f.name.startswith(prefix) or not f.name.endswith(".json"):
            continue
        suffix = f.name[len(prefix):-len(".json")]
        if not suffix.isdigit():
            continue
        try:
            runs.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"WARN: failed to load {f}: {e}", file=sys.stderr)
    if not runs:
        return {"tier": tier, "n": 0, "metrics": {}}
    metrics: dict[str, Any] = {}
    for path, name, kind in METRICS:
        values: list[float] = []
        for r in runs:
            v = jpath(r, path)
            if v is None:
                continue
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            try:
                values.append(float(v))
            except Exception:
                continue
        metrics[name] = summarize(values, kind)
    # Failure counts for the process exit code: a run fails cargo if
    # cargo.cargo_toml_valid is not truthy, and fails webserver if
    # webserver.webserver_ok is not truthy (missing/None counts as a failure).
    cargo_failures = sum(1 for r in runs if not jpath(r, "cargo.cargo_toml_valid"))
    webserver_failures = sum(1 for r in runs if not jpath(r, "webserver.webserver_ok"))
    # Process-fidelity (informational — NOT part of the pass/fail exit code).
    # Counts only runs that carry the metric; absent (older, un-backfilled) runs
    # are excluded from both numerator and denominator so the rate isn't diluted.
    fd_runs = [r for r in runs if isinstance(jpath(r, "followed_directions"), dict)
               and jpath(r, "followed_directions.followed_directions") is not None]
    fd_followed = sum(1 for r in fd_runs if jpath(r, "followed_directions.followed_directions"))
    return {
        "tier": tier,
        "n": len(runs),
        "metrics": metrics,
        "runs": [r.get("run") for r in runs],
        "cargo_failures": cargo_failures,
        "webserver_failures": webserver_failures,
        "followed_directions_scored": len(fd_runs),
        "followed_directions_followed": fd_followed,
    }


def write_csv(report: dict[str, Any], out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "kind", "n", "mean", "std", "p50", "p90", "ci_lo_95", "ci_hi_95", "nonzero_runs"])
        for name, m in report["metrics"].items():
            if m.get("n", 0) == 0:
                w.writerow([name, "", 0, "", "", "", "", "", "", ""])
                continue
            w.writerow([
                name, m.get("kind", ""), m["n"],
                f"{m['mean']:.4f}", f"{m['std']:.4f}",
                f"{m['p50']:.4f}", f"{m['p90']:.4f}",
                f"{m['ci_lo_95']:.4f}", f"{m['ci_hi_95']:.4f}",
                m["nonzero_runs"],
            ])


def write_md(report: dict[str, Any], out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Harness aggregate — tier `{report['tier']}` (N={report['n']})",
        "",
        f"Generated from `bench/fp8_dgx2_drift/harness/runs/run_{report['tier']}_*.json`.",
        f"Runs: {report.get('runs', [])}",
        "",
        "| metric | kind | n | mean ± std | p50 | p90 | 95% CI | non-zero runs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, m in report["metrics"].items():
        if m.get("n", 0) == 0:
            lines.append(f"| {name} | | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {name} | {m.get('kind','')} | {m['n']} | "
            f"{m['mean']:.3f} ± {m['std']:.3f} | "
            f"{m['p50']:.3f} | {m['p90']:.3f} | "
            f"[{m['ci_lo_95']:.3f}, {m['ci_hi_95']:.3f}] | "
            f"{m['nonzero_runs']}/{m['n']} |"
        )
    out.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    here = pathlib.Path(__file__).resolve().parent
    ap.add_argument("--runs-dir", type=pathlib.Path, default=here / "runs")
    ap.add_argument("--reports-dir", type=pathlib.Path, default=here / "reports")
    ap.add_argument("--tier", default=None, help="Aggregate only this tier; else all.")
    args = ap.parse_args()

    if args.tier:
        tiers = [args.tier]
    else:
        tiers = sorted({p.stem.split("_")[1] for p in args.runs_dir.glob("run_*_*.json")})

    if not tiers:
        print("no tiers found in runs dir", file=sys.stderr)
        return 1

    total_failures = 0
    for tier in tiers:
        report = aggregate_tier(args.runs_dir, tier)
        write_csv(report, args.reports_dir / f"{tier}.csv")
        write_md(report, args.reports_dir / f"{tier}.md")
        cf = report.get("cargo_failures", 0)
        wf = report.get("webserver_failures", 0)
        total_failures += cf + wf
        fds = report.get("followed_directions_scored", 0)
        fdf = report.get("followed_directions_followed", 0)
        fd_str = f" followed_dirs={fdf}/{fds}" if fds else ""
        print(
            f"  tier={tier} n={report['n']} cargo_fail={cf} webserver_fail={wf}{fd_str} "
            f"-> {args.reports_dir}/{tier}.{{csv,md}}",
            file=sys.stderr,
        )
    # Exit code = total failure count (cargo + webserver), so 0 == all green.
    # Clamped to 255 (POSIX exit codes are 8-bit); a >255 failure count still
    # reports non-zero. The full count is printed above for the true total.
    print(f"=== TOTAL FAILURES (cargo+webserver): {total_failures} ===", file=sys.stderr)
    return min(total_failures, 255)


if __name__ == "__main__":
    sys.exit(main())
