#!/usr/bin/env python3
"""Serve-matrix release gate — turn single_gpu_suite.py outputs into a PASS/FAIL.

`tests/run_all_models.py` boots each model×quant and writes one
`tests/all_models_results/<label>.json` per model (the schema emitted by
`tests/single_gpu_suite.py`). That orchestrator never exits non-zero on a test
failure — it only persists JSON. This script is the missing verdict: it reads
those results, applies explicit per-model bars, and exits 0 (ship) or 1 (block).

    python3 tests/run_all_models.py                 # produce results + _manifest.json
    python3 tests/gate_results.py                   # the gate — exit 0/1

Coverage is enforced, not assumed. run_all_models.py writes `_manifest.json`
listing every model the run *intended* to cover. This gate checks that roster:
a planned model that never booted writes no `<label>.json`, and that MISSING
result is a FAIL — not silently absent. Without this, a glob-only gate would
score the survivors and report a false green when half the matrix crashed at
boot. ("Serve all" means all planned, not all-that-happened-to-write.)

Design (see .claude/skills/atlas-release/references/verify-matrix.md §Gate):
  * SBIO — the decision logic here is pure over the results JSON; the only I/O is
    reading the files. No server/host/port is baked in.
  * PCND — every bar is an explicit constant chosen up front, with a rationale.
    No implicit "looks close enough"; no implicit "coverage was probably fine."
    A missing manifest is a hard FAIL unless coverage is explicitly waived.
  * SSOT — the roster lives in run_all_models.py (the manifest); this gate
    derives from it and never re-lists models.
A model "verifies" iff it is in the manifest AND its results file exists (it
booted past the /v1/models health check — single_gpu_suite exits before writing
if the server is unreachable) AND it clears every bar below.
"""

import argparse
import glob
import json
import os
import sys

MANIFEST_NAME = "_manifest.json"

# ── Per-model bars (PCND: explicit, rationale inline) ────────────────────────
COHERENCE_MIN_PASS = 2   # of 3 probes (factual + reasoning + creative); >=2 tolerates
                         #   the one temp>0 creative probe occasionally missing.
FIB_MIN_PASS = 1         # the fibonacci codegen smoke must exec-verify.
# Tools: known-gap parsers score every tool test "N/A" and must NOT fail the gate.
# A model whose parser IS wired up must land at least one real PASS — a result
# that is all WARN/FAIL (parser supposedly supported but never produced a call)
# is a regression.


def _status_pass(s):
    return "PASS" in (s or "")


def _status_na(s):
    return "N/A" in (s or "")


def verdict(model_result):
    """Pure: return the list of bar names this model FAILED (empty == verified)."""
    fails = []

    coh = model_result.get("coherence", []) or []
    coh_pass = sum(1 for r in coh if _status_pass(r.get("status")))
    if coh and coh_pass < COHERENCE_MIN_PASS:
        fails.append(f"coherence({coh_pass}/{len(coh)})")

    fib = model_result.get("fibonacci", []) or []
    fib_pass = sum(1 for r in fib if _status_pass(r.get("status")))
    if fib and fib_pass < FIB_MIN_PASS:
        fails.append("fibonacci")

    tc = model_result.get("tool_calls", []) or []
    if tc:
        any_pass = any(_status_pass(r.get("status")) for r in tc)
        all_na = all(_status_na(r.get("status")) for r in tc)
        if not any_pass and not all_na:
            fails.append("tool_calls")

    tps = [r.get("tps") for r in (model_result.get("tps", []) or [])
           if isinstance(r.get("tps"), (int, float))]
    if tps and (sum(tps) / len(tps)) <= 0:
        fails.append("tps(0)")

    return fails


def _is_model_result(obj):
    return isinstance(obj, dict) and "coherence" in obj and "model" in obj


def load_result_file(path):
    """Read one <label>.json. Return (model_result | None, error_str | None).

    A per-model file is written directly by single_gpu_suite.py, so it should be
    a flat model-result shape. Tolerate the historical dict-of-results aggregate
    by descending one level.
    """
    try:
        with open(path) as f:
            obj = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return None, str(e)
    if _is_model_result(obj):
        return obj, None
    if isinstance(obj, dict):
        for v in obj.values():
            if _is_model_result(v):
                return v, None
    return None, "malformed: no coherence/model fields"


def load_manifest(results_dir, manifest_path):
    """Return the planned [(label, model)] roster, or None if no manifest."""
    path = manifest_path or os.path.join(results_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        man = json.load(f)
    return [(e["label"], e.get("model", e["label"])) for e in man.get("labels", [])]


def gate_manifest(results_dir, roster):
    """Coverage-enforcing gate. Every planned label must have a passing result.

    Returns (verified_count, total, failures) where failures is
    [(label_or_model, [reasons])]. A planned model with no result file is a
    failure ("did not boot / no results"), which is the whole point.
    """
    failures = []
    for label, model in roster:
        name = model or label
        path = os.path.join(results_dir, f"{label}.json")
        if not os.path.isfile(path):
            print(f"  [FAIL] {name}  <- no result (did not boot / never wrote {label}.json)")
            failures.append((name, ["no-result"]))
            continue
        result, err = load_result_file(path)
        if result is None:
            print(f"  [FAIL] {name}  <- unreadable: {err}")
            failures.append((name, [f"unreadable: {err}"]))
            continue
        bars = verdict(result)
        mark = "PASS" if not bars else "FAIL"
        print(f"  [{mark}] {name}" + ("" if not bars else f"  <- {', '.join(bars)}"))
        if bars:
            failures.append((name, bars))
    return len(roster) - len(failures), len(roster), failures


def gate_glob(results_dir):
    """Coverage-UNENFORCED fallback: score whatever <label>.json files exist.

    Only reachable with --allow-missing-manifest. Cannot catch a model that
    crashed at boot (no file). Prints a loud warning so a green here is never
    mistaken for full-matrix coverage.
    """
    print("[gate] WARNING: no manifest — coverage is NOT enforced. A model that "
          "crashed at boot is invisible here. Pass a real run's _manifest.json "
          "for a trustworthy verdict.", file=sys.stderr)
    failures = []
    seen = 0
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path) == MANIFEST_NAME:
            continue
        result, err = load_result_file(path)
        if result is None:
            continue  # aggregate/partial files, not per-model results
        seen += 1
        name = result.get("model", os.path.basename(path))
        bars = verdict(result)
        mark = "PASS" if not bars else "FAIL"
        print(f"  [{mark}] {name}" + ("" if not bars else f"  <- {', '.join(bars)}"))
        if bars:
            failures.append((name, bars))
    return seen - len(failures), seen, failures


def main():
    ap = argparse.ArgumentParser(description="Serve-matrix release gate (exit 0=ship, 1=block).")
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "all_models_results"),
                    help="dir of per-model single_gpu_suite.py JSON outputs")
    ap.add_argument("--manifest", default=None,
                    help=f"path to the run manifest (default <results-dir>/{MANIFEST_NAME})")
    ap.add_argument("--allow-missing-manifest", action="store_true",
                    help="score only the files present (coverage NOT enforced). "
                         "Explicit opt-out — the default is to FAIL without a manifest.")
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"[gate] FAIL: no results dir {args.results_dir} — run tests/run_all_models.py first", file=sys.stderr)
        return 1

    roster = load_manifest(args.results_dir, args.manifest)
    if roster is None:
        if not args.allow_missing_manifest:
            print(f"[gate] FAIL: no {MANIFEST_NAME} in {args.results_dir}. Run "
                  f"tests/run_all_models.py (it writes the coverage manifest), or "
                  f"pass --allow-missing-manifest to score present files without "
                  f"coverage enforcement.", file=sys.stderr)
            return 1
        verified, total, failed = gate_glob(args.results_dir)
    else:
        if not roster:
            print("[gate] FAIL: manifest lists zero planned models.", file=sys.stderr)
            return 1
        verified, total, failed = gate_manifest(args.results_dir, roster)

    if total == 0:
        print(f"[gate] FAIL: no per-model results found in {args.results_dir}", file=sys.stderr)
        return 1

    print(f"\n[gate] {verified}/{total} models verified"
          + ("" if roster is None else " (full planned coverage)") + ".")
    if failed:
        print(f"[gate] FAIL — do not ship this image. {len(failed)} model(s) below bar:", file=sys.stderr)
        for name, bars in failed:
            print(f"         {name}: {', '.join(bars)}", file=sys.stderr)
        return 1
    print("[gate] PASS — serve matrix clean; image is shippable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
