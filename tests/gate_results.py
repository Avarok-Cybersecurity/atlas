#!/usr/bin/env python3
"""Serve-matrix release gate — turn single_gpu_suite.py outputs into a PASS/FAIL.

`tests/run_all_models.py` boots each model×quant and writes one
`tests/all_models_results/<label>.json` per model (the schema emitted by
`tests/single_gpu_suite.py`). That orchestrator never exits non-zero on a test
failure — it only persists JSON. This script is the missing verdict: it reads
those results, applies explicit per-model bars, and exits 0 (ship) or 1 (block).

    python3 tests/run_all_models.py                 # produce results + _manifest.json
    python3 tests/gate_results.py                   # the gate — exit 0/1
    python3 tests/gate_results.py --update-baselines # bless current tps (review+commit)

Beyond liveness (tps>0), the gate catches perf *regressions*: a committed
`tests/baselines/<label>.json` records the blessed tokens/sec, and a run more
than TPS_TOLERANCE below it fails. Models without a baseline are checked for
liveness only (with a loud note) until one is blessed — pass --require-baselines
to make a missing baseline a hard fail for a release milestone.

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
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baselines")

# ── Per-model bars (PCND: explicit, rationale inline) ────────────────────────
COHERENCE_MIN_PASS = 2   # of 3 probes (factual + reasoning + creative); >=2 tolerates
                         #   the one temp>0 creative probe occasionally missing.
FIB_MIN_PASS = 1         # the fibonacci codegen smoke must exec-verify.
TPS_TOLERANCE = 0.10     # tps regression bar. A committed tests/baselines/<label>.json
                         #   records the blessed tokens/sec; a run >10% below it is a
                         #   regression. With NO baseline the gate can only check
                         #   liveness (tps>0) and says so — never a silent pass, but
                         #   not a hard block unless --require-baselines (the stricter
                         #   bar is opt-in for a milestone, not the default).
# Tools: known-gap parsers score every tool test "N/A" and must NOT fail the gate.
# A model whose parser IS wired up must land at least one real PASS — a result
# that is all WARN/FAIL (parser supposedly supported but never produced a call)
# is a regression.


def _status_pass(s):
    return "PASS" in (s or "")


def _status_na(s):
    return "N/A" in (s or "")


def _avg_tps(model_result):
    """Mean measured tokens/sec across the tps probes, or None if none numeric."""
    vals = [r.get("tps") for r in (model_result.get("tps", []) or [])
            if isinstance(r.get("tps"), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def verdict(model_result, baseline=None, require_baseline=False):
    """Pure: return the list of bar names this model FAILED (empty == verified).

    `baseline` is the {"tps": <blessed tok/s>} dict for this label (or None). The
    tps bar is: dead server (avg<=0) always fails; if a baseline exists, falling
    >TPS_TOLERANCE below it is a regression; if none exists it fails only when
    `require_baseline` is set (else liveness-only, surfaced as a note by caller).
    """
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

    avg = _avg_tps(model_result)
    if avg is not None:
        if avg <= 0:
            fails.append("tps(0)")
        elif baseline and isinstance(baseline.get("tps"), (int, float)) and baseline["tps"] > 0:
            floor = baseline["tps"] * (1 - TPS_TOLERANCE)
            if avg < floor:
                fails.append(f"tps({avg:.1f}<{floor:.1f})")
        elif require_baseline:
            fails.append("tps(no-baseline)")

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


def load_baseline(label, baseline_dir):
    """Return the {"tps": ...} baseline for a label, or None if not committed."""
    path = os.path.join(baseline_dir, f"{label}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def write_baseline(label, data, baseline_dir):
    """Merge `data` into tests/baselines/<label>.json (create dir if needed)."""
    os.makedirs(baseline_dir, exist_ok=True)
    path = os.path.join(baseline_dir, f"{label}.json")
    existing = {}
    if os.path.isfile(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(data)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")


def _score_one(label, name, result, baseline_dir, require_baseline):
    """(bars, note): score a loaded result against its baseline; note flags a
    positive-tps model with no baseline (liveness-only, not a block by default)."""
    baseline = load_baseline(label, baseline_dir)
    bars = verdict(result, baseline=baseline, require_baseline=require_baseline)
    note = ""
    avg = _avg_tps(result)
    if avg and avg > 0 and not (baseline and isinstance(baseline.get("tps"), (int, float))) \
            and not require_baseline:
        note = "  (no tps baseline — liveness only; run --update-baselines)"
    return bars, note


def load_manifest(results_dir, manifest_path):
    """Return the planned [(label, model)] roster, or None if no manifest."""
    path = manifest_path or os.path.join(results_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        man = json.load(f)
    return [(e["label"], e.get("model", e["label"])) for e in man.get("labels", [])]


def gate_manifest(results_dir, roster, baseline_dir=BASELINE_DIR, require_baseline=False):
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
        bars, note = _score_one(label, name, result, baseline_dir, require_baseline)
        mark = "PASS" if not bars else "FAIL"
        print(f"  [{mark}] {name}" + ("" if not bars else f"  <- {', '.join(bars)}") + note)
        if bars:
            failures.append((name, bars))
    return len(roster) - len(failures), len(roster), failures


def gate_glob(results_dir, baseline_dir=BASELINE_DIR, require_baseline=False):
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
        label = os.path.splitext(os.path.basename(path))[0]
        name = result.get("model", label)
        bars, note = _score_one(label, name, result, baseline_dir, require_baseline)
        mark = "PASS" if not bars else "FAIL"
        print(f"  [{mark}] {name}" + ("" if not bars else f"  <- {', '.join(bars)}") + note)
        if bars:
            failures.append((name, bars))
    return seen - len(failures), seen, failures


def update_baselines(results_dir, roster, baseline_dir):
    """Write tests/baselines/<label>.json = {"tps": <avg>} from present results.

    Roster (label, model) list if a manifest exists, else every present result
    file. Only positive tps is recorded. Returns [(label, avg)] written.
    """
    if roster is None:
        items = [(os.path.splitext(os.path.basename(p))[0], None)
                 for p in sorted(glob.glob(os.path.join(results_dir, "*.json")))
                 if os.path.basename(p) != MANIFEST_NAME]
    else:
        items = roster
    written = []
    for label, _ in items:
        path = os.path.join(results_dir, f"{label}.json")
        if not os.path.isfile(path):
            continue
        result, _err = load_result_file(path)
        if result is None:
            continue
        avg = _avg_tps(result)
        if avg is None or avg <= 0:
            continue
        write_baseline(label, {"tps": round(avg, 2)}, baseline_dir)
        written.append((label, avg))
    return written


def main():
    ap = argparse.ArgumentParser(description="Serve-matrix release gate (exit 0=ship, 1=block).")
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(__file__), "all_models_results"),
                    help="dir of per-model single_gpu_suite.py JSON outputs")
    ap.add_argument("--manifest", default=None,
                    help=f"path to the run manifest (default <results-dir>/{MANIFEST_NAME})")
    ap.add_argument("--allow-missing-manifest", action="store_true",
                    help="score only the files present (coverage NOT enforced). "
                         "Explicit opt-out — the default is to FAIL without a manifest.")
    ap.add_argument("--baseline-dir", default=BASELINE_DIR,
                    help="dir of committed per-label tps baselines (tests/baselines)")
    ap.add_argument("--update-baselines", action="store_true",
                    help="write current tps as the new baseline instead of gating "
                         "(deliberate, reviewed refresh — commit the diff). Exits 0.")
    ap.add_argument("--require-baselines", action="store_true",
                    help="a planned model with no committed tps baseline FAILS "
                         "(stricter milestone bar; default warns + liveness-only).")
    args = ap.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"[gate] FAIL: no results dir {args.results_dir} — run tests/run_all_models.py first", file=sys.stderr)
        return 1

    roster = load_manifest(args.results_dir, args.manifest)

    if args.update_baselines:
        written = update_baselines(args.results_dir, roster, args.baseline_dir)
        for label, avg in written:
            print(f"  [baseline] {label} <- {avg:.1f} tok/s")
        print(f"[gate] wrote {len(written)} baseline(s) to {args.baseline_dir}. "
              f"Review the diff and commit.")
        return 0

    if roster is None:
        if not args.allow_missing_manifest:
            print(f"[gate] FAIL: no {MANIFEST_NAME} in {args.results_dir}. Run "
                  f"tests/run_all_models.py (it writes the coverage manifest), or "
                  f"pass --allow-missing-manifest to score present files without "
                  f"coverage enforcement.", file=sys.stderr)
            return 1
        verified, total, failed = gate_glob(args.results_dir, args.baseline_dir, args.require_baselines)
    else:
        if not roster:
            print("[gate] FAIL: manifest lists zero planned models.", file=sys.stderr)
            return 1
        verified, total, failed = gate_manifest(args.results_dir, roster, args.baseline_dir, args.require_baselines)

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
