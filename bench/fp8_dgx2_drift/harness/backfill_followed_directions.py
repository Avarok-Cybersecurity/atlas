#!/usr/bin/env python3
"""Backfill the `followed_directions` metric onto already-scored run JSONs.

For runs scored before score_run.py gained the metric, recompute it from the
saved opencode event log (/tmp/harness-<tier>-r<N>.json) + the target dir
(/tmp/harness-<tier>-r<N>) and splice the block into runs/run_<tier>_<N>.json.

Purely additive: only the top-level `followed_directions` key is written; every
other field is preserved byte-for-byte via a round-trip load/dump. Idempotent
(re-running overwrites only that key). Skips a run if its event log is gone.

Usage:
    python3 backfill_followed_directions.py [--tier TIER ...] [--force]
    (no --tier  → every tier found under runs/)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

_HD = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HD))
from followed_directions import _load_events, compute_followed_directions  # noqa: E402

RUNS_DIR = _HD / "runs"
_RUN_RE = re.compile(r"^run_(?P<tier>.+)_(?P<n>\d+)\.json$")


def backfill_one(run_json: pathlib.Path, force: bool) -> str:
    m = _RUN_RE.match(run_json.name)
    if not m:
        return "skip(name)"
    tier, n = m.group("tier"), m.group("n")
    try:
        record = json.loads(run_json.read_text())
    except Exception as e:
        return f"skip(unreadable: {e})"
    if "followed_directions" in record and not force:
        return "skip(present)"

    events_path = pathlib.Path(f"/tmp/harness-{tier}-r{n}.json")
    target = pathlib.Path(f"/tmp/harness-{tier}-r{n}")
    if not events_path.exists():
        return "skip(no-event-log)"

    events = _load_events(events_path)
    fd = compute_followed_directions(events, target)
    # Insert right after 'webserver' to mirror score_run.py's key order, but
    # plain assignment is fine — JSON has no ordering contract consumers rely on.
    record["followed_directions"] = fd
    run_json.write_text(json.dumps(record, indent=2))
    fdir = fd.get("followed_directions")
    return f"ok(followed={fdir} steps={fd.get('steps_completed')}/{fd.get('steps_total')})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", action="append", default=None,
                    help="restrict to these tier(s); repeatable. Default: all.")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if followed_directions already present")
    args = ap.parse_args()

    if not RUNS_DIR.is_dir():
        print(f"no runs dir: {RUNS_DIR}", file=sys.stderr)
        return 1

    files = sorted(RUNS_DIR.glob("run_*.json"))
    if args.tier:
        keep = set(args.tier)
        files = [f for f in files if (_RUN_RE.match(f.name) and _RUN_RE.match(f.name).group("tier") in keep)]

    n_ok = 0
    for f in files:
        status = backfill_one(f, args.force)
        if status.startswith("ok"):
            n_ok += 1
        print(f"{f.name}: {status}")
    print(f"\nbackfilled {n_ok}/{len(files)} run files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
