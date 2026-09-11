#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
Which workflow runs are stuck before they ever created a job.

A run that never reaches job creation is the worst failure shape this pipeline
has, because it is INVISIBLE. It produces no check runs at all, so a required
context reads as *absent* rather than red: `gh pr checks` says "0 fail" while
the PR is unmergeable and nothing anywhere is showing as broken. Worse, such a
run holds its ref's concurrency group, so every later run on that ref queues
behind it forever, and `gh run cancel` reports success without doing anything —
only the force-cancel endpoint releases it.

That is not hypothetical: it happened on #975, cost most of an hour, and was
found only by noticing that a run had `jobs=0` long after it was created.

The decision is here rather than inline in the workflow so it can be tested
against fixtures instead of against live Actions state. Reads a runs listing on
stdin, writes the ids to force-cancel on stdout, one per line.

Input: {"now": "<iso8601>", "threshold_minutes": N, "runs": [...]} where each
run is GitHub's own shape, plus a "job_count" the caller resolved.
"""

import datetime
import json
import sys

# Statuses that mean "has not started doing work". A run that is `in_progress`
# has jobs by definition and is somebody else's problem.
NOT_STARTED = {"queued", "pending", "waiting", "requested"}


def _parse(ts):
    """GitHub stamps are Zulu; datetime wants an offset it recognises."""
    return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def stuck_run_ids(payload):
    """
    Ids of runs that have not started AND have created no job AND have been in
    that state longer than the threshold.

    All three conditions are required, and each rules out a legitimate case:
    a run waiting on a busy runner pool is `queued` but young; a run that is
    merely slow has jobs; a run held for a deployment approval is `waiting` but
    will have jobs once released.
    """
    now = _parse(payload["now"])
    threshold = datetime.timedelta(minutes=int(payload["threshold_minutes"]))
    stuck = []
    for run in payload.get("runs") or []:
        if run.get("status") not in NOT_STARTED:
            continue
        # A missing job_count is NOT zero. The caller failed to resolve it, and
        # force-cancelling on an unknown is the fail-open direction that would
        # kill healthy runs.
        if run.get("job_count") is None:
            continue
        if int(run["job_count"]) != 0:
            continue
        if now - _parse(run["created_at"]) < threshold:
            continue
        stuck.append(int(run["id"]))
    return stuck


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:
        # Refuse rather than pass: a guard that cannot read its input and exits
        # 0 reports "nothing stuck" for every run forever.
        print(f"cannot parse the run listing: {exc}", file=sys.stderr)
        return 2
    if "now" not in payload or "threshold_minutes" not in payload:
        print("payload needs both 'now' and 'threshold_minutes'", file=sys.stderr)
        return 2
    for run_id in stuck_run_ids(payload):
        print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
