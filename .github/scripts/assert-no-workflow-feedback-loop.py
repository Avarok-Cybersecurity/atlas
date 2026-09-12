#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""A workflow that pushes a branch must not be re-triggered by that branch's CI.

`governance-harvest.yml` triggers on `workflow_run: [CI] completed` and force-
pushes `bot/governance-harvest`, opening a pull request. That PR is a pull
request, so its CI completing re-triggered the harvest, which pushed again.

Measured 2026-09-04 before the fix: twelve harvest PRs opened and closed in one
day (#854..#871, one every 20-30 minutes), and **11 of the 27 runs queued at
that moment belonged to that one branch** -- about half the repository's runner
backlog, self-inflicted, starving every other PR of capacity.

The loop stayed hot because the harvest closes its own PR whenever it looks
unmergeable, and an unstamped PR always looks unmergeable: certification is HELD
until a human comments `/stamp`, and nobody stamps a bot PR. So each cycle went
open -> look broken -> close -> recreate -> run CI -> trigger the harvest.

THE RULE: if a workflow is triggered by `workflow_run` AND pushes a branch, its
job condition must exclude runs whose `head_branch` is that branch. Anything
else is a loop waiting for a reason to spin.

Scope: this checks the branch a workflow force-pushes, discovered from the
`git push --force ... refs/heads/<branch>` lines in its own run blocks, so it
cannot drift from a hand-kept list.
"""
import re
import sys
import pathlib
import yaml

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / "workflows"
PUSH = re.compile(r"refs/heads/([A-Za-z0-9._/-]+)")

problems: list[str] = []


def check(path: pathlib.Path) -> None:
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception:
        return
    if not isinstance(doc, dict):
        return
    on = doc.get(True) or doc.get("on") or {}
    if not isinstance(on, dict) or "workflow_run" not in on:
        return

    pushed = set()
    for job in (doc.get("jobs") or {}).values():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            run = step.get("run")
            if isinstance(run, str) and "push --force" in run:
                for line in run.splitlines():
                    if "push --force" in line or "refs/heads/" in line:
                        pushed |= set(PUSH.findall(line))
    pushed = {b for b in pushed if not b.startswith("$")}
    if not pushed:
        return

    conds = " ".join(
        str(job.get("if", "")) for job in (doc.get("jobs") or {}).values() if isinstance(job, dict)
    )
    for branch in sorted(pushed):
        if branch not in conds:
            problems.append(
                f"{path.name} is triggered by `workflow_run` and force-pushes `{branch}`, "
                f"but no job condition mentions that branch.\n"
                f"         That branch's own CI will re-trigger this workflow, which pushes it "
                f"again. Add `github.event.workflow_run.head_branch != '{branch}'` to the job's `if`."
            )


def main() -> None:
    files = sorted(WORKFLOWS.glob("*.y*ml"))
    for f in files:
        check(f)
    if problems:
        for p in problems:
            print(f"REFUSE: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"ok: no workflow_run-triggered workflow among {len(files)} can re-trigger itself via a branch it pushes")


if __name__ == "__main__":
    main()
