#!/usr/bin/env python3
"""Assert that jobs whose verdict is supposed to gate something actually do.

Two instances of one defect prompted this, and they were found a wave apart:

  #810  `Site unit tests` ran on every PR, reported green, and blocked neither
        the merge nor the deploy. It was not a required context, and `deploy`
        declared `needs: build` alone. A latching-state regression (#805)
        reached main during exactly that window.

  the merge-ancestry guard  `Merge-ancestry guard self-test` -- the test that
        proves the guard CAN fail -- was a required context, while `PR shares
        history with its base`, the guard's actual verdict on your branch, was
        not. An orphan-history PR (the #452 class, which squash-merges a diff
        that silently reverts everything before it) went red and merged anyway.

A check that reports without gating is worse than no check, because it reads as
safety. What this file pins:

  * `needed_by` -- some other job must refuse to run if this one failed. This
    is the half of "gating" that a file in the tree can express.

  * unconditionality -- the job must carry no `if:` and no `needs:`, and its
    workflow's `pull_request` trigger must carry no `paths:` filter. This is
    NOT a stylistic preference. A required context is only satisfiable if the
    job is *created*; a job held behind an `if:` is never created for the runs
    it skips, and GitHub blocks on it forever. Five workflows here carry that
    scar. So the assertion guards the fix from becoming the outage.

  * `merge_group` -- if the job is meant to be a required context, it must
    report in the merge queue too, or every queue entry deadlocks.

The other half of gating -- membership in `main`'s required-context list --
lives in branch protection, which no committed file can express. See
docs/ROBUSTNESS.md for the API call and the before/after diff.
"""
import sys
import pathlib
import yaml

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / "workflows"

# workflow, job, human name, job that must depend on it (or None), and whether
# the job is a required context on main (and so must also report in the queue).
GATES = [
    {
        "workflow": "site.yml",
        "job": "unit",
        "reports_as": "Site unit tests",
        "needed_by": "deploy",
        "required_context": True,
    },
    {
        "workflow": "merge-ancestry.yml",
        "job": "guard",
        "reports_as": "PR shares history with its base",
        "needed_by": None,
        "required_context": True,
    },
]

problems: list[str] = []


def check(gate: dict) -> None:
    path = WORKFLOWS / gate["workflow"]
    doc = yaml.safe_load(path.read_text())
    jobs = doc.get("jobs") or {}
    job_id, wf = gate["job"], gate["workflow"]

    if job_id not in jobs:
        problems.append(f"{wf} has no `{job_id}` job; this guard is pinned to a job that no longer exists")
        return
    job = jobs[job_id]

    if (job.get("name") or job_id) != gate["reports_as"]:
        problems.append(
            f"{wf} `{job_id}` reports as {job.get('name') or job_id!r}, not {gate['reports_as']!r}; "
            f"a renamed job leaves the required context uncreated"
        )

    consumer = gate["needed_by"]
    if consumer:
        needs = jobs.get(consumer, {}).get("needs") or []
        if isinstance(needs, str):
            needs = [needs]
        if job_id not in needs:
            problems.append(
                f"{wf} `{consumer}` does not need `{job_id}` (needs: {needs or 'nothing'}). "
                f"A failing {gate['reports_as']} would not stop it."
            )

    if job.get("if") is not None:
        problems.append(f"{wf} `{job_id}` grew an `if:`; it is a required context and must report on every run")
    if job.get("needs"):
        problems.append(f"{wf} `{job_id}` grew a `needs:`; a skipped dependency leaves the required context uncreated")

    on = doc.get(True) or doc.get("on") or {}
    pr = on.get("pull_request")
    if isinstance(pr, dict) and pr.get("paths"):
        problems.append(f"{wf} `pull_request` grew a `paths:` filter; the required context would never be created for other PRs")
    if gate["required_context"] and "merge_group" not in on:
        problems.append(f"{wf} has no `merge_group` trigger; `{gate['reports_as']}` would never report in the queue and every entry would deadlock")


def main() -> None:
    for gate in GATES:
        check(gate)
    if problems:
        for p in problems:
            print(f"REFUSE: {p}", file=sys.stderr)
        sys.exit(1)
    for gate in GATES:
        via = f"needed by `{gate['needed_by']}`, " if gate["needed_by"] else ""
        print(f"ok: {gate['reports_as']} ({gate['workflow']}) {via}reports unconditionally in PRs and the queue")


if __name__ == "__main__":
    main()
