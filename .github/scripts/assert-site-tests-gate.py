#!/usr/bin/env python3
"""Assert the site's unit suite actually gates something.

Issue #810: `Site unit tests` ran on every PR, reported green, and blocked
nothing -- it was not a required context, and `deploy` declared `needs: build`
alone. A red suite stopped neither the merge nor the deploy for the whole life
of the suite. A check that reports without gating is worse than no check,
because it reads as safety.

Two things have to hold, and they are not the same thing:

  1. `deploy` must not run unless `unit` passed. This is the half that is
     checkable offline, from the workflow file, which is why it lives here.
  2. `Site unit tests` must be a required context on `main`. That half lives in
     branch protection, which no file in the tree can express -- see
     docs/ROBUSTNESS.md for the API call and why it is safe.

The second assertion below guards the *first* one from backfiring. A required
context is only safe if the job reports on every PR: a job held behind an `if:`
or a `needs:` on a diff classifier is simply never created for the PRs it
skips, and GitHub waits for it forever. Five workflows in this repo carry that
lesson. So if someone later makes `unit` conditional, this fails loudly here
rather than silently deadlocking every non-site PR.
"""
import sys
import pathlib
import yaml

WORKFLOW = pathlib.Path(__file__).resolve().parents[1] / "workflows" / "site.yml"
SUITE = "unit"
CONSUMER = "deploy"

def fail(msg: str) -> None:
    print(f"REFUSE: {msg}", file=sys.stderr)
    sys.exit(1)

def main() -> None:
    doc = yaml.safe_load(WORKFLOW.read_text())
    jobs = doc.get("jobs") or {}

    for name in (SUITE, CONSUMER):
        if name not in jobs:
            fail(f"site.yml has no `{name}` job; this guard is pinned to a job that no longer exists")

    needs = jobs[CONSUMER].get("needs") or []
    if isinstance(needs, str):
        needs = [needs]
    if SUITE not in needs:
        fail(
            f"site.yml `{CONSUMER}` does not need `{SUITE}` (needs: {needs or 'nothing'}). "
            f"A failing site test would not stop the deploy."
        )

    # `unit` must be unconditional, or making it a required context deadlocks
    # every PR for which it is skipped. Note this checks the *job*; the
    # workflow's `pull_request` trigger carries no `paths:` filter, which is the
    # other half of the same requirement and is asserted just below.
    if jobs[SUITE].get("if") is not None:
        fail(f"site.yml `{SUITE}` grew an `if:`; it is a required context and must report on every PR")
    if jobs[SUITE].get("needs"):
        fail(f"site.yml `{SUITE}` grew a `needs:`; a skipped dependency would leave the required context uncreated")

    on = doc.get(True) or doc.get("on") or {}
    pr = on.get("pull_request")
    if isinstance(pr, dict) and pr.get("paths"):
        fail("site.yml `pull_request` grew a `paths:` filter; the required context would never be created for other PRs")

    print(f"ok: site.yml `{CONSUMER}` needs `{SUITE}`, and `{SUITE}` reports unconditionally")

if __name__ == "__main__":
    main()
