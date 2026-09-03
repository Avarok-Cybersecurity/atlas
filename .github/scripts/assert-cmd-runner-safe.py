#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fail if the self-hosted command runner's label could execute untrusted code.

GitHub warns against self-hosted runners on public repos because a fork's PR can
run arbitrary code on your hardware. That warning does NOT apply to the
certification command workflow, for one specific reason: it pins
`ref: default_branch` and never checks out a PR head.

That is a property of the current file, not a guarantee. This test makes it a
guarantee: any workflow that reaches the command runner must never check out
untrusted code, and must never be reachable from a `pull_request` trigger.

Run with no arguments from the repo root.
"""
import pathlib
import re
import sys

import yaml

LABEL = "atlas-cmd"
# Triggers whose payload a fork controls. `pull_request_target` and
# `issue_comment` run from the DEFAULT branch, so they are safe on their own --
# what makes them unsafe is checking out the head ref, which is checked below.
FORK_CONTROLLED = {"pull_request"}
UNSAFE_REFS = (
    "github.event.pull_request.head",
    "github.event.pull_request.merge_commit_sha",
    "github.head_ref",
    "github.event.merge_group.head_ref",
)


def uses_cmd_runner(job):
    ro = job.get("runs-on")
    blob = str(ro)
    # Matches a bare label, a list, or the `vars.CMD_RUNNER_LABEL` indirection.
    return LABEL in blob or "CMD_RUNNER_LABEL" in blob


def main():
    problems = []
    for path in sorted(pathlib.Path(".github/workflows").glob("*.yml")):
        try:
            wf = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # a malformed workflow is a different test's job
            print(f"  skip {path.name}: {exc}")
            continue
        if not isinstance(wf, dict):
            continue
        triggers = set((wf.get(True) or wf.get("on") or {}).keys())
        for name, job in (wf.get("jobs") or {}).items():
            if not isinstance(job, dict) or not uses_cmd_runner(job):
                continue
            bad = triggers & FORK_CONTROLLED
            if bad:
                problems.append(
                    f"{path.name}:{name} runs on {LABEL} and is triggered by "
                    f"{sorted(bad)} — a fork could run its own code on our hardware."
                )
            for step in job.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                if "checkout" not in str(step.get("uses", "")):
                    continue
                ref = str((step.get("with") or {}).get("ref", ""))
                if not ref:
                    problems.append(
                        f"{path.name}:{name} checks out with no explicit ref on "
                        f"{LABEL}; the default is the event's head."
                    )
                elif any(u in ref for u in UNSAFE_REFS):
                    problems.append(
                        f"{path.name}:{name} checks out '{ref}' on {LABEL} — that is "
                        f"untrusted code on our own machine."
                    )
    if problems:
        print("the self-hosted command runner is reachable from untrusted code:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"no workflow on '{LABEL}' checks out untrusted code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
