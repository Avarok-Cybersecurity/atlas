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

There is a SECOND self-hosted pool with a different bargain. `atlas-pr-cheap`
exists precisely to run the cheap PR checks that were starving in the hosted
queue, so it DOES execute PR code — but only from branches in this repository.
A fork's PR must still go to a GitHub-hosted runner, and the only thing standing
between those two outcomes is one expression in `runs-on`. So this test pins that
expression: a job may name the cheap pool under a `pull_request` trigger only if
its `runs-on` also carries the same-repo comparison. Forget the guard and CI goes
red here, not on the day a fork sends a pull request.

Run with no arguments from the repo root.
"""
import pathlib
import re
import sys

import yaml

LABEL = "atlas-cmd"
# The cheap PR pool. Routed through a repository variable so
# cmd-runner-health.yml can flip the whole fleet back to hosted without a code
# change -- the same escape hatch CMD_RUNNER_LABEL already provides.
# Two pools, same bargain, different capability: `atlas-pr-cheap` is every
# runner, `atlas-pr-rust` is the subset carrying a Rust toolchain (a bounded
# subset ON PURPOSE -- eight concurrent `cargo test --workspace` target
# directories would fill the disk).
PR_POOLS = (
    ("atlas-pr-cheap", "PR_CHEAP_RUNNER"),
    ("atlas-pr-rust", "PR_RUST_RUNNER"),
)
PR_LABEL = PR_POOLS[0][0]
PR_VAR = PR_POOLS[0][1]
# The exact comparison that keeps a fork off our hardware. Matched as a
# substring with whitespace collapsed, so reflowing the YAML cannot break it
# while changing its meaning could not survive it.
SAME_REPO_GUARD = (
    "github.event.pull_request.head.repo.full_name == github.repository"
)
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


def pr_pool_of(job):
    """Which self-hosted PR pool this job routes to, or None."""
    blob = str(job.get("runs-on"))
    for label, var in PR_POOLS:
        if label in blob or var in blob:
            return label
    return None


def guards_against_forks(job):
    """Does this job's `runs-on` restrict the cheap pool to this repository?"""
    blob = " ".join(str(job.get("runs-on")).split())
    return SAME_REPO_GUARD in blob


def routing_blob(job):
    """The cheap pool's `runs-on`, whitespace-collapsed for comparison."""
    return " ".join(str(job.get("runs-on")).split())


def main():
    problems = []
    routed = []
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
            if not isinstance(job, dict):
                continue
            pool = pr_pool_of(job)
            if pool is not None:
                routed.append((pool, routing_blob(job), f"{path.name}:{name}"))
                if triggers & FORK_CONTROLLED and not guards_against_forks(job):
                    problems.append(
                        f"{path.name}:{name} runs on {pool} under a "
                        f"pull_request trigger without the same-repo guard "
                        f"({SAME_REPO_GUARD}) in runs-on — a fork's PR would "
                        f"execute on our hardware."
                    )
                for step in job.get("steps") or []:
                    if not isinstance(step, dict):
                        continue
                    if "checkout" not in str(step.get("uses", "")):
                        continue
                    ref = str((step.get("with") or {}).get("ref", ""))
                    if any(u in ref for u in UNSAFE_REFS):
                        problems.append(
                            f"{path.name}:{name} checks out '{ref}' on "
                            f"{pool} — an explicit fork ref defeats the "
                            f"same-repo guard."
                        )
            if not uses_cmd_runner(job):
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
    # SSOT: the routing expression is repeated per job because GitHub Actions
    # has no anchors and no per-job include. Repetition is tolerable only while
    # every copy is IDENTICAL -- one job drifting to a weaker expression is
    # exactly the hole this file exists to close, and it would not be visible in
    # review of a 19-job diff. So the shapes are compared, not just each one's
    # guard.
    by_pool = {}
    for pool, blob, where in routed:
        by_pool.setdefault(pool, {}).setdefault(blob, []).append(where)
    for pool, shapes in sorted(by_pool.items()):
        if len(shapes) > 1:
            listing = "; ".join(
                f"{len(v)} job(s) use {k!r}"
                for k, v in sorted(shapes.items(), key=lambda kv: -len(kv[1]))
            )
            problems.append(
                f"the {pool} pool is routed with more than one expression — "
                f"{listing}. Every job on one pool must use the same one."
            )

    if problems:
        print("the self-hosted command runner is reachable from untrusted code:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(
        f"no workflow on '{LABEL}' checks out untrusted code, and every "
        f"job on {[p[0] for p in PR_POOLS]} under a pull_request trigger carries "
        f"the same-repo guard, one expression per pool."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
