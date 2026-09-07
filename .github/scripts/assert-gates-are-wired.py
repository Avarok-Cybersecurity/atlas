#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
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

# ---------------------------------------------------------------------------
# Every required context on `main`, and the job that produces it.
# ---------------------------------------------------------------------------
# Pulled from
#   gh api repos/.../branches/main/protection --jq '.required_status_checks.contexts[]'
# Branch protection names a STRING. Nothing in the tree makes that string
# resolve to a job, so a rename, a delete or a move retires a required check
# silently -- the job keeps passing under its new name and protection keeps
# waiting for the old one, or (worse) an admin removes the stale entry and the
# gate is simply gone. Listing them here turns each into a tree-checkable fact.
#
# `nested` names a reusable-workflow job: the context is then
# "<caller job name> / <called job name>", and BOTH halves have to hold.
REQUIRED_CONTEXTS = [
    ("cargo fmt --check", "ci.yml", "fmt", None),
    ("cargo clippy --tests", "ci.yml", "clippy", None),
    ("cargo test --workspace", "ci.yml", "test", None),
    ("typos", "ci.yml", "typos", None),
    ("SPDX license headers", "ci.yml", "license-headers", None),
    ("kernel shadow structure", "ci.yml", "kernel-structure", None),
    ("cargo test --features metal (macOS aarch64)", "ci.yml", "test-macos-metal", None),
    ("PR benchmark gate", "ci.yml", "pr-benchmark-gate-alias", None),
    ("release matrix / dry-run summary", "ci.yml", "release-matrix",
     ("release-build.yml", "dry-run-summary")),
    ("Enforce \u2264500 LoC per source file", "file-size-cap.yml", "check-file-sizes", None),
    ("cargo deny", "security.yml", "cargo-deny", None),
    ("cargo llvm-cov --workspace", "coverage.yml", "coverage", None),
    ("Build mdBook + rustdoc", "docs.yml", "build", None),
    ("nvcc -> PTX (all gb10 targets)", "kernel-compile.yml", "compile", None),
    ("Verify committed GDN binaries match PINS.sha256", "gdn-so-pin.yml", "verify-gdn-so-pins", None),
    ("No block_on under tui/ or recipe/", "tui-threading.yml", "no-blocking-on-the-render-thread", None),
    ("Build SvelteKit site", "site.yml", "build", None),
    ("Site unit tests", "site.yml", "unit", None),
    ("Merge-ancestry guard self-test", "merge-ancestry.yml", "self-test", None),
    ("CLAAssistant", "cla.yml", "CLAAssistant", None),
]

# An `if:` that names one of these suppresses the implicit `success()` that
# `needs:` otherwise carries, so the job still RUNS when a dependency failed.
STATUS_FUNCTIONS = ("always()", "cancelled()", "failure()", "success()")

problems: list[str] = []


def _doc(workflow: str) -> dict:
    return yaml.safe_load((WORKFLOWS / workflow).read_text())


def check_required_context(entry: tuple) -> None:
    """A required context must be PRODUCIBLE and must not be skippable by a
    dependency's failure.

    Two distinct ways a required check reaches a verdict that is not about the
    thing it claims to test:

      * it is gone or renamed -- protection waits on a string no job emits, and
        the PR hangs on "Expected" forever;

      * it carries `needs:` without a status function in its `if:`, so a FAILED
        dependency skips it -- and branch protection counts a skipped check as
        satisfied. The gate then reports safety it never measured. ci.yml's
        clippy/test and coverage.yml spell this correctly and say why in a
        comment; docs.yml's `build` did not, and a broken classify-diff.sh
        would have waved every PR past the docs gate.
    """
    context, workflow, job_id, nested = entry
    jobs = _doc(workflow).get("jobs") or {}
    if job_id not in jobs:
        problems.append(
            f"{workflow} has no `{job_id}` job, but branch protection requires "
            f"{context!r}; nothing would ever report it and every PR would hang"
        )
        return
    job = jobs[job_id]

    want = context.split(" / ")[0] if nested else context
    got = job.get("name") or job_id
    if got != want:
        problems.append(
            f"{workflow} `{job_id}` reports as {got!r}, but branch protection "
            f"requires {context!r}; a renamed job leaves the required context uncreated"
        )

    cond = str(job.get("if") or "")
    if job.get("needs") and not any(f in cond for f in STATUS_FUNCTIONS):
        problems.append(
            f"{workflow} `{job_id}` has `needs:` but no status function in its "
            f"`if:` ({cond or 'no if:'}); a FAILED dependency would skip it, and a "
            f"skipped required check counts as satisfied -- {context!r} would pass vacuously"
        )

    if nested:
        sub_wf, sub_job = nested
        if str(job.get("uses") or "") != f"./.github/workflows/{sub_wf}":
            problems.append(
                f"{workflow} `{job_id}` no longer calls {sub_wf}; the nested required "
                f"context {context!r} would never be created"
            )
        sub_jobs = _doc(sub_wf).get("jobs") or {}
        if sub_job not in sub_jobs:
            problems.append(f"{sub_wf} has no `{sub_job}` job; {context!r} cannot be reported")
            return
        sub_name = sub_jobs[sub_job].get("name") or sub_job
        if sub_name != context.split(" / ")[1]:
            problems.append(
                f"{sub_wf} `{sub_job}` reports as {sub_name!r}; {context!r} would never be created"
            )


# ---------------------------------------------------------------------------
# A gate must not disable the thing it is gating.
# ---------------------------------------------------------------------------
# `cargo test --features metal (macOS aarch64)` inherited ci.yml's
# workflow-level `ATLAS_SKIP_BUILD: "1"` -- which exists so the ubuntu jobs can
# type-check without nvcc. atlas-kernels' build.rs honours it before anything
# else and emits a stub whose `metallib_modules()` is `Vec::new()`, so
# `MetalGpuBackend::new` built an EMPTY library cache and all 35 parity tests
# died with `Metal: unknown module`. The required check was red about the stub,
# never about the kernels, and the merge queue was impassable for every non-web
# PR. Neither verdict it could produce was about Metal.
#
# The truthiness set is build.rs's own: `Ok("1") | Ok("true")`.
SKIP_TRUTHY = {"1", "true"}
STUB_FREE_STEPS = [
    {
        "workflow": "ci.yml",
        "job": "test-macos-metal",
        "step": "cargo test -p spark-runtime --features metal",
        "var": "ATLAS_SKIP_BUILD",
        "also": {"ATLAS_TARGET_HW": "metal"},
    },
]


def check_stub_free(spec: dict) -> None:
    doc = _doc(spec["workflow"])
    job = (doc.get("jobs") or {}).get(spec["job"])
    if job is None:
        problems.append(f"{spec['workflow']} has no `{spec['job']}` job to check for a stub build")
        return
    steps = [s for s in (job.get("steps") or []) if s.get("name") == spec["step"]]
    if len(steps) != 1:
        problems.append(
            f"{spec['workflow']} `{spec['job']}` has {len(steps)} steps named "
            f"{spec['step']!r}; this guard is pinned to exactly one"
        )
        return
    # Workflow env -> job env -> step env, in that precedence order.
    env = {}
    for scope in (doc.get("env") or {}, job.get("env") or {}, steps[0].get("env") or {}):
        env.update({k: str(v) for k, v in scope.items()})
    var = spec["var"]
    if env.get(var) in SKIP_TRUTHY:
        problems.append(
            f"{spec['workflow']} `{spec['job']}` runs {spec['step']!r} with {var}="
            f"{env[var]!r}, so atlas-kernels emits a stub and `metallib_modules()` is "
            f"empty; the suite can only report `Metal: unknown module` and its verdict "
            f"is about the stub, not the kernels"
        )
    for k, v in spec["also"].items():
        if env.get(k) != v:
            problems.append(
                f"{spec['workflow']} `{spec['job']}` runs {spec['step']!r} with "
                f"{k}={env.get(k)!r}, want {v!r}; without it build.rs takes the macOS "
                f"auto-skip and embeds no kernels"
            )


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
    for entry in REQUIRED_CONTEXTS:
        check_required_context(entry)
    for spec in STUB_FREE_STEPS:
        check_stub_free(spec)
    if problems:
        for p in problems:
            print(f"REFUSE: {p}", file=sys.stderr)
        sys.exit(1)
    for gate in GATES:
        via = f"needed by `{gate['needed_by']}`, " if gate["needed_by"] else ""
        print(f"ok: {gate['reports_as']} ({gate['workflow']}) {via}reports unconditionally in PRs and the queue")
    print(f"ok: all {len(REQUIRED_CONTEXTS)} required contexts resolve to a live job, "
          f"and none can be skipped by a dependency's failure")
    for spec in STUB_FREE_STEPS:
        print(f"ok: {spec['workflow']} `{spec['job']}` runs {spec['step']!r} against real kernels, not a build stub")


if __name__ == "__main__":
    main()
