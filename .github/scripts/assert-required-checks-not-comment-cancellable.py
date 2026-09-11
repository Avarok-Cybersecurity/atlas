#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fail if a REQUIRED check can be cancelled by the timing of a comment.

A cancelled check run is indistinguishable from a failed one to branch
protection and to `gh pr checks`: both read red, and the PR stops. That is
tolerable when the canceller is a NEW PUSH -- the superseding run reports and
the PR goes green again. It is not tolerable when the canceller is a COMMENT,
because nothing re-reports: the PR is simply stuck until a human re-dispatches
the workflow by hand.

Observed on #907, 2026-09-06: an explanatory comment followed seconds later by a
bare `/stamp` made the second `CLAAssistant` run cancel the first. `CLAAssistant`
is a required context, so the PR sat red on a check that had merely been shot,
with no way to notice from the PR page that "failure" meant "cancelled".

The rule: a workflow that (a) emits a required context and (b) can be triggered
by a comment must not carry `cancel-in-progress: true`. Serialise instead --
runner-seconds are worth far less than an unmergeable PR.

Push- and pull_request-triggered required checks are deliberately NOT flagged:
there the cancellation is a supersession and the replacement run reports.

Run with no arguments from the repo root.
"""
import pathlib
import sys

import yaml

# The contexts `main`'s branch protection requires BY NAME. Branch protection
# lives in the API, not in any committed file, so this list is a mirror -- and
# a stale mirror only ever makes this check WEAKER (a required context missing
# from it is simply not examined), never wrong in the dangerous direction.
REQUIRED_CONTEXTS = {
    "Build SvelteKit site",
    "Build mdBook + rustdoc",
    "CLAAssistant",
    "Enforce ≤500 LoC per source file",
    "Merge-ancestry guard self-test",
    "No block_on under tui/ or recipe/",
    "PR benchmark gate",
    "SPDX license headers",
    "Site unit tests",
    "Verify committed GDN binaries match PINS.sha256",
    "cargo clippy --tests",
    "cargo deny",
    "cargo fmt --check",
    "cargo llvm-cov --workspace",
    "cargo test --features metal (macOS aarch64)",
    "cargo test --workspace",
    "kernel shadow structure",
    "nvcc -> PTX (all gb10 targets)",
    "release matrix / dry-run summary",
    "typos",
}

# Triggers a human can fire repeatedly without changing the head commit.
COMMENT_TRIGGERS = {"issue_comment", "pull_request_review_comment", "pull_request_review"}


def cancels(section):
    """Whether this concurrency block can cost a required context its run.

    ★ `cancel-in-progress: true` is NOT the only way. From GitHub's docs: when a
    run is queued while another in the same group is in progress it goes
    PENDING, and "any previously pending job or workflow in the concurrency
    group will be cancelled." That happens whatever `cancel-in-progress` says —
    it governs the IN-PROGRESS run, not the pending one.

    So a shared group is enough. On 2026-09-06 `cla.yml` carried
    `cancel-in-progress: false`, added after the #907 incident and believed to
    close this class, and `CLAAssistant` still went missing on #934, #935 and
    #908 at once: each run for the current head was `completed/cancelled` with
    ZERO jobs. This function said those workflows were safe, because it only
    looked at the flag.

    Any concurrency block therefore counts. A required context is worth more
    than the runner-seconds a group saves.
    """
    return isinstance(section, dict) and bool(section.get("group"))


def main(root="."):
    problems = []
    wf_dir = pathlib.Path(root) / ".github/workflows"
    for path in sorted(wf_dir.glob("*.yml")):
        try:
            wf = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # a malformed workflow is a different test's job
            print(f"  skip {path.name}: {exc}")
            continue
        if not isinstance(wf, dict):
            continue
        # PyYAML parses a bare `on:` key as the boolean True.
        triggers = set((wf.get(True) or wf.get("on") or {}).keys())
        comment_fired = triggers & COMMENT_TRIGGERS
        if not comment_fired:
            continue
        for name, job in (wf.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            context = job.get("name", name)
            if context not in REQUIRED_CONTEXTS:
                continue
            if cancels(job.get("concurrency")) or (
                job.get("concurrency") is None and cancels(wf.get("concurrency"))
            ):
                problems.append(
                    f"{path.name}:{name} emits the required context {context!r}, "
                    f"is triggered by {sorted(comment_fired)}, and declares a "
                    f"concurrency group — a later event leaves this run pending and "
                    f"cancel it and leave the PR red on a check that was never run. "
                    f"Set cancel-in-progress: false."
                )
    if problems:
        print("a required check can be cancelled by comment timing:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("no required check is cancellable by comment timing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
