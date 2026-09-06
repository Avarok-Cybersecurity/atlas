#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Every input passed to a reusable workflow must be declared by it.

`jobs.<id>.with.<key>` on a job that `uses:` a local reusable workflow is
checked by GitHub at DISPATCH time, not at parse time. Pass a key the callee
does not declare and the whole calling workflow fails to start with
"Invalid input, <key> is not defined in the referenced workflow" — so every
context that workflow was going to report is simply never created. Branch
protection cannot tell "never created" from "still running": the PR waits
forever on a check nobody will write.

`release-build.yml:236` already records this repo losing an hour to a
required context that no runner would ever answer. This is the same wound
from the other side.

Caught this for real on 2026-09-06: porting `builds_binaries` out of #876
brought `stack_layer` with it in `ci.yml`'s `with:` block, but the input
DECLARATION lived in a third commit that was not ported. Every YAML file
parsed, every shell script was valid, and the selftest was green — the
failure only exists in the relationship between two files.

Both directions are checked:

  * a key passed but not declared  -> the dispatch error above;
  * a `required: true` input not passed -> the same dispatch error.

Only LOCAL callees (`./.github/workflows/x.yml`) are resolvable, so
third-party `owner/repo/.github/workflows/x.yml@ref` uses are skipped rather
than guessed at.

Run with no arguments from the repo root, or pass the root as argv[1].
"""
import pathlib
import sys

import yaml


def triggers_of(wf):
    # PyYAML parses a bare `on:` key as the boolean True.
    on = wf.get(True) or wf.get("on") or {}
    return on if isinstance(on, dict) else {}


def load(path):
    try:
        wf = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None  # a malformed workflow is a different check's job
    return wf if isinstance(wf, dict) else None


def main(root="."):
    wf_dir = pathlib.Path(root) / ".github/workflows"
    files = sorted(wf_dir.glob("*.y*ml"))

    declared = {}
    for path in files:
        wf = load(path)
        if wf is None:
            continue
        on = triggers_of(wf)
        if "workflow_call" in on:
            # `workflow_call:` with nothing under it parses to None, and that
            # is a perfectly valid callee taking no inputs. Testing the VALUE
            # rather than the KEY made the first draft of this script report
            # ci.yml as "not a workflow_call workflow" -- a false alarm about
            # release.yml that took a trigger dump to disprove. Presence of the
            # key is the question; the inputs mapping may legitimately be empty.
            call = on["workflow_call"] or {}
            declared[path.name] = (call.get("inputs") or {})

    problems = []
    for path in files:
        wf = load(path)
        if wf is None:
            continue
        for job_id, job in (wf.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            uses = str(job.get("uses", ""))
            # Local callee only: `./.github/workflows/name.yml`.
            if not uses.startswith("./.github/workflows/"):
                continue
            callee = uses.rsplit("/", 1)[-1]
            if callee not in declared:
                problems.append(
                    f"{path.name}:{job_id} uses {uses}, which is not a "
                    f"workflow_call workflow in this repository"
                )
                continue
            spec = declared[callee]
            passed = set(job.get("with") or {})
            extra = sorted(passed - set(spec))
            missing = sorted(
                k for k, v in spec.items()
                if isinstance(v, dict) and v.get("required") and k not in passed
            )
            if extra:
                problems.append(
                    f"{path.name}:{job_id} passes {extra} to {callee}, which does not "
                    f"declare them — the calling workflow will fail to start and every "
                    f"context it reports will never be created"
                )
            if missing:
                problems.append(
                    f"{path.name}:{job_id} omits required input(s) {missing} of {callee} "
                    f"— same failure, from the other side"
                )

    if problems:
        print("a reusable-workflow call site does not match its callee:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"ok: every local reusable-workflow call site among {len(files)} workflows matches its callee")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
