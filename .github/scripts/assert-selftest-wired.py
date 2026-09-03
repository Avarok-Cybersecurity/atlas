#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""The certification self-test must actually run, in a job that must pass.

Eighty-one checks hang on one line in one workflow. Delete that line and every
one of them stops running -- silently, with the job it lived in still green.

This cannot live inside the self-test itself: if the step is removed the suite
never executes, so a check inside it can never fire. It therefore runs from a
DIFFERENT required job, so removing the self-test breaks something else that
must pass.

Asserts two things:
  1. some job invokes certification-selftest.sh, and
  2. that job reports under a name branch protection requires -- a suite in a
     job nobody has to pass is a suite that can be ignored.
"""
import pathlib
import sys

import yaml

SUITE = "certification-selftest.sh"
# Branch protection on main requires this context. Kept as a literal because the
# assertion has to work without network access; if protection changes, this line
# is the one to update, and the failure message says so.
REQUIRED_CONTEXTS = {"cargo deny"}


def main():
    hosts = []
    for path in sorted(pathlib.Path(".github/workflows").glob("*.yml")):
        try:
            wf = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(wf, dict):
            continue
        for job_key, job in (wf.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                if isinstance(step, dict) and SUITE in str(step.get("run", "")):
                    hosts.append((path.name, job_key, job.get("name") or job_key))

    if not hosts:
        print(f"nothing runs {SUITE}.")
        print("  The certification self-test is 81 checks and 50 negative controls.")
        print("  If this is deliberate, delete the suite too -- a suite that does not")
        print("  run is worse than none, because the repo still looks tested.")
        return 1

    enforced = [h for h in hosts if h[2] in REQUIRED_CONTEXTS]
    if not enforced:
        names = ", ".join(sorted({h[2] for h in hosts}))
        print(f"{SUITE} runs, but only in non-required job(s): {names}")
        print(f"  Branch protection requires: {sorted(REQUIRED_CONTEXTS)}")
        print("  A failing suite would not block a merge, so it does not gate anything.")
        return 1

    for f, k, n in enforced:
        print(f"{SUITE} runs in {f}:{k} (reports as '{n}', which is required).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
