#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Every `needs.<job>.outputs.<name>` must be exported by that job.

★ A STEP OUTPUT IS NOT A JOB OUTPUT, AND THE DIFFERENCE IS SILENT.

`ci.yml`'s `changes` job computed `is_stack_layer`, printed it to the log, and
never listed it under the job's `outputs:`. Consumers therefore read the EMPTY
STRING, and the two consumers failed in opposite directions:

    builds_binaries != 'false'   ->  empty is not 'false'  ->  BUILT (safe by luck)
    is_stack_layer   == 'true'   ->  empty is not 'true'   ->  NOT a stack layer

The second one is the expensive half. `release-build.yml` skips its nine build
legs for a lower stack layer precisely so that N layers do not spend N release
matrices to land one tree -- and every layer was being told it was not a layer.
Measured 2026-09-07 on PR #946: the classify job logged `is_stack_layer=true`
while `release matrix / validate inputs` in the SAME run reported
`stack_layer: false`, and nine builds ran on a pool completing ~8 jobs an hour.

Nothing in Actions warns about this. An unexported output is not an error, not
a warning, not even a lint -- it is an empty string, and an empty string is a
perfectly good value for an `if:` to be false about.

WHAT THIS CHECKS. For every workflow file: collect each job's declared
`outputs:` keys, then find every `needs.<job>.outputs.<name>` reference in the
same file and require `<name>` to be among that job's declared keys. A
reference to a job that is not in the same file is skipped (reusable-workflow
outputs live elsewhere and are checked by their own call site).

WHAT IT CANNOT SEE, stated rather than hidden:
  * whether the exported expression names a step that exists, or the right one;
  * outputs consumed across workflow files (`workflow_call` outputs);
  * a job output exported but never consumed -- harmless, and not its business.

Stdlib only, no network. Exit 1 on any unexported consumer.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

NEEDS = re.compile(r"needs\.([A-Za-z0-9_-]+)\.outputs\.([A-Za-z0-9_-]+)")


def check(path: Path) -> list[str]:
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception as exc:  # a workflow that will not parse is CI's problem, not ours
        return [f"{path}: could not parse: {exc}"]
    if not isinstance(doc, dict):
        return []
    jobs = doc.get("jobs")
    if not isinstance(jobs, dict):
        return []

    declared = {
        name: set((job.get("outputs") or {}).keys())
        for name, job in jobs.items()
        if isinstance(job, dict)
    }

    bad = []
    for job_name, ref_job, ref_out in (
        (jn, m.group(1), m.group(2))
        for jn, job in jobs.items()
        if isinstance(job, dict)
        for m in NEEDS.finditer(yaml.safe_dump(job))
    ):
        if ref_job not in declared:
            continue  # not a job in this file
        if ref_out not in declared[ref_job]:
            bad.append(
                f"{path}: job '{job_name}' reads needs.{ref_job}.outputs.{ref_out}, "
                f"but job '{ref_job}' does not export '{ref_out}' "
                f"(it exports: {', '.join(sorted(declared[ref_job])) or 'nothing'}). "
                f"That reference is the empty string at runtime."
            )
    return bad


def main() -> int:
    root = Path(".github/workflows")
    if not root.is_dir():
        print(f"REFUSING: {root} does not exist — this guard cannot find its input")
        return 1
    files = sorted(root.glob("*.yml")) + sorted(root.glob("*.yaml"))
    if not files:
        print("REFUSING: no workflow files found — a guard that finds nothing must fail")
        return 1
    bad = [msg for f in files for msg in check(f)]
    for msg in bad:
        print(f"::error title=Unexported job output::{msg}")
    print(f"checked {len(files)} workflow file(s); {len(bad)} unexported consumer(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
