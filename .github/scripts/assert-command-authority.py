#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Only the commands that are meant to change a verdict may change one.

`/help` and `/review` post text. They must never mint a Stamp, a Seal or an
Expedite, never re-run CI, and never PATCH anything -- because their permission
checks are correspondingly weak: `/review` is open to anyone who can comment.
If a future edit gave `/review` the ability to create a check run, an outside
contributor could mint their own seal by asking a question.

That is true of the file today by inspection. This makes it true by rule.

Also pins the workflow to least privilege: a top-level `permissions:` wider than
`contents: read` would hand every step of it more authority than any step needs.
"""
import pathlib
import sys

import yaml

WORKFLOW = pathlib.Path(".github/workflows/certification-commands.yml")
# Verbs that legitimately change state, and the marks each may mint.
MAY_MINT = {"/stamp and /seal", "/expedite"}
FORBIDDEN = (
    ("check-runs", "mint a check run"),
    ("/rerun", "re-run CI"),
    ("-X PATCH", "PATCH a resource"),
    ("-X PUT", "PUT a resource"),
    ("-X DELETE", "DELETE a resource"),
)
ALLOWED_PERMISSIONS = {"contents": "read"}


def main():
    wf = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    problems = []

    perms = wf.get("permissions")
    if perms != ALLOWED_PERMISSIONS:
        problems.append(
            f"top-level permissions are {perms!r}, expected {ALLOWED_PERMISSIONS!r} — "
            f"every step inherits these, and none of them needs more."
        )

    for job_name, job in (wf.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            name = step.get("name") or ""
            run = step.get("run") or ""
            if not run or not name.startswith("/"):
                continue
            if name in MAY_MINT:
                continue
            for needle, what in FORBIDDEN:
                if needle in run:
                    problems.append(
                        f"{job_name}:{name} can {what} ('{needle}'), but only "
                        f"{sorted(MAY_MINT)} may change a verdict. {name}'s permission "
                        f"check is weaker on purpose because it only posts text."
                    )

    if problems:
        print("a command has more authority than its permission check justifies:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("only the state-changing commands can change state; permissions are minimal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
