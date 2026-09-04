#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""A required check must not be cancellable by an event that cannot replace it.

`cancel-in-progress: true` is a runner-saving setting, and it is safe when the
event that cancels also schedules a replacement on the same commit: a second
push supersedes the first, and the new run writes the check the old one would
have. It is NOT safe when the concurrency group spans two trigger types that
report to different places.

That shipped here. `cla.yml` grouped on the PR number across both
`issue_comment` and `pull_request_target`, with cancelling on. Only the
`pull_request_target` run writes a `CLAAssistant` check to the head sha. So any
comment on a pull request killed the run that owned a REQUIRED context and put
nothing in its place, leaving `cancelled` -- which branch protection reads as
failure -- with no subsequent event able to clear it.

Measured before the fix: **13 of 60 open pull requests** were blocked this way,
several of whose only offence was having been commented on.

The rule pinned here is narrow on purpose, because a blanket ban would be wrong.
Cancelling is refused only when ALL of:

  1. the workflow triggers on an event that writes a check to a PR head
     (`pull_request`, `pull_request_target`, `merge_group`), AND
  2. it also triggers on an event that does not (`issue_comment`, `schedule`,
     `workflow_dispatch`, `issues`), AND
  3. `cancel-in-progress` is true, AND
  4. the group keys on the PR/issue NUMBER, which is identical across both
     event types, AND
  5. the group does not discriminate by `github.event_name`.

Point 4 is what makes this precise rather than a blanket ban, and the first
draft of this guard got it wrong: it flagged nine workflows that key on
`github.ref`. Those are safe, because `github.ref` differs by event
(`refs/pull/N/merge` for a pull_request, a branch ref for a dispatch), so runs
from the two classes land in different groups and never cancel each other. Only
a PR-identity key collides across event types.

Point 5 is the escape hatch: a group including the event name confines
cancellation to runs of the same kind, so a workflow that genuinely wants both
behaviours can have them.
"""
import re
import sys
import pathlib
import yaml

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / "workflows"
WRITES_PR_CHECK = {"pull_request", "pull_request_target", "merge_group"}
NO_PR_CHECK = {"issue_comment", "issues", "schedule", "workflow_dispatch", "workflow_run"}

problems: list[str] = []


def truthy(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def check(path: pathlib.Path) -> None:
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception as exc:
        problems.append(f"{path.name} does not parse: {exc}")
        return
    if not isinstance(doc, dict):
        return
    conc = doc.get("concurrency")
    if not isinstance(conc, dict) or not truthy(conc.get("cancel-in-progress")):
        return

    on = doc.get(True) or doc.get("on") or {}
    events = set(on) if isinstance(on, dict) else {on} if isinstance(on, str) else set(on or [])
    writes = events & WRITES_PR_CHECK
    blind = events & NO_PR_CHECK
    if not (writes and blind):
        return

    group = str(conc.get("group", ""))
    if "event_name" in group:
        return
    # Only a PR-identity key collides across event types. A `github.ref` key
    # does not -- see the note in the module docstring.
    if not re.search(r"(pull_request|issue)\.number", group):
        return

    problems.append(
        f"{path.name} cancels in-progress runs across trigger types that report differently.\n"
        f"         writes a PR check: {sorted(writes)}; does not: {sorted(blind)}\n"
        f"         group: {group}\n"
        f"         An event from the second set will cancel a run from the first and write no\n"
        f"         replacement, leaving a required context `cancelled` forever. Either set\n"
        f"         cancel-in-progress: false, or add ${{{{ github.event_name }}}} to the group."
    )


def main() -> None:
    files = sorted(WORKFLOWS.glob("*.y*ml"))
    for f in files:
        check(f)
    if problems:
        for p in problems:
            print(f"REFUSE: {p}", file=sys.stderr)
        sys.exit(1)
    print(f"ok: no workflow among {len(files)} cancels a check that the cancelling event cannot rewrite")


if __name__ == "__main__":
    main()
