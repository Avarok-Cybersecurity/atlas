#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Every write call the pipeline SUPPRESSES must have a real probe behind it.

Twice now the same defect has shipped, one endpoint family apart:

  #843  the certificate image never uploaded. `contents:write` was absent, the
        PUT is non-fatal by design, and the preflight probed `contents:read`.
        The comment shipped a broken image and nothing said so.

  #847  `/stamp` recorded its mark and did not re-run the held lane.
        `actions:write` was absent, the re-run's stderr went to /dev/null, and
        the preflight probed `actions:read` -- under a justification that
        literally read "/stamp re-runs the held CI run", a write operation.

Both times a read probe stood where a write probe was needed, and both times the
failure was silent because the call site swallows it.

Scope, deliberately narrow, and the reason is worth stating. This does NOT
require a probe for every write the pipeline makes. A write whose failure is
*loud* -- one that runs under `set -e` with no `||` and no `if` -- announces
itself the first time it breaks; a probe would be belt-and-braces. A write whose
failure is *suppressed* is invisible until someone reads a job log, and by then
it has already shipped a broken artifact. Those are the ones pinned here.

Widening this to all writes would mean inventing noisy probes (posting and
deleting comments, creating labels) to cover failures that are already
self-announcing. That is the busywork this guard is meant to displace.
"""
import re
import sys
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
PREFLIGHT = WORKFLOWS / "certification-preflight.yml"
SCANNED = ["certification-commands.yml", "certification-bot.yml"]

# URL path fragment -> the App repository permission that write needs.
PERMISSION_FOR = [
    ("/check-runs", "checks:write"),
    ("/contents/", "contents:write"),
    ("/actions/", "actions:write"),
    ("/issues/", "issues:write"),
    ("/pulls/", "pull_requests:write"),
]

WRITE = re.compile(r"gh api\s+-X\s+(POST|PUT|PATCH|DELETE)\b")


def statements(run: str):
    """Yield logical shell statements, rejoining backslash continuations.

    A write call routinely spans five lines of `-f` flags, and the `||` that
    suppresses its failure sits on the last of them. Matching line-by-line would
    read every such call as unsuppressed -- which is the exact opposite of the
    truth, and would make this guard pass by construction.
    """
    buf = ""
    for line in run.splitlines():
        buf += line.rstrip("\n")
        if buf.rstrip().endswith("\\"):
            buf = buf.rstrip()[:-1] + " "
            continue
        yield buf
        buf = ""
    if buf:
        yield buf


def permission_for(stmt: str) -> str | None:
    for fragment, perm in PERMISSION_FOR:
        if fragment in stmt:
            return perm
    return None


def suppressed(stmt: str) -> bool:
    s = stmt.strip()
    return "||" in s or s.startswith("if ") or bool(re.match(r"^\w+=\$\(", s))


def run_blocks(path: pathlib.Path):
    doc = yaml.safe_load(path.read_text())
    for job in (doc.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            if isinstance(step.get("run"), str):
                yield step["run"]


def main() -> None:
    needed: dict[str, str] = {}
    for name in SCANNED:
        for run in run_blocks(WORKFLOWS / name):
            for stmt in statements(run):
                if not WRITE.search(stmt) or not suppressed(stmt):
                    continue
                perm = permission_for(stmt)
                if perm:
                    needed.setdefault(perm, f"{name}: {stmt.strip()[:90]}")

    probed = set(re.findall(r'probe_write\s+"([a-z_]+:write)"', PREFLIGHT.read_text()))

    missing = {p: w for p, w in needed.items() if p not in probed}
    if missing:
        for perm, where in sorted(missing.items()):
            print(
                f"REFUSE: {perm} is written by a call that swallows its own failure, and "
                f"certification-preflight.yml has no probe_write for it -- so losing that "
                f"permission would be silent.\n         {where}",
                file=sys.stderr,
            )
        sys.exit(1)

    for perm in sorted(needed):
        print(f"ok: {perm} is suppressed at its call site and probed for real by the preflight")
    if not needed:
        print("ok: no write call in the certification workflows suppresses its own failure")


if __name__ == "__main__":
    main()
