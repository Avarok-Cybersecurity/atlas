#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Check the SPDX header on every file `.licenserc.yaml` says needs one.

This replaces `apache/skywalking-eyes`, which is a composite action that
internally uses an UNPINNED `actions/setup-go@v5`. This repository sets
`sha_pinning_required`, and that policy applies transitively, so the action
cannot run here however carefully we pin the action itself. Checking a header
is a grep; it does not justify a third-party action and a Go toolchain in CI.

`.licenserc.yaml` stays the single source of the paths and the ignores — this
reads them rather than restating them, so adding a path there still works.
"""
import fnmatch
import pathlib
import re
import sys

import yaml


def braces(pattern: str) -> list[str]:
    """Expand one `{a,b}` group; the config uses at most one per pattern."""
    m = re.search(r"\{([^}]*)\}", pattern)
    if not m:
        return [pattern]
    return [
        pattern[: m.start()] + alt + pattern[m.end() :] for alt in m.group(1).split(",")
    ]


def main() -> int:
    cfg = yaml.safe_load(pathlib.Path(".licenserc.yaml").read_text())["header"]
    token = cfg["license"]["content"].strip()
    includes = [p for pat in cfg["paths"] for p in braces(pat)]
    ignores = [p for pat in cfg.get("paths-ignore", []) for p in braces(pat)]

    missing = []
    for path in pathlib.Path(".").rglob("*"):
        if not path.is_file():
            continue
        rel = path.as_posix()
        if not any(fnmatch.fnmatch(rel, pat) for pat in includes):
            continue
        if any(fnmatch.fnmatch(rel, pat) for pat in ignores):
            continue
        # The header must be at the top: a match buried in the body is a
        # mention, not a declaration.
        head = "".join(path.open(encoding="utf-8", errors="replace").readlines()[:5])
        if token not in head:
            missing.append(rel)

    if missing:
        print(f"::error::{len(missing)} file(s) missing '{token}' in the first 5 lines")
        for m in sorted(missing):
            print(f"  {m}")
        return 1
    print(f"SPDX headers OK ({token})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
