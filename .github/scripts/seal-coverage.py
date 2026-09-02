#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Do these sealers own every path in the diff?

Usage: seal-coverage.py <codeowners-file> <sealer>[,<sealer>...] < changed-paths

Exit 0 and print `covered`, or exit 1 and print the paths nobody sealed.

── Why this fails CLOSED, when the Rust one fails open ─────────────────────

`crates/atlas-plugin/src/gate/codeowners.rs` implements the same last-match-wins
matching and deliberately fails OPEN: an unsupported pattern there means somebody
is not @-mentioned on a PR they own, which is a missed notification and nothing
more. Its own doc says so — "it does not let a change through a gate".

Here the same table decides whether a seal counts. If an unsupported pattern
silently matched nothing, a sealer would appear to own a path they do not, and
the seal would be wider than the reviewer intended. So an unrecognised pattern is
an error, loudly, rather than a quiet widening. Same table, opposite failure
direction, because the consequence is opposite.

Supported, matching the Rust subset and the real file: a leading `/`, a trailing
`/` (directory prefix), and `*` inside one segment. `**`, `?`, character classes
and escapes are refused.
"""
import re
import sys

UNSUPPORTED = ("**", "?", "[", "]", "\\", "!")


def parse(text):
    """[(pattern, {owners})] in file order. Last match wins, as GitHub does."""
    rules = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        pattern, owners = parts[0], {o.lstrip("@").lower() for o in parts[1:]}
        for bad in UNSUPPORTED:
            if bad in pattern:
                raise SystemExit(
                    f"CODEOWNERS:{lineno}: pattern {pattern!r} uses {bad!r}, which this "
                    f"checker does not implement. Refusing rather than guessing: a "
                    f"pattern we mis-match would make a seal cover paths its author "
                    f"never reviewed."
                )
        rules.append((pattern, owners))
    return rules


def matches(pattern, path):
    if pattern == "*":
        return True
    p = pattern[1:] if pattern.startswith("/") else pattern
    if p.endswith("/"):                       # directory prefix
        return path.startswith(p) or path == p.rstrip("/")
    # `*` matches within a segment only, so it must not cross a `/`.
    rx = "^" + "".join("[^/]*" if c == "*" else re.escape(c) for c in p) + "(/.*)?$"
    return re.match(rx, path) is not None


def owners_of(rules, path):
    """Last matching rule wins — GitHub's rule, and the Rust module's."""
    found = set()
    for pattern, owners in rules:
        if matches(pattern, path):
            found = owners
    return found


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    with open(sys.argv[1], encoding="utf-8") as fh:
        rules = parse(fh.read())
    sealers = {s.strip().lstrip("@").lower() for s in sys.argv[2].split(",") if s.strip()}
    paths = [l.strip() for l in sys.stdin if l.strip()]

    uncovered = [p for p in paths if not (owners_of(rules, p) & sealers)]
    if uncovered:
        shown = uncovered[:10]
        tail = f"\n  ... and {len(uncovered) - 10} more" if len(uncovered) > 10 else ""
        print("uncovered:\n  " + "\n  ".join(shown) + tail)
        return 1
    print("covered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
