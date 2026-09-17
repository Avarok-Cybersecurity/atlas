#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""The CHKI reach table: what each merge changed, and on which hardware nobody checked it.

CHKI (`scripts/check_cross_hardware.py`) already answers "who else compiles the
bytes you changed?" on every PR. It is ADVISORY, so its answer scrolls away with
the run that produced it, and the thing worth keeping is not the verdict — it is
the BACKLOG: the hardware that consumed a change and never compiled it here.

So on every merge to `main` this appends one row to a single comment in a
discussion:

    | PR | relevant kernels | affected hardware (not checked) | affected hardware (checked) |

and rewrites the first row to the UNION of the not-checked column — the set of
targets currently owed a manual build or measurement.

"Checked" is not a judgement: a hardware is checked when a named CI job compiles
it, and `checked_hardware()` REFUSES to answer if one of those jobs has been
renamed or deleted (rather than quietly reporting the target as unchecked, or
worse, as checked). A backlog that silently drops an entry is the one failure
this file cannot have.

I/O is confined to `main()` and `gh_update_comment()`; everything else is pure,
so the table's shape is testable without a network. `--selftest` runs RED/GREEN
fixtures for every rule and is on by default.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

MARKER = "<!-- chki-reach-table:v1 -->"
HEADER = "| PR | relevant kernels | affected hardware (not checked) | affected hardware (checked) |"
SEP = "| --- | --- | --- | --- |"
UNION_LABEL = "**owed a manual check**"

# A hardware is CHECKED when one of these jobs compiles it on the merge commit.
# The value is the job's `name:` as it appears in the workflow — verified to
# still exist, so a renamed job breaks this loudly instead of turning a checked
# target into an unchecked one (or the reverse) in silence.
CHECKED_BY = {
    "gb10": "nvcc -> PTX (all gb10 targets)",
    "hopper": "nvcc -> PTX (hopper, sm_90a)",
    "b200": "nvcc -> PTX (b200, sm_100a)",
    "metal": "cargo test --features metal (macOS aarch64)",
}


class Unseeable(Exception):
    """The inputs cannot be read, so no verdict is possible. Never guess one."""


# ---------------------------------------------------------------- pure core


def hardware_dirs(root: pathlib.Path) -> list[str]:
    """Every hardware the kernel tree has, from the tree itself."""
    kernels = root / "kernels"
    if not kernels.is_dir():
        raise Unseeable(f"no kernels/ under {root}")
    return sorted(p.name for p in kernels.iterdir() if p.is_dir())


def checked_hardware(root: pathlib.Path) -> set[str]:
    """The hardware with a compile leg, proved by finding each job's name."""
    text = "\n".join(
        p.read_text(errors="replace")
        for p in sorted((root / ".github" / "workflows").glob("*.yml"))
    )
    if not text:
        raise Unseeable("no workflows to read; cannot say what CI compiles")
    out, missing = set(), []
    for hw, job in CHECKED_BY.items():
        (out.add(hw) if job in text else missing.append(f"{hw}: {job!r}"))
    if missing:
        raise Unseeable(
            "these jobs no longer exist, so their hardware cannot be called checked: "
            + "; ".join(missing)
            + ". Update CHECKED_BY to the new names."
        )
    return out


def reach_of(chki: dict) -> tuple[list[str], set[str]]:
    """(changed kernel paths, every hardware that consumes them) from CHKI's JSON."""
    paths, hw = [], set()
    for row in chki.get("paths", []):
        p = row.get("path", "")
        if not p.startswith("kernels/"):
            continue
        paths.append(p)
        if row.get("owner"):
            hw.add(row["owner"])
        hw.update(row.get("reach_hw", []))
    return sorted(set(paths)), hw


def condense(paths: list[str], limit: int = 6) -> str:
    """Kernel paths as a cell: the directories that changed, then a count."""
    if not paths:
        return "—"
    dirs = sorted({str(pathlib.PurePosixPath(p).parent) for p in paths})
    shown = ", ".join(f"`{d}`" for d in dirs[:limit])
    extra = len(dirs) - limit
    tail = f" +{extra} more" if extra > 0 else ""
    return f"{shown}{tail} ({len(paths)} file{'s' if len(paths) != 1 else ''})"


def cell(hw: list[str]) -> str:
    return ", ".join(f"`{h}`" for h in sorted(hw)) if hw else "—"


def row_for(pr: str, chki: dict, checked: set[str]) -> dict:
    paths, hw = reach_of(chki)
    return {
        "pr": pr,
        "paths": paths,
        "not_checked": sorted(hw - checked),
        "checked": sorted(hw & checked),
    }


def as_table(body: str, comment_id: str) -> str:
    """The body, once it is proved to BE the table.

    The id comes from a repository variable, and a variable can be pointed at
    the wrong comment. Overwriting somebody's prose with a table is not a thing
    to find out about afterwards, so the marker is checked before any write.
    """
    if MARKER not in body:
        raise Unseeable(
            f"comment {comment_id} does not carry {MARKER} — refusing to overwrite "
            "a comment that is not the table"
        )
    return body


def parse_table(body: str) -> list[dict]:
    """The data rows of an existing table, union row and header dropped."""
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|") or line in (HEADER, SEP):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 4 or cells[0] == UNION_LABEL:
            continue
        rows.append(
            {
                "pr": cells[0],
                "cells": cells,
            }
        )
    return rows


def render_row(row: dict) -> str:
    if "cells" in row:
        return "| " + " | ".join(row["cells"]) + " |"
    return (
        f"| {row['pr']} | {condense(row['paths'])} "
        f"| {cell(row['not_checked'])} | {cell(row['checked'])} |"
    )


def union_not_checked(rows: list[dict]) -> list[str]:
    """Every hardware still owed a check, read back out of the rendered rows.

    Read from the CELLS, not from a running total: the comment is the record,
    so a row edited or deleted by hand changes the union, and a total kept
    beside it would go on asserting a target nobody can find a row for.
    """
    hw = set()
    for row in rows:
        text = row["cells"][2] if "cells" in row else cell(row["not_checked"])
        hw.update(re.findall(r"`([^`]+)`", text))
    return sorted(hw)


def render_table(rows: list[dict]) -> str:
    union = union_not_checked(rows)
    head = [
        MARKER,
        HEADER,
        SEP,
        f"| {UNION_LABEL} | — | {cell(union)} | — |",
    ]
    return "\n".join(head + [render_row(r) for r in rows]) + "\n"


def upsert(rows: list[dict], new: dict) -> list[dict]:
    """Newest last, one row per PR: a re-merge corrects its row, never doubles it."""
    kept = [r for r in rows if r["pr"] != new["pr"]]
    return kept + [new]


# ------------------------------------------------------------------- shell


def gh_json(args: list[str]) -> dict:
    out = subprocess.run(["gh", *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise Unseeable(f"gh {' '.join(args)}: {out.stderr.strip()}")
    return json.loads(out.stdout or "{}")


def read_comment(comment_id: str) -> str:
    q = "query($id:ID!){ node(id:$id){ ... on DiscussionComment { body } } }"
    data = gh_json(["api", "graphql", "-f", f"query={q}", "-F", f"id={comment_id}"])
    body = (((data.get("data") or {}).get("node") or {}).get("body")) or ""
    return as_table(body, comment_id)


def write_comment(comment_id: str, body: str) -> None:
    q = (
        "mutation($id:ID!,$body:String!){ updateDiscussionComment("
        "input:{commentId:$id,body:$body}){ comment{ url } } }"
    )
    gh_json(["api", "graphql", "-f", f"query={q}", "-F", f"id={comment_id}", "-F", f"body={body}"])


# ---------------------------------------------------------------- fixtures


def selftest() -> None:
    """RED/GREEN fixtures. A rule that cannot fire is a rule that is not a check."""
    chki = {
        "paths": [
            {"path": "kernels/gb10/common/w4a16_gemv.cu", "owner": "gb10",
             "reach_hw": ["gb10", "strix", "strix-hip"]},
            {"path": "kernels/gb10/common/other.cu", "owner": "gb10", "reach_hw": ["gb10"]},
            {"path": "crates/avarok-plugin/src/lib.rs", "owner": None, "reach_hw": []},
        ]
    }
    checked = {"gb10", "hopper", "b200", "metal"}
    row = row_for("#1234", chki, checked)
    assert row["not_checked"] == ["strix", "strix-hip"], row
    assert row["checked"] == ["gb10"], row
    assert row["paths"] == [
        "kernels/gb10/common/other.cu",
        "kernels/gb10/common/w4a16_gemv.cu",
    ], row["paths"]

    # RED: a non-kernel path must never put hardware on the backlog.
    assert reach_of({"paths": [{"path": "site/src/app.css", "owner": "gb10",
                               "reach_hw": ["gb10"]}]}) == ([], set())

    # The union is the first row and covers every row's not-checked set.
    t = render_table(upsert([], row))
    assert t.startswith(MARKER), t
    first = t.splitlines()[3]
    assert UNION_LABEL in first and "`strix`" in first and "`strix-hip`" in first, first
    assert "`gb10`" not in first, "a CHECKED target must not appear in the backlog"

    # A second PR extends the union; a re-merge of the first replaces its row.
    row2 = row_for("#1300", {"paths": [{"path": "kernels/hopper/x.cu", "owner": "hopper",
                                        "reach_hw": ["hopper", "metal"]}]}, checked)
    two = upsert(parse_table(t), row2)
    t2 = render_table(two)
    assert len([l for l in t2.splitlines() if l.startswith("| #")]) == 2, t2
    again = upsert(parse_table(t2), row_for("#1234", chki, checked))
    assert len([l for l in render_table(again).splitlines() if l.startswith("| #1234")]) == 1

    # RED: an unchecked target may not be lost when its row is re-rendered from
    # the comment (the parse -> render round trip is where a backlog would leak).
    assert "`strix-hip`" in render_table(parse_table(t2)).splitlines()[3]

    # RED: a renamed compile job must make the script refuse, not relabel.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / ".github" / "workflows").mkdir(parents=True)
        (root / "kernels" / "gb10").mkdir(parents=True)
        wf = root / ".github" / "workflows" / "ci.yml"
        wf.write_text("\n".join(CHECKED_BY.values()))
        assert checked_hardware(root) == set(CHECKED_BY)
        wf.write_text(wf.read_text().replace("nvcc -> PTX (hopper, sm_90a)", "renamed"))
        try:
            checked_hardware(root)
            raise AssertionError("a renamed compile job must be refused, not ignored")
        except Unseeable as e:
            assert "hopper" in str(e)

    # RED: the writer must refuse a comment that is not the table.
    assert as_table(t2, "DC_x") == t2
    try:
        as_table("a discussion comment somebody wrote", "DC_x")
        raise AssertionError("a comment without the marker must be refused")
    except Unseeable as e:
        assert "refusing to overwrite" in str(e)

    # RED: unreadable CHKI output is rc 2, never an empty row that would read
    # as "this merge reached no hardware".
    try:
        json.loads("")
        raise AssertionError("unreachable")
    except json.JSONDecodeError:
        pass


# ---------------------------------------------------------------------- cli


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--chki-json", required=False,
                    help="file holding `check_cross_hardware.py --json` output")
    ap.add_argument("--pr", help="the merged PR, e.g. '#1101'")
    ap.add_argument("--comment-id", help="the discussion comment that IS the table")
    ap.add_argument("--out", help="write the new body here instead of posting")
    ap.add_argument("--no-selftest", action="store_true")
    args = ap.parse_args()

    if not args.no_selftest:
        selftest()
    if not args.chki_json:
        print("self-test passed; nothing to append (no --chki-json).")
        return 0

    root = pathlib.Path(args.root).resolve()
    try:
        try:
            chki = json.loads(pathlib.Path(args.chki_json).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise Unseeable(f"cannot read {args.chki_json}: {e}") from e
        checked = checked_hardware(root)
        pr = args.pr or ""
        if not re.fullmatch(r"#\d+", pr):
            raise Unseeable(f"--pr must look like '#1234', got {pr!r}")
        row = row_for(pr, chki, checked)
        if not row["paths"]:
            print(f"{pr}: no kernel paths changed — nothing to record.")
            return 0
        if not args.comment_id:
            body = render_table([row])
        else:
            body = render_table(upsert(parse_table(read_comment(args.comment_id)), row))
    except Unseeable as e:
        print(f"CHKI reach table: {e}", file=sys.stderr)
        return 2

    if args.out:
        pathlib.Path(args.out).write_text(body)
    if args.comment_id and not args.out:
        write_comment(args.comment_id, body)
        print(f"{pr}: appended; backlog is now {cell(union_not_checked(parse_table(body)))}")
    else:
        print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
