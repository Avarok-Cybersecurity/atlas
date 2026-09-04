#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""For every open PR: is it already on `main`?

The single highest-yield question in a large backlog, and the cheapest. Of the
first fifteen PRs triaged by hand during this pass, **ten were already merged or
superseded** -- their content was in the tree and only the PR was still open.
Landing those individually would have spent ten certification campaigns (~5 GPU
hours each) re-certifying code that was already there.

Method, and why it is trustworthy: cherry-pick the PR's own commits onto current
`main` in a throwaway worktree. Git reports `empty commit set passed` when the
branch is an ancestor, and "The previous cherry-pick is now empty" when the
content is present under different shas. Both mean the same thing for our
purposes -- there is nothing left to land.

VERDICTS
  contained   already on main (ancestor, or content-identical). Close it.
  applies     real content not on main. A stacking candidate.
  conflicts   needs a rebase before it can be judged at all.
  no-base     shares no history with main -- the #452 orphan class.

Two cautions learned the hard way while doing this by hand:

  * The clone must be DEEP. A truncated `main` makes `git merge-base` return
    nothing, which reads as "orphan" and is simply false. This script refuses to
    run against a shallow or truncated clone rather than reporting a fiction.
  * `contained` is a claim about CONTENT, not intent. Confirm the specific
    artifact (a file, a constant, a symbol) before closing someone's PR, and say
    what you confirmed. Every close this pass named its evidence.
"""
import json
import subprocess
import sys
import tempfile
import pathlib

REPO = "Avarok-Cybersecurity/atlas"


def run(*a, **kw):
    return subprocess.run(a, capture_output=True, text=True, **kw)


def gh(*a):
    p = run("gh", *a)
    return p.stdout if p.returncode == 0 else ""


def main() -> None:
    if pathlib.Path(".git/shallow").exists():
        print("REFUSE: shallow clone -- merge-base would lie. Run `git fetch --unshallow`.", file=sys.stderr)
        sys.exit(2)
    depth = int(run("git", "rev-list", "--count", "origin/main").stdout.strip() or 0)
    if depth < 100:
        print(f"REFUSE: origin/main is only {depth} commits deep. A truncated history makes "
              f"`git merge-base` return nothing, which reads as 'orphan' and is false. "
              f"Run `git fetch --deepen=2000 origin main`.", file=sys.stderr)
        sys.exit(2)

    prs = json.loads(gh("pr", "list", "--repo", REPO, "--state", "open", "--limit", "100",
                        "--json", "number,title,isDraft,author") or "[]")
    verdicts = {}
    with tempfile.TemporaryDirectory() as td:
        wt = pathlib.Path(td) / "wt"
        run("git", "worktree", "add", "--detach", str(wt), "origin/main")
        try:
            for pr in prs:
                n = pr["number"]
                if run("git", "fetch", "-q", "origin", f"pull/{n}/head:pr-{n}", "-f").returncode != 0:
                    verdicts[n] = ("unfetchable", pr); continue
                base = run("git", "merge-base", "origin/main", f"pr-{n}").stdout.strip()
                if not base:
                    verdicts[n] = ("no-base", pr); continue
                run("git", "-C", str(wt), "reset", "-q", "--hard", "origin/main")
                revs = run("git", "rev-list", "--reverse", f"{base}..pr-{n}").stdout.split()
                if not revs:
                    verdicts[n] = ("contained", pr); continue
                # COMMIT BY COMMIT, not as a range. `git cherry-pick A..B` stops at
                # the first commit that turns out empty and reports "is now empty" --
                # which says nothing about the commits after it. Reading that as a
                # verdict for the whole PR marked #702 `contained` when it has four
                # empty commits followed by one that conflicts. A PR is contained
                # only if EVERY one of its commits is already in the tree.
                empty = applied = 0
                conflicted = False
                for rev in revs:
                    cp = run("git", "-C", str(wt), "cherry-pick", rev)
                    blob = cp.stdout + cp.stderr
                    if "is now empty" in blob or "empty commit set" in blob:
                        empty += 1
                        run("git", "-C", str(wt), "cherry-pick", "--skip")
                    elif cp.returncode == 0:
                        applied += 1
                    else:
                        conflicted = True
                        run("git", "-C", str(wt), "cherry-pick", "--abort")
                        break
                if conflicted:
                    verdicts[n] = ("conflicts", pr)
                elif applied == 0:
                    verdicts[n] = ("contained", pr)
                else:
                    verdicts[n] = ("applies", pr)
                run("git", "-C", str(wt), "cherry-pick", "--quit")
        finally:
            run("git", "worktree", "remove", "--force", str(wt))

    order = ["contained", "applies", "conflicts", "no-base", "unfetchable"]
    counts = {k: 0 for k in order}
    for v, _ in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    print("  " + "  ".join(f"{k}={counts.get(k,0)}" for k in order) + f"   (of {len(verdicts)})\n")
    for want in order:
        rows = [(n, p) for n, (v, p) in sorted(verdicts.items()) if v == want]
        if not rows:
            continue
        print(f"{want.upper()}")
        for n, p in rows:
            d = " [draft]" if p["isDraft"] else ""
            print(f"  #{n:<5} {p['author']['login']:<16}{d} {p['title'][:62]}")
        print()


if __name__ == "__main__":
    main()
