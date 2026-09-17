#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Plan which open PRs can be landed together as one certified stack.

Certification is the expensive resource here: a PR touching `PERF_PATHS`
re-opens all 11 gates and owes a ~5h GPU campaign. Landing such PRs one at a
time pays that cost once per PR. Landing a GROUP of them on one branch pays it
once for the group — which is only sound if the group is coherent, because the
campaign certifies the composed tree, not the individual diffs.

The economics have a second half that matters more at this backlog size: a PR
touching NO `PERF_PATHS` file owes no campaign at all. Those can be stacked and
landed immediately, and separating them from the ones that owe a campaign is
the single biggest lever on a 98-PR backlog.

This is an operator tool, not a gate: it needs the network, so it cannot run in
CI. It reports; a human decides.

Definitions, kept honest:

  free   — touches nothing in PERF_PATHS. No campaign. Stackable on sight.
  owes   — touches PERF_PATHS. One campaign per STACK, not per PR.
  blocked— DIRTY (conflicts with base) or a draft. Not stackable until resolved;
           listed so they are not silently dropped from the count.

Overlap is reported per candidate stack because two PRs editing the same file
will conflict when stacked, and finding that out at `git cherry-pick` time is
worse than finding it out here.
"""
import json
import subprocess
import sys
import collections

REPO = "Avarok-Cybersecurity/atlas"
PERF_PATHS = (
    "crates", "kernels", "Cargo.toml", "Cargo.lock",
    "vendor", "jinja-templates", "rust-toolchain.toml", "3rdparty_patches",
)


def gh(*args: str) -> str:
    for _ in range(3):
        p = subprocess.run(["gh", *args], capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
    return ""


def touches_perf(paths: list[str]) -> list[str]:
    hit = set()
    for f in paths:
        top = f.split("/", 1)[0]
        if top in PERF_PATHS:
            hit.add(top)
    return sorted(hit)


def main() -> None:
    raw = gh("pr", "list", "--repo", REPO, "--state", "open", "--limit", "100",
             "--json", "number,title,mergeStateStatus,isDraft,author")
    if not raw:
        print("could not list PRs", file=sys.stderr)
        sys.exit(1)
    prs = json.loads(raw)

    rows = []
    for pr in prs:
        n = pr["number"]
        files = gh("api", "--paginate", f"repos/{REPO}/pulls/{n}/files", "--jq", ".[].filename")
        paths = [f for f in files.split("\n") if f]
        rows.append({
            "n": n,
            "title": pr["title"],
            "state": pr["mergeStateStatus"],
            "draft": pr["isDraft"],
            "author": pr["author"]["login"],
            "paths": paths,
            "perf": touches_perf(paths),
        })

    free = [r for r in rows if not r["perf"] and not r["draft"] and r["state"] != "DIRTY"]
    owes = [r for r in rows if r["perf"] and not r["draft"] and r["state"] != "DIRTY"]
    stuck = [r for r in rows if r["draft"] or r["state"] == "DIRTY"]

    print(f"{len(rows)} open PRs: {len(free)} free, {len(owes)} owe a campaign, {len(stuck)} blocked/draft\n")

    print("FREE — no PERF_PATHS file, no campaign, stackable now")
    for r in sorted(free, key=lambda r: r["n"]):
        print(f"  #{r['n']:<5} {r['state']:<9} {len(r['paths']):>3}f  {r['author']:<16} {r['title'][:58]}")

    print("\nOWES A CAMPAIGN — group these so one campaign covers the group")
    by_area = collections.defaultdict(list)
    for r in owes:
        by_area[",".join(r["perf"])].append(r)
    for area, group in sorted(by_area.items()):
        print(f"  [{area}]")
        for r in sorted(group, key=lambda r: r["n"]):
            print(f"    #{r['n']:<5} {r['state']:<9} {len(r['paths']):>3}f  {r['author']:<16} {r['title'][:52]}")

    # file collisions inside each candidate group
    print("\nFILE COLLISIONS (would conflict if stacked)")
    for label, group in [("free", free)] + [(f"owes[{a}]", g) for a, g in sorted(by_area.items())]:
        owner = collections.defaultdict(list)
        for r in group:
            for f in r["paths"]:
                owner[f].append(r["n"])
        clashes = {f: ns for f, ns in owner.items() if len(ns) > 1}
        if clashes:
            print(f"  {label}: {len(clashes)} shared file(s)")
            for f, ns in sorted(clashes.items())[:6]:
                print(f"    {f}  <- {ns}")
        else:
            print(f"  {label}: none — this group stacks cleanly")


if __name__ == "__main__":
    main()
