#!/usr/bin/env bash
# Decide which per-PR ledger files fall out of the bounded window.
#
# `governance/` holds one `pr-<n>.jsonl` per classified PR and has never had a
# retention rule, so it grows one file per PR forever. This turns it into a
# bounded FIFO queue: the newest `cap` PRs keep their own file, older ones are
# evicted and recorded in `governance/archive.csv` as (pr, hash, merged_at) —
# the hash being the commit that last carried the file, so
# `git show <hash>:governance/pr-<n>.jsonl` recovers it verbatim. Nothing is
# destroyed; the git tree is still the store, the CSV is the index into it.
#
# This lives outside the workflow for the same reason `harvest-triage.sh` does:
# the decision is where the bugs are, and a decision in a YAML block is a
# decision nobody can run.
#
# ── The rule that matters ───────────────────────────────────────────────────
#
# NEVER evict a PR that is still open. `gate::required::intent_source()` looks
# its ledger up by bare path with no fallback, so an evicted file does not
# raise — it silently reports `NotRecorded` and the PR quietly drops to
# path-only gating. That is a wrong answer with no error, which is the worst
# kind. The cap yields to this rule: if too few closed candidates exist, fewer
# files are evicted and the window is allowed to run over.
#
# Usage: governance-evict.sh <cap> <open-prs>   [ledger PR numbers on stdin]
#   <cap>       maximum per-PR files to keep as individual files
#   <open-prs>  space-separated PR numbers that must not be evicted (may be "")
#
# Prints the PR numbers to evict, one per line, oldest first. Nothing else.
# Exit 0 always on well-formed input (an empty result is a normal outcome);
# exit 2 on malformed input.
set -euo pipefail

cap="${1?cap required}"
open_prs="${2?open PR list required (may be empty)}"

case "$cap" in
  ''|*[!0-9]*) echo "cap must be a non-negative integer, got '$cap'" >&2; exit 2 ;;
esac
for pr in $open_prs; do
  case "$pr" in
    *[!0-9]*) echo "open PR list must be numbers, got '$pr'" >&2; exit 2 ;;
  esac
done

# Read the candidate ledger PR numbers from stdin, rejecting anything that is
# not a plain number rather than silently skipping it: a malformed line means
# the caller's globbing changed, and guessing past that is how a file gets
# deleted for the wrong reason.
candidates=()
while IFS= read -r line; do
  [ -n "$line" ] || continue
  case "$line" in
    *[!0-9]*) echo "ledger PR numbers must be digits, got '$line'" >&2; exit 2 ;;
  esac
  candidates+=("$line")
done

total=${#candidates[@]}
outside=$(( total - cap ))
[ "$outside" -gt 0 ] || exit 0

# FIFO by PR number ascending. PR numbers are monotonic, so this is a stable
# "oldest first" without consulting timestamps that a re-harvest could move.
mapfile -t sorted < <(printf '%s\n' "${candidates[@]}" | sort -n)

is_open() {
  for o in $open_prs; do
    [ "$o" = "$1" ] && return 0
  done
  return 1
}

# ★ Only the files OUTSIDE the newest-`cap` window are candidates, and an open
# one among them is skipped rather than substituted for.
#
# The tempting alternative — walk from the oldest and keep evicting until the
# count reaches `cap` — holds the bound exactly, and is wrong. With old PRs
# still open it reaches the bound by evicting the NEWEST ledgers instead, which
# is the recent intent data the trajectory view is built on. Better to overshoot
# the window than to throw away the newest records to defend it.
#
# The overshoot is bounded by how many old PRs are open at once and disappears
# on its own as they close, so it self-corrects without any extra machinery.
for (( i = 0; i < outside; i++ )); do
  pr="${sorted[$i]}"
  is_open "$pr" && continue
  echo "$pr"
done
