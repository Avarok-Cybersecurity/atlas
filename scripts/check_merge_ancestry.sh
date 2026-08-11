#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Merge-ancestry guard: refuse PRs whose head shares NO history with the base.
#
# Why this exists (PR #452, 2026-08-10): a branch whose first commit was a
# fresh repository ROOT — a full tree snapshot with no parent — showed up as
# `mergeable: MERGEABLE`, because GitHub computes mergeability over TREES,
# not ancestry. The snapshot predated five merged PRs, so squash-merging it
# would have applied a diff that silently reverted all of them. Nothing in
# the PR UI said so. `git merge origin/main` refusing with "unrelated
# histories" was the only tell. This script makes that refusal a deliberate,
# per-PR, loud check instead of a side effect someone has to diagnose.
#
# Usage:
#   check_merge_ancestry.sh <base_sha> <head_sha> [base_ref]
#
# base_ref is only used to render actionable remedy commands (defaults to
# "main"). Both commits must already be present in the local object store —
# this script performs NO network I/O; fetching is the caller's job.
#
# Exit codes:
#   0  PASS (possibly with ::warning annotations)
#   1  FAIL — no merge base: head is unrelated to base (the #452 class)
#   2  usage / infrastructure error (missing args, commit not present)
#
# Env:
#   ANCESTRY_HISTORY_CAP  How many base-side commits the revert-signature
#                         scan walks. Default 2000: covers years of history
#                         here (~a few thousand commits total) while bounding
#                         worst-case runtime to a few seconds of local tree
#                         diffs. Purely a scan bound, never a correctness gate.
set -euo pipefail

BASE_SHA="${1:-}"
HEAD_SHA="${2:-}"
BASE_REF="${3:-main}"
HISTORY_CAP="${ANCESTRY_HISTORY_CAP:-2000}"

if [[ -z "$BASE_SHA" || -z "$HEAD_SHA" ]]; then
  echo "::error::usage: check_merge_ancestry.sh <base_sha> <head_sha> [base_ref]" >&2
  exit 2
fi

for sha in "$BASE_SHA" "$HEAD_SHA"; do
  if ! git cat-file -e "${sha}^{commit}" 2>/dev/null; then
    echo "::error::commit $sha is not present locally — the caller must fetch full history (fetch-depth: 0) before running this check" >&2
    exit 2
  fi
done

# ── Case 1: no merge base — the #452 class. Hard FAIL. ─────────────────────
# `git merge-base` exits non-zero and prints nothing when the two commits
# share no ancestor. This is only trustworthy against a FULL commit graph:
# a shallow clone cannot distinguish "no common ancestor" from "not deep
# enough", which is why the workflow fetches with fetch-depth: 0.
merge_base="$(git merge-base "$BASE_SHA" "$HEAD_SHA" || true)"

if [[ -z "$merge_base" ]]; then
  short_base="${BASE_SHA:0:12}"
  short_head="${HEAD_SHA:0:12}"
  echo "::error::NO MERGE BASE between base $short_base and head $short_head — this branch shares no history with '$BASE_REF'. Merging it would silently revert work already on '$BASE_REF' (see PR #452). Rebuild the branch on the current '$BASE_REF' and re-apply the patch."
  cat <<EOF

  ✗ NO MERGE BASE: head $HEAD_SHA
                   base $BASE_SHA ('$BASE_REF')

  git merge-base found no common ancestor: this branch's history starts at
  its own root commit (an orphan tree snapshot) instead of branching from
  '$BASE_REF'. GitHub still reports the PR as MERGEABLE because it computes
  mergeability over trees, not ancestry — but merging (squash included)
  would apply the snapshot's stale tree against '$BASE_REF' and silently
  revert every commit that landed after the snapshot was taken. That is
  exactly what PR #452 would have done to five merged PRs.

  Remedy (what fixed #452) — rebuild on the current base, re-apply the patch:

    git fetch origin $BASE_REF
    git switch -c my-fix-rebuilt origin/$BASE_REF
    # re-apply each intended commit as a patch (cherry-pick cannot cross
    # unrelated histories cleanly; the root commit IS the whole tree):
    git diff ${short_head}^ $short_head | git apply -3
    git add -A && git commit
    git push --force-with-lease origin HEAD:<your-branch>

  Do NOT "fix" this with \`git merge --allow-unrelated-histories\` — merging
  the stale snapshot is precisely the destruction this check exists to stop.
EOF
  exit 1
fi

# ── Case 2a: merely behind — WARN only, never FAIL. ────────────────────────
# Nearly every open PR is some commits behind main; with a real merge base a
# 3-way (or squash) merge preserves base-side work, and branch protection's
# `strict: true` already forces an update before merge. Failing here would
# fire on the whole queue and teach people to ignore the check.
behind="$(git rev-list --count "$merge_base..$BASE_SHA")"
if [[ "$behind" -gt 0 ]]; then
  echo "::warning::PR is $behind commit(s) behind its base '$BASE_REF' (merge base ${merge_base:0:12}). Not a failure — branch protection (strict: true) requires an update before merge."
fi

# ── Case 2b: revert signature — WARN loudly, listing the files. ────────────
# The dangerous cousin of the orphan: a branch WITH ancestry whose diff
# rewinds files to a pre-merge-base state (e.g. an old tree committed on top
# of a current base — mergeable, conflict-free, and it reverts silently).
# Detection: a file the PR modifies whose head-side blob is byte-identical
# to a version that existed in the base's history BEFORE the merge base.
# WARN rather than FAIL because deliberate revert PRs are legitimate; this
# exists so a reviewer sees "this PR rewinds N files to old contents" in the
# checks output instead of nothing at all. Blob comparison is by OID, so a
# blobless (filter=blob:none) clone suffices — no file contents downloaded.
reverted="$(
  {
    # Wanted set: paths the PR modifies, with their head-side blob OIDs.
    git diff --raw --no-abbrev --no-renames "$merge_base" "$HEAD_SHA"
    echo "== HISTORY =="
    # Base-side history: every (path, blob) pair that ever appeared in the
    # last $HISTORY_CAP commits reachable from the merge base, both sides
    # of each change so pre-images count too.
    git log --format='%H' --raw --no-abbrev --no-renames \
      --max-count="$HISTORY_CAP" "$merge_base"
  } | awk '
    /^== HISTORY ==$/ { in_history = 1; next }
    /^:/ {
      # :oldmode newmode oldoid newoid status<TAB>path
      split($0, halves, "\t"); path = halves[2]
      if (!in_history) {
        if ($5 == "M") want[path] = $4          # head-side blob OID
      } else {
        if ($3 !~ /^0+$/) seen[path "\t" $3] = 1
        if ($4 !~ /^0+$/) seen[path "\t" $4] = 1
      }
    }
    END {
      for (path in want)
        if (seen[path "\t" want[path]]) print path
    }
  ' | sort
)"

if [[ -n "$reverted" ]]; then
  count="$(wc -l <<<"$reverted")"
  echo "::warning::REVERT SIGNATURE: this PR rewinds $count file(s) to contents that predate its merge base ${merge_base:0:12} — if that is not a deliberate revert, the branch was built from a stale snapshot of '$BASE_REF'."
  echo ""
  echo "  Files reverted to pre-merge-base contents (first 20):"
  head -n 20 <<<"$reverted" | sed 's/^/    /'
fi

echo ""
echo "✓ merge-ancestry: base ${BASE_SHA:0:12} and head ${HEAD_SHA:0:12} share merge base ${merge_base:0:12}."
