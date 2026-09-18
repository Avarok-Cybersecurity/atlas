#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Say WHY a record-bearing performance PR was thrown out of the merge queue.
#
# Two different conditions produce the identical symptom -- green at the PR
# head, red in the merge group, entry removed by github-merge-queue[bot] -- and
# they have different fixes:
#
#   A. COMPOSED BENEATH. Another queued PR sits under this one in the group.
#      Two campaigns measured apart never compose. Fix: land them one at a
#      time (the queue's max_entries_to_build is 1 for this reason).
#
#   B. MAIN MOVED. Nothing is underneath, but `main` took a perf-path commit
#      after this PR's records were measured, so the group's tree is not the
#      tree the campaign covered. Fix: freeze onto current main and re-measure
#      (scripts/queue-perf-pr.sh).
#
# The step this replaces asserted A unconditionally. On 2026-09-17 #941 was
# pure B -- eleven groups, base = main's tip, nothing beneath it, #1103's five
# `crates/` files landing 34 minutes before the first enqueue -- and the
# annotation sent its reader looking for a composition that was not there,
# while an enqueue loop spent 95 minutes and eleven merge-group CI runs. A
# diagnostic that names the wrong cause is worse than none: it is trusted.
#
# usage: diagnose-queue-bounce.sh <merge-group base sha> [<group head, default HEAD>]
#
# Never fails the job. It runs when the job is already red, and its own
# inability to look must not become a second, louder failure -- so every
# unanswerable question is printed as "could not look", which is a different
# statement from "did not happen".
set -uo pipefail
cd "$(dirname "$0")/../.."

BASE=${1:-}
HEAD_REF=${2:-HEAD}
if [ -z "$BASE" ]; then
  echo "could not look: no merge-group base sha was passed."
  echo "  usage: $0 <merge-group base sha> [<group head>]"
  exit 0
fi

# SSOT: the paths that invalidate a record are defined once, in Rust, and this
# script reads that definition rather than keeping a second copy. A second copy
# is how a guard comes to disagree with the gate it explains -- and the list has
# grown twice (vendor/jinja-templates/rust-toolchain.toml, then
# 3rdparty_patches, which closed a real bypass). PCND: if the const cannot be
# parsed there is NO fallback list, because a short list would silently answer
# "main moved on nothing".
COVERAGE=crates/avarok-plugin/src/gate/coverage.rs
mapfile -t PERF_PATHS < <(
  sed -n '/^pub const PERF_PATHS/,/^];/p' "$COVERAGE" 2>/dev/null |
    sed -n 's/^[[:space:]]*"\([^"]\+\)",\{0,1\}$/\1/p'
)
if [ "${#PERF_PATHS[@]}" -lt 8 ]; then
  echo "could not look: parsed only ${#PERF_PATHS[@]} entries from PERF_PATHS in $COVERAGE."
  echo "  The perf-path list is defined there and nowhere else. If it moved, this"
  echo "  script must be pointed at the new home -- it deliberately has no fallback"
  echo "  copy, because a short list would report 'main moved on nothing'."
  exit 0
fi

have() { git cat-file -e "$1^{commit}" 2>/dev/null; }
have "$BASE" || { echo "could not look: $BASE is not in this clone (shallow checkout?)."; exit 0; }
have "$HEAD_REF" || { echo "could not look: $HEAD_REF is not in this clone."; exit 0; }

p2=$(git rev-parse -q --verify "$HEAD_REF^2" 2>/dev/null)
p1=$(git rev-parse -q --verify "$HEAD_REF^1" 2>/dev/null)
if [ -z "$p2" ] || [ -z "$p1" ]; then
  echo "could not look: $HEAD_REF is not a merge commit, so the queued head cannot be"
  echo "  separated from what the queue placed under it."
  exit 0
fi

# A. What did the queue put beneath us? With one entry the group's first parent
# IS the base; anything reachable from it and not from the base is another
# entry's history.
beneath=$(git log --no-merges --format='%h %s' "$BASE..$p1" 2>/dev/null)

# B. What did main carry in that our records never saw? Measured from the
# merge base of the queued head and the group base -- i.e. main-only history --
# so the PR's own perf changes (which its campaign DID cover) are excluded.
mb=$(git merge-base "$p2" "$BASE" 2>/dev/null)
moved=""
if [ -n "$mb" ]; then
  moved=$(git diff --name-only "$mb" "$BASE" -- "${PERF_PATHS[@]}" 2>/dev/null)
else
  echo "could not look: no merge base between the queued head and $BASE."
fi

say_a() {
  echo "::error title=Queue bounce: composed beneath you::The merge group placed $(printf '%s\n' "$beneath" | grep -c .) other commit(s) under this PR. Records measured apart never compose -- the combined tree's interactions are unmeasured and the gate refuses by design. Retrying will not help. Land record-bearing PRs one at a time; see docs/gate-queue-protocol.md."
  printf '%s\n' "$beneath" | head -10 | sed 's/^/  beneath: /'
}
say_b() {
  echo "::error title=Queue bounce: main moved under you::Nothing else was in this group, but main took $(printf '%s\n' "$moved" | grep -c .) perf-path change(s) after this PR's records were measured, so the tree under test is not the tree the campaign covered. Retrying will not help. Run scripts/queue-perf-pr.sh <branch> to freeze onto current main, re-run the campaign at the frozen sha, commit the records, and queue alone. See docs/gate-queue-protocol.md."
  printf '%s\n' "$moved" | head -10 | sed 's/^/  main-only: /'
}

if [ -n "$beneath" ] && [ -n "$moved" ]; then
  echo "both conditions hold -- fix the freeze first, then land alone:"
  say_a; say_b
elif [ -n "$beneath" ]; then
  say_a
elif [ -n "$moved" ]; then
  say_b
else
  # The honest third answer. Asserting a cause here is what made the old step
  # misleading.
  echo "::error title=Certification failed, and it is not a queue condition::Nothing else was in this group and main moved on no perf path since these records were measured, so neither queue composition nor a moved main explains this. Read the certification log above -- the refusal is about this PR's own tree."
fi
exit 0
