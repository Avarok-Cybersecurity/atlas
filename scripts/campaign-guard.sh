#!/usr/bin/env bash
# campaign-guard.sh — has the branch drifted out from under a running campaign?
#
# WHY THIS EXISTS. On 2026-08-28 a ten-gate campaign ran for nine hours against
# `wip/radixark-qwen4-exp` at 1d30d5d5ff. Twenty minutes into the final gate a
# collaborator pushed one 23-line change to
# crates/spark-model/src/weight_loader/qwen4_exp/ffn.rs — a PERF_PATH. Records
# are content-anchored, so all ten were invalidated. That was discovered at the
# END, by which point the GPU time was spent. Called between gates, and on a
# timer during the long ones, this turns nine wasted hours into twenty minutes.
#
# It answers a CONTENT question, not a SHA one. A push that touches only docs,
# scripts or the site leaves the anchor intact and must not abort anything —
# aborting on a README commit would be self-inflicted waste.
#
#   scripts/campaign-guard.sh <anchor-sha> [remote-ref]
#
# exit 0  the anchor still describes the branch (unmoved, or moved harmlessly)
# exit 1  a perf path moved; whatever is running is measuring a dead tree
# exit 2  the guard could not answer (never treated as "safe to continue")
set -uo pipefail

if [ $# -lt 1 ]; then
    # exit 2, not 1: a caller wiring this into `guard || abort` must be able to
    # tell "a perf path moved" (1) from "I could not answer" (2). Bash's
    # ${1:?} would exit 1 and be read as the former.
    echo "campaign-guard: usage: campaign-guard.sh <anchor-sha> [remote-ref]" >&2
    exit 2
fi
ANCHOR="$1"
REF="${2:-HEAD}"
REMOTE="${REMOTE:-origin}"

die() { echo "campaign-guard: $1" >&2; exit 2; }

# PERF_PATHS is read from the gate's own source rather than copied here. A
# second list would drift, and the moment it did this guard would report safe
# for a path the gate counts.
COVERAGE=crates/atlas-plugin/src/gate/coverage.rs
[ -f "$COVERAGE" ] || die "run me from the repo root; $COVERAGE not found"
paths=$(sed -n '/pub const PERF_PATHS/,/];/p' "$COVERAGE" | grep -oE '"[^"]+"' | tr -d '"')
[ -n "$paths" ] || die "could not read PERF_PATHS out of $COVERAGE"

git rev-parse --verify --quiet "$ANCHOR^{commit}" >/dev/null || die "anchor $ANCHOR is not a commit here"

git fetch -q "$REMOTE" "$REF" 2>/dev/null || die "could not fetch $REMOTE $REF"
head=$(git rev-parse FETCH_HEAD)

if [ "$head" = "$(git rev-parse "$ANCHOR")" ]; then
    echo "campaign-guard: anchor is still the head of $REF"
    exit 0
fi

# Moved. The only question that matters is whether it moved where the gate looks.
# shellcheck disable=SC2086  # $paths must word-split: it is a list of pathspecs
moved=$(git diff --name-only "$ANCHOR" "$head" -- $paths)

if [ -z "$moved" ]; then
    echo "campaign-guard: $REF moved to ${head:0:10}, but no PERF_PATH changed — records still stand"
    exit 0
fi

echo "campaign-guard: STOP — $REF moved to ${head:0:10} and these PERF_PATHS changed:" >&2
echo "$moved" | sed 's/^/  /' >&2
echo "campaign-guard: every record anchored to ${ANCHOR:0:10} is now invalid; do not keep measuring." >&2
exit 1
