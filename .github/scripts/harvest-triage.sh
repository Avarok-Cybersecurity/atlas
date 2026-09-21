#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# harvest-triage.sh — decide what governance-harvest.yml should do about an
# already-open harvest PR. Prints exactly one word:
#
#   wait    — checks are running; leave it and let them finish
#   leave   — healthy and idle; leave it, its successor carries anything new
#   repair  — it cannot merge on its own; close it and open a fresh one
#             (a pushed fix does not make an existing PR's checks report)
#
# This lives outside the workflow because the decision is where every bug in
# this subsystem has been. In one day it churned an open PR with identical
# content (#571), churned it with legitimately new content (#574), abandoned
# one whose build hung for six hours (#585), and called a conflicting PR
# "idle and healthy" (#595). Each was found in production, on a stalled PR.
# A table-driven test is cheaper than a fifth round of that.
#
# Usage: harvest-triage.sh <state> <in_flight> <broken> [failing_check_names]
#
# The optional 4th argument is the newline-separated NAMES of the failing or
# cancelled checks. When given, checks that are HELD rather than broken are
# discounted -- see the list below and the reason it exists.
set -euo pipefail

state="${1:?mergeStateStatus required}"
in_flight="${2:?in-flight check count required}"
broken="${3:?failed-or-cancelled check count required}"
failing_names="${4:-}"

case "$in_flight$broken" in
  *[!0-9]*) echo "counts must be non-negative integers, got '$in_flight' and '$broken'" >&2; exit 2 ;;
esac

# HELD IS NOT BROKEN, and conflating them is a recreate loop.
#
# `repair` means "close this PR and rebuild the branch". That is only worth
# doing if rebuilding would change the outcome. These three checks are
# IDENTICAL on a fresh PR, because they are waiting on a human, not on CI:
#
#   PR benchmark gate / PR Benchmark Certifications — certification is HELD
#     until someone comments /stamp. Nobody stamps a bot PR.
#   seal status — fails closed until a codeowner comments /seal.
#
# Counting them as broken meant every harvest cycle read its own healthy PR as
# terminal and recreated it. Measured 2026-09-04: twelve PRs opened and closed
# in one day, about half the repository's runner queue. The `workflow_run`
# guard in governance-harvest.yml stops that loop being re-armed by the PR's
# own CI; this stops it being armed in the first place.
if [ -n "$failing_names" ]; then
  held=$(printf '%s\n' "$failing_names" \
    | grep -cxE 'PR benchmark gate|PR Benchmark Certifications|seal status' || true)
  broken=$((broken - held))
  [ "$broken" -lt 0 ] && broken=0
fi

# DIRTY first, and deliberately ahead of the in-flight check. A conflicting
# branch will still be conflicting when those checks finish, so waiting only
# spends a full CI cycle to reach the same rebuild.
if [ "$state" = DIRTY ]; then
  echo repair
  exit 0
fi

# BEHIND is NOT treated that way. main moves constantly here, so a PR can go
# BEHIND minutes after opening; repairing on sight would restart its checks on
# every unrelated merge, which is precisely the livelock #571 and #574 closed.
# Waiting for the in-flight run first is what keeps it convergent.
if [ "$in_flight" -gt 0 ]; then
  echo wait
  exit 0
fi

if [ "$broken" -gt 0 ] || [ "$state" = BEHIND ]; then
  echo repair
  exit 0
fi

echo leave
