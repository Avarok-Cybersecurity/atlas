#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Table-driven test for harvest-triage.sh. Every row is a state the harvest
# has actually been in, or one it will be in eventually; the comment names the
# incident where that row was learned the hard way.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
triage="$here/harvest-triage.sh"
fails=0

check() {  # expected, state, in_flight, broken, why
  local expected="$1" state="$2" in_flight="$3" broken="$4" why="$5" got
  got="$("$triage" "$state" "$in_flight" "$broken" 2>&1)" || got="EXIT$?:$got"
  if [ "$got" != "$expected" ]; then
    printf 'FAIL  %-7s %-9s in_flight=%s broken=%s -> got %-8s (%s)\n' \
      "$expected" "$state" "$in_flight" "$broken" "$got" "$why"
    fails=$((fails + 1))
  else
    printf 'ok    %-7s %-9s in_flight=%s broken=%s  %s\n' \
      "$expected" "$state" "$in_flight" "$broken" "$why"
  fi
}

# A healthy PR is never touched: pushing to it restarts its checks, and on a
# busy repo the restarts outrun the checks. This is the #574 invariant.
check leave  CLEAN     0 0 "healthy and idle stays untouched (#574)"
check wait   CLEAN     3 0 "checks in flight are allowed to finish (#574)"
check wait   BLOCKED   5 0 "blocked only because checks have not reported yet"

# BEHIND must NOT repair while checks run, or every unrelated merge to main
# restarts the PR and it never converges — the #571/#574 livelock.
check wait   BEHIND    2 0 "BEHIND with checks running waits, no livelock (#571)"
check repair BEHIND    0 0 "BEHIND and idle is terminal, rebuild it (#585)"

# A hung or cancelled build never clears by waiting. #580 sat for six hours
# after the linux CUDA job hit GitHub's job ceiling.
check repair CLEAN     0 1 "a failed check with nothing running is terminal (#585)"
check repair BLOCKED   0 2 "several broken checks, idle, rebuild (#585)"
check wait   CLEAN     1 1 "one broken but another still running, let it finish"

# DIRTY outranks in-flight checks: a conflict survives whatever they conclude.
check repair DIRTY     0 0 "conflicting PR is not 'idle and healthy' (#595)"
check repair DIRTY     4 0 "conflict beats in-flight checks, no wasted cycle"
check repair DIRTY     0 3 "conflicting and broken still rebuilds"

# Unknown states are left alone rather than churned: GitHub reports UNKNOWN
# while it computes mergeability, and it resolves on its own.
check leave  UNKNOWN   0 0 "transient UNKNOWN is not a reason to rebuild"

# Bad input is refused rather than silently treated as zero.
got="$("$triage" CLEAN x 0 2>&1)" && st=0 || st=$?
if [ "$st" -ne 2 ]; then
  echo "FAIL  non-numeric count should exit 2, got exit $st"
  fails=$((fails + 1))
else
  echo "ok    exit 2  non-numeric count is refused, not read as zero"
fi

if [ "$fails" -ne 0 ]; then
  echo "$fails triage case(s) failed"
  exit 1
fi
echo "all triage cases pass"
