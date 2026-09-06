#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Table-driven test for harvest-triage.sh. Every row is a state the harvest
# has actually been in, or one it will be in eventually; the comment names the
# incident where that row was learned the hard way.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
triage="$here/harvest-triage.sh"
fails=0

check() {  # expected, state, in_flight, broken, why [, failing_names]
  local expected="$1" state="$2" in_flight="$3" broken="$4" why="$5" names="${6:-}" got
  if [ -n "$names" ]; then
    got="$("$triage" "$state" "$in_flight" "$broken" "$names" 2>&1)" || got="EXIT$?:$got"
  else
    got="$("$triage" "$state" "$in_flight" "$broken" 2>&1)" || got="EXIT$?:$got"
  fi
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

# HELD IS NOT BROKEN. `repair` closes the PR and rebuilds the branch, which is
# only worth doing if rebuilding changes the outcome. A gate waiting on a human
# is identical on a fresh PR, so counting it as broken is an infinite recreate
# loop -- twelve PRs in one day, about half the runner queue (2026-09-04).
check leave  CLEAN 0 1 "a HELD certification gate is not breakage" "PR benchmark gate"
check leave  CLEAN 0 1 "an unsealed PR is waiting on a human, not on CI" "seal status"
check leave  CLEAN 0 2 "the real bot PR: held gate + unsealed" "PR benchmark gate
seal status"
check leave  CLEAN 0 3 "all three held checks together" "PR benchmark gate
PR Benchmark Certifications
seal status"

# ...but a genuine failure still repairs, including when mixed with held ones.
# A rule that discounted everything would be worse than the bug it replaced.
check repair CLEAN 0 1 "a real failing check still rebuilds" "cargo test --workspace"
check repair CLEAN 0 2 "held + genuine: the genuine one still counts" "PR benchmark gate
cargo test --workspace"
check wait   CLEAN 2 1 "in-flight still wins over a held failure" "PR benchmark gate"

# The 3-argument form must keep its old meaning: callers that do not pass names
# cannot distinguish held from broken, and must stay conservative.
check repair CLEAN 0 1 "legacy 3-arg form is unchanged (no names => all broken)"

if [ "$fails" -ne 0 ]; then
  echo "$fails triage case(s) failed"
  exit 1
fi
echo "all triage cases pass"
