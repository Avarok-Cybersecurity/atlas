#!/usr/bin/env bash
# Table-driven test for governance-evict.sh. The ledger is a real record and
# eviction is the only operation that removes anything from it, so every row
# here is a case where getting it wrong deletes a file that was still needed.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
evict="$here/governance-evict.sh"
fails=0

check() {  # expected (space-separated, "" for none), cap, open, ledger, why
  local expected="$1" cap="$2" open="$3" ledger="$4" why="$5" got
  got="$(printf '%s\n' $ledger | "$evict" "$cap" "$open" 2>&1 | tr '\n' ' ')"
  got="${got% }"
  if [ "$got" != "$expected" ]; then
    printf 'FAIL  cap=%-3s open=[%-9s] -> got [%s], wanted [%s]  (%s)\n' \
      "$cap" "$open" "$got" "$expected" "$why"
    fails=$((fails + 1))
  else
    printf 'ok    cap=%-3s open=[%-9s] -> [%-11s] %s\n' "$cap" "$open" "$got" "$why"
  fi
}

# The common case: the window is not full, so nothing moves. This is what the
# cap does for most of its life and it must be a true no-op.
check ""            100 ""     "500 501 502 503 504 505" "under the cap evicts nothing"
check ""            6   ""     "500 501 502 503 504 505" "exactly at the cap evicts nothing"

# Over the cap: evict exactly what falls outside the newest-`cap` window, oldest
# first. Never a round number, never "while we're here" — over-eviction is
# unrecoverable once the source CI artifacts expire at 30 days.
check "500"         5   ""     "500 501 502 503 504 505" "one over the cap evicts exactly one"
check "500 501"     4   ""     "500 501 502 503 504 505" "two over evicts the two oldest"
check "500 501 502 503 504 505" 0 "" "500 501 502 503 504 505" "cap 0 evicts every closed PR"

# FIFO is by PR NUMBER, not by input order — the workflow's glob order is not
# a contract and must not become one.
check "500 501"     4   ""     "505 500 503 501 504 502" "unsorted input still evicts the oldest two"

# ── The rule the whole script exists for ────────────────────────────────────
# `gate::required::intent_source()` looks a ledger up by bare path with no
# fallback. Evicting an OPEN PR's file does not raise — it silently reports
# NotRecorded and that PR drops to path-only gating. A wrong answer with no
# error, so the cap yields to this rule rather than the other way round.
check "501"         4   "500"  "500 501 502 503 504 505" "an open PR outside the window is skipped, not substituted"
check ""            4   "500 501" "500 501 502 503 504 505" "all outside candidates open: overshoots rather than break gating"

# ★ The regression this table caught before it shipped. Walking from the oldest
# and evicting until the count reaches `cap` also satisfies "bounded", but with
# old PRs pinned open it gets there by evicting the NEWEST ledgers — the recent
# intent data the trajectory view is built on. Only files outside the window are
# ever candidates.
check ""            2   "500 501" "500 501 502 503"       "never reaches into the window to make up the shortfall"

# Degenerate input that occurs on the very first run.
check ""            100 ""     ""                        "an empty ledger evicts nothing"

# Malformed input is exit 2, never a guess. The candidate list comes from a
# glob; if the glob changes shape, deleting files on a best-effort reading of
# it is exactly the wrong response.
bad() {  # why, cap, open, ledger
  local why="$1" cap="$2" open="$3" ledger="$4" out rc
  out="$(printf '%s\n' $ledger | "$evict" "$cap" "$open" 2>&1)"; rc=$?
  if [ "$rc" -ne 2 ]; then
    printf 'FAIL  expected exit 2 for %s, got %s (%s)\n' "$why" "$rc" "$out"
    fails=$((fails + 1))
  else
    printf 'ok    exit 2  %s\n' "$why"
  fi
}
bad "a non-numeric cap"          "abc" ""    "502"
bad "a non-numeric open PR"      "1"   "x502" "502"
bad "a non-numeric ledger entry" "1"   ""    "pr-502.jsonl"

if [ "$fails" -gt 0 ]; then
  echo "$fails eviction case(s) failed"
  exit 1
fi
echo "all eviction cases pass"
