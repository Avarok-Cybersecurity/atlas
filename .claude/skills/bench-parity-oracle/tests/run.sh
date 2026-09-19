#!/usr/bin/env bash
# Three fixtures, three verdicts. The middle one is the control that matters:
# a check that can only ever say "in parity" measures nothing.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); O="$here/../parity.py"; fail=0
chk(){ python3 "$O" < "$here/$1" >/dev/null 2>&1; rc=$?
  [ "$rc" = "$2" ] && echo "  ok   $1 -> rc=$rc ($3)" || { echo "  FAIL $1 -> rc=$rc, wanted $2 ($3)"; fail=1; }; }
chk in-parity.jsonl    0 "IN PARITY"
chk osl-differs.jsonl  1 "NOT IN PARITY - only the vllm leg's OSL changed"
chk undetermined.jsonl 1 "UNDETERMINED - client axes not supplied"
exit $fail
