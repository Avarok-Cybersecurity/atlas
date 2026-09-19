#!/usr/bin/env bash
# Five fixtures, and the ones that matter are the ones that must NOT say
# "in parity". A check that can only ever agree measures nothing.
#
# Every command here is a real spelling, not a convenient one. The first
# version of in-parity.jsonl gave vLLM `--num-speculative-tokens 3` with no
# method -- a flag combination vLLM 0.27 does not accept as a complete
# speculation config -- and it passed only because the oracle was comparing
# draft widths as bare strings. It is now the `spec-unreadable` fixture, where
# UNDETERMINED is the right answer, and in-parity.jsonl carries the
# `--speculative-config '{"method":"mtp",...}'` form that
# bench/ladder38/published.json actually ran.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); O="$here/../parity.py"; fail=0
chk(){ python3 "$O" "$here/$1" >/dev/null 2>&1; rc=$?
  [ "$rc" = "$2" ] && echo "  ok   $1 -> rc=$rc ($3)" || { echo "  FAIL $1 -> rc=$rc, wanted $2 ($3)"; fail=1; }; }
# and once through stdin, because both entry points must agree
chkin(){ python3 "$O" < "$here/$1" >/dev/null 2>&1; rc=$?
  [ "$rc" = "$2" ] && echo "  ok   $1 (stdin) -> rc=$rc" || { echo "  FAIL $1 (stdin) -> rc=$rc, wanted $2"; fail=1; }; }

chk in-parity.jsonl       0 "IN PARITY — the published ladder38 Atlas vs vLLM+MTP pair"
chk osl-differs.jsonl     1 "NOT IN PARITY — only the vllm leg's OSL changed"
chk undetermined.jsonl    1 "UNDETERMINED — client axes not supplied"
chk spec-off.jsonl        1 "NOT IN PARITY — one leg runs without speculation at all"
chk spec-unreadable.jsonl 1 "UNDETERMINED — a draft width with no method names no comparable setting"
chkin in-parity.jsonl     0
chkin spec-off.jsonl      1
exit $fail
