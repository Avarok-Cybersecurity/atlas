#!/bin/bash
# Wait for the A/B chain to finish, then run the kernel-controlled C arms.
OUT=/home/ms/.claude/jobs/5a7bd33d/tmp/mangle2
while ! grep -a -q "ALL ARMS COMPLETE" "$OUT/chain.log" 2>/dev/null; do sleep 30; done
echo "CHAIN-C: A/B done, starting C at $(date -u +%FT%TZ)"
for T in C1 C2; do
  bash "$OUT/arm_c.sh" "$T" > "$OUT/$T.console.log" 2>&1
  echo "CHAIN-C: $T finished rc=$? at $(date -u +%FT%TZ)"
done
echo "CHAIN-C: ALL C ARMS COMPLETE at $(date -u +%FT%TZ)"
