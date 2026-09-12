#!/bin/bash
# Counterbalanced A/B: A1 (prefix ON), B1 (OFF), B2 (OFF), A2 (ON).
# Order is counterbalanced so any drift in box state does not alias onto the arm.
OUT=/home/ms/.claude/jobs/5a7bd33d/tmp/mangle2
for spec in "A1 on" "B1 off" "B2 off" "A2 on"; do
  set -- $spec
  bash "$OUT/arm.sh" "$1" "$2" > "$OUT/$1.console.log" 2>&1
  echo "CHAIN: $1 finished rc=$? at $(date -u +%FT%TZ)"
done
echo "CHAIN: ALL ARMS COMPLETE at $(date -u +%FT%TZ)"
