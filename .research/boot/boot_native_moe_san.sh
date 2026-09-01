#!/bin/bash
# compute-sanitizer boot v2: API-error noise off so real findings print.
exec /usr/local/cuda/bin/compute-sanitizer --tool memcheck \
  --report-api-errors no --print-limit 20 --error-exitcode 99 \
  bash /home/ms/atlas/.claude/worktrees/exl3-research/boot_native_moe.sh
