#!/bin/bash
cd /home/ms/atlas/.claude/worktrees/exl3-research || exit 9
EXL3_PF_DEBUG=1 ./target/release/examples/exl3_native_parity \
  > /home/ms/.claude/jobs/5a7bd33d/tmp/pf_debug.log 2>&1
echo "EXIT=$?"
tail -15 /home/ms/.claude/jobs/5a7bd33d/tmp/pf_debug.log
