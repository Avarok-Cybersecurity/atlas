#!/bin/bash
# Debug boot: CUDA_LAUNCH_BLOCKING=1 to localize the illegal-address launch.
export CUDA_LAUNCH_BLOCKING=1
exec bash /home/ms/atlas/.claude/worktrees/exl3-research/boot_native_moe.sh
