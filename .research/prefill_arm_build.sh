#!/bin/bash
# Prefill-arm build/test driver (exact task env). Usage:
#   ./prefill_arm_build.sh <logname> <cargo args...>
set -u
cd /home/ms/atlas/.claude/worktrees/exl3-research || exit 9
export PATH=/usr/local/cuda/bin:$PATH
export CUTLASS_HOME=/home/ms/cutlass
export FLASHINFER_HOME=/home/ms/flashinfer
export RUSTFLAGS="-L/home/ms/nccl/build/lib -L/usr/local/cuda/lib64"
export ATLAS_TARGET_HW=gb10
export ATLAS_TARGET_MODEL=qwen3.8-flash-next
export ATLAS_TARGET_QUANT=nvfp4
LOG=/home/ms/.claude/jobs/5a7bd33d/tmp/$1.log
shift
cargo "$@" > "$LOG" 2>&1
echo "EXIT=$? LOG=$LOG"
