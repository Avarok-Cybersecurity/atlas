#!/bin/bash
cd /home/ms/atlas/.claude/worktrees/exl3-research || exit 1
export PATH=/usr/local/cuda/bin:$PATH
export CUTLASS_HOME=/home/ms/cutlass
export FLASHINFER_HOME=/home/ms/flashinfer
export RUSTFLAGS="-L/home/ms/nccl/build/lib -L/usr/local/cuda/lib64"
export ATLAS_TARGET_HW=gb10
export ATLAS_TARGET_MODEL=qwen3.8-flash-next
export ATLAS_TARGET_QUANT=nvfp4
cargo build --release --bin spark
echo "build rc=$?"
