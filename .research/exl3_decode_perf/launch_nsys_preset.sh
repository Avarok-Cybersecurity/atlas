#!/bin/bash
# Boot the branch binary through the named preset under nsys launch mode (capture via nsys start/stop).
# $1 = port. SPARK_BIN overrides the binary. Extra args appended.
PORT=${1:-8899}; shift
BIN=${SPARK_BIN:-/home/ms/atlas/.claude/worktrees/exl3-research/target/release/spark}
export TMPDIR=/home/ms/.claude/jobs/5a7bd33d/tmp/exl3bench/nsystmp
export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export RUST_LOG=${RUST_LOG:-info}
cd /home/ms/atlas/.claude/worktrees/exl3-research
exec /usr/local/cuda/bin/nsys launch --trace=cuda,nvtx \
  "$BIN" serve qwen3.8-flash-next-exl3 --model-from-path /tank/exl3-ckpt/qwen38-flash-next-4.05bpw \
  --bind 127.0.0.1 --port "$PORT" "$@"
