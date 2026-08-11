#!/usr/bin/env bash
# Fast Rust-only iteration loop. Skips the multi-minute PTX compile
# (ATLAS_SKIP_BUILD=1) and bypasses cudarc's nvcc probe
# (CUDARC_CUDA_VERSION). Use for `cargo check`, `cargo clippy`,
# `cargo clippy --tests`, etc. when you only care about Rust correctness.
#
# Usage:
#   scripts/check.sh                       # cargo check on the workspace
#   scripts/check.sh clippy --tests        # cargo clippy --tests
#   scripts/check.sh clippy -p spark-server
#
# Anything that needs to actually launch a kernel (Docker build, perf
# tests, runtime smoke) MUST go through the real build path — the stub
# registry produced under ATLAS_SKIP_BUILD has zero PTX.

set -euo pipefail

# 13000, matching ci.yml and docs.yml. This only short-circuits cudarc's
# `nvcc --version` probe, but it must not claim a version below the repo's
# CUDA 13.0 floor: GB10 is sm_121 and a 12.0 claim describes a toolkit that
# cannot target it.
export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13000}"

# ATLAS_SKIP_BUILD is the name the kernel build actually reads
# (crates/atlas-kernels/build.rs). This script exported only the
# SKIP_ATLAS_BUILD spelling, which atlas-kernels does NOT match — so the
# "fast" wrapper ran the full nvcc PTX compile every time. Both are exported
# because spark-storage/build.rs accepts either and callers may already have
# one of them set.
export ATLAS_SKIP_BUILD="${ATLAS_SKIP_BUILD:-${SKIP_ATLAS_BUILD:-1}}"
export SKIP_ATLAS_BUILD="${SKIP_ATLAS_BUILD:-$ATLAS_SKIP_BUILD}"

# `cargo` from PATH. An absolute path here worked on one box and on no
# contributor's machine.
CARGO="${CARGO:-cargo}"

# If the first arg is a known cargo subcommand, pass through verbatim.
# Otherwise default to `check` and forward all args (e.g. `check.sh -p
# spark-server` runs `cargo check -p spark-server`).
case "${1:-}" in
  ""|build|check|clippy|test|fmt|doc|run|tree|metadata|update|fix)
    exec "$CARGO" "${@:-check}"
    ;;
  *)
    exec "$CARGO" check "$@"
    ;;
esac
