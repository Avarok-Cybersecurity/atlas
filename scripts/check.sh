#!/usr/bin/env bash
# Fast Rust-only iteration loop. Skips the multi-minute PTX compile
# (SKIP_ATLAS_BUILD=1) and bypasses cudarc's nvcc probe
# (CUDARC_CUDA_VERSION=12000). Use for `cargo check`, `cargo clippy`,
# `cargo clippy --tests`, etc. when you only care about Rust correctness.
#
# Usage:
#   scripts/check.sh                       # cargo check on the workspace
#   scripts/check.sh clippy --tests        # cargo clippy --tests
#   scripts/check.sh clippy -p spark-server
#
# Anything that needs to actually launch a kernel (Docker build, perf
# tests, runtime smoke) MUST go through the real build path — the stub
# registry produced under SKIP_ATLAS_BUILD has zero PTX.

set -euo pipefail

export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-12000}"
export SKIP_ATLAS_BUILD="${SKIP_ATLAS_BUILD:-1}"

# If the first arg is a known cargo subcommand, pass through verbatim.
# Otherwise default to `check` and forward all args (e.g. `check.sh -p
# spark-server` runs `cargo check -p spark-server`).
case "${1:-}" in
  ""|build|check|clippy|test|fmt|doc|run|tree|metadata|update|fix)
    exec /workspace/.cargo/bin/cargo "${@:-check}"
    ;;
  *)
    exec /workspace/.cargo/bin/cargo check "$@"
    ;;
esac
