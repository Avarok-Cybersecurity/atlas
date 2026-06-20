#!/usr/bin/env bash
# Build Atlas for AMD GPUs via SCALE (recompiles the unmodified CUDA kernels).
# Verified: gfx1151 / Strix Halo, SCALE 1.7.1, native Ubuntu. See
# docs/porting/amd-strix-halo-scale.md.
set -euo pipefail
cd "$(dirname "$0")"
: "${SCALE_HOME:=$HOME/scale171/scale-1.7.1-Linux}"
export SCALE_HOME
export ATLAS_TARGET_HW="${ATLAS_TARGET_HW:-strix}"
export ATLAS_TARGET_MODEL="${ATLAS_TARGET_MODEL:-qwen3.6-27b}"
export ATLAS_TARGET_QUANT="${ATLAS_TARGET_QUANT:-nvfp4}"
export CUDA_PATH="$SCALE_HOME/targets/gfx1151"
export CUDA_HOME="$CUDA_PATH"
export PATH="$SCALE_HOME/targets/gfx1151/bin:/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:$SCALE_HOME/targets/gfx1151/lib:${LD_LIBRARY_PATH:-}"
export CUDARC_CUDA_VERSION=12080
echo "nvcc -> $(command -v nvcc)  (SCALE for $ATLAS_TARGET_HW/$ATLAS_TARGET_MODEL/$ATLAS_TARGET_QUANT)"
rm -rf target/release/build/atlas-kernels-* target/release/build/spark-storage-*
cargo build --release -p spark-server --no-default-features --features cuda
