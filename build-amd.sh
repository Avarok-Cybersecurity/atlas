#!/usr/bin/env bash
# Build Atlas for AMD GPUs. Verified on gfx1151 / Strix Halo, ROCm 7.13, native
# Ubuntu. See docs/porting/amd-strix-halo-scale.md.
#
# Two backends, selected with ATLAS_TARGET_HW:
#
#   strix-hip  (default)  native HIP via hipcc. Needs ROCm and cargo, nothing
#                         else — atlas-kernels builds its own libcuda/libcudart/
#                         libcublasLt shims from crates/atlas-kernels/hip/.
#   strix                 SCALE (scale-lang.com) recompiles the unmodified CUDA.
#                         Needs a SCALE install at $SCALE_HOME.
#
# ATLAS_TARGET_MODEL selects which kernel targets are embedded. The default '*'
# builds every target under kernels/$ATLAS_TARGET_HW/ into ONE binary that can
# serve any of them — Qwen3.8-27B reuses Qwen3.6-27B's kernel tree via
# `kernel_source`, so this costs far less than it sounds (measured: 98 unique
# nvcc invocations for 283 requested, 2.9x dedup). Set it to a single target
# name (e.g. qwen3.8-27b) for a smaller binary.
set -euo pipefail
cd "$(dirname "$0")"

export ATLAS_TARGET_HW="${ATLAS_TARGET_HW:-strix-hip}"
export ATLAS_TARGET_MODEL="${ATLAS_TARGET_MODEL:-*}"
export ATLAS_TARGET_QUANT="${ATLAS_TARGET_QUANT:-nvfp4}"
export CUDARC_CUDA_VERSION=12080
# Strix Halo is a single-APU laptop part; the RDMA verbs shim is irrelevant and
# this opt-out removes the libibverbs-dev prerequisite the old recipe carried.
export ATLAS_NO_RDMA="${ATLAS_NO_RDMA:-1}"

case "$ATLAS_TARGET_HW" in
  strix-hip)
    export ATLAS_HIPCC="${ATLAS_HIPCC:-/opt/rocm/bin/hipcc}"
    export ATLAS_HIP_COMPAT_INCLUDE="$PWD/crates/atlas-kernels/hip/compat"
    export PATH="/opt/rocm/bin:$PATH"
    # Deliberately NO `RUSTFLAGS=-L .../hip-port/link`. atlas-kernels/build.rs
    # compiles the three HIP shims into OUT_DIR and puts that first on the link
    # path; pointing -L at a hand-built shim directory shadows them with an
    # older libcuda.so and the link fails on cuStreamQuery /
    # cuMemHostGetDevicePointer_v2 / the 11 cublasLt* symbols.
    echo "hipcc -> $ATLAS_HIPCC  (native HIP, $ATLAS_TARGET_HW/$ATLAS_TARGET_MODEL/$ATLAS_TARGET_QUANT)"
    ;;
  strix)
    : "${SCALE_HOME:=$HOME/scale171/scale-1.7.1-Linux}"
    export SCALE_HOME
    export CUDA_PATH="$SCALE_HOME/targets/gfx1151"
    export CUDA_HOME="$CUDA_PATH"
    export PATH="$SCALE_HOME/targets/gfx1151/bin:/opt/rocm/bin:$PATH"
    export LD_LIBRARY_PATH="/opt/rocm/lib:$SCALE_HOME/targets/gfx1151/lib:${LD_LIBRARY_PATH:-}"
    echo "nvcc -> $(command -v nvcc)  (SCALE, $ATLAS_TARGET_HW/$ATLAS_TARGET_MODEL/$ATLAS_TARGET_QUANT)"
    ;;
  *) echo "ATLAS_TARGET_HW must be strix-hip or strix (got: $ATLAS_TARGET_HW)" >&2; exit 2 ;;
esac

# rustup installs cargo to ~/.cargo/bin, which a non-interactive shell (ssh
# "cmd", CI, systemd) does not get from the login profile.
command -v cargo >/dev/null || export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || { echo "cargo not found; install Rust (rustup) first" >&2; exit 2; }

rm -rf target/release/build/atlas-kernels-* target/release/build/spark-storage-*
cargo build --release -p spark-server --no-default-features --features cuda
