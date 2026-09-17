#!/usr/bin/env bash
# Build Atlas for AMD GPUs via SCALE (recompiles the unmodified CUDA kernels).
#
# The hardware target comes from ATLAS_TARGET_HW and defaults to `strix`, so
# an unset environment builds exactly what it always did. The SCALE toolchain
# directory is not hardcoded: it is read from the `arch` key of
# kernels/$ATLAS_TARGET_HW/HARDWARE.toml, which is the same key
# atlas-kernels/build_target.rs compiles with, so the script and the build
# cannot disagree about which arch this is.
#
#   ./build-amd.sh                        # strix  / gfx1151 / qwen3.6-27b
#   ATLAS_TARGET_HW=r9700 ./build-amd.sh  # r9700  / gfx1201 / qwen3.8-27b
#
# Verified: gfx1151 / Strix Halo, SCALE 1.7.1, native Ubuntu; and gfx1201 /
# Radeon AI PRO R9700, SCALE 1.7.1 targets/gfx1201, ROCm 7.2.0 (97/97 kernels
# compile, 94-kernel spark-server build green). See
# docs/porting/amd-strix-halo-scale.md and the r9700 section of
# docs/HARDWARE.md.
set -euo pipefail
cd "$(dirname "$0")"

: "${SCALE_HOME:=$HOME/scale171/scale-1.7.1-Linux}"
export SCALE_HOME
export ATLAS_TARGET_HW="${ATLAS_TARGET_HW:-strix}"

# The model default follows the hardware. Strix was brought up on qwen3.6-27b
# and stays there; the r9700 tree also carries qwen3.8-27b (gb10's MODEL.toml
# with kernel_source = "qwen3.6-27b"), which is what its bring-up builds.
case "$ATLAS_TARGET_HW" in
  r9700) : "${ATLAS_TARGET_MODEL:=qwen3.8-27b}" ;;
  *) : "${ATLAS_TARGET_MODEL:=qwen3.6-27b}" ;;
esac
export ATLAS_TARGET_MODEL
export ATLAS_TARGET_QUANT="${ATLAS_TARGET_QUANT:-nvfp4}"

hardware_toml="kernels/$ATLAS_TARGET_HW/HARDWARE.toml"
if [[ ! -f "$hardware_toml" ]]; then
  echo "build-amd.sh: no such hardware target: $hardware_toml" >&2
  echo "  ATLAS_TARGET_HW must name a kernels/<hw>/ directory, e.g. strix or r9700." >&2
  exit 1
fi

# One grep + sed rather than a TOML tool: `arch` is a bare quoted scalar on its
# own line in every kernels/<hw>/HARDWARE.toml, and a bring-up box is not
# guaranteed to have a parser installed. The leading-whitespace anchor keeps
# this off the `# ... arch selects ...` comment lines above the key.
arch="$(grep -m1 -E '^[[:space:]]*arch[[:space:]]*=' "$hardware_toml" \
  | sed -E 's/^[^=]*=[[:space:]]*"([^"]+)".*/\1/')" || true
if [[ -z "$arch" ]]; then
  echo "build-amd.sh: no [hardware].arch key in $hardware_toml" >&2
  exit 1
fi

scale_target="$SCALE_HOME/targets/$arch"
if [[ ! -x "$scale_target/bin/nvcc" ]]; then
  echo "build-amd.sh: SCALE has no usable $arch target at $scale_target" >&2
  echo "  ($hardware_toml declares arch = \"$arch\"; expected $scale_target/bin/nvcc)" >&2
  echo "  Install the SCALE build that ships targets/$arch, or point SCALE_HOME at it." >&2
  exit 1
fi

export CUDA_PATH="$scale_target"
export CUDA_HOME="$CUDA_PATH"
export PATH="$scale_target/bin:/opt/rocm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/rocm/lib:$scale_target/lib:${LD_LIBRARY_PATH:-}"
# cudarc gates its API surface on this; SCALE presents a CUDA 12.8 API.
export CUDARC_CUDA_VERSION=12080

echo "nvcc -> $(command -v nvcc)  (SCALE $arch for $ATLAS_TARGET_HW/$ATLAS_TARGET_MODEL/$ATLAS_TARGET_QUANT)"
rm -rf target/release/build/atlas-kernels-* target/release/build/spark-storage-*
cargo build --release -p spark-server --no-default-features --features cuda
