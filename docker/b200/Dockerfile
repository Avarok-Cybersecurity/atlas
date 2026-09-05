# Atlas Spark — B200 / GB200 (sm_100a), all five datacentre models
#
# WHY THIS FILE EXISTS. Renting a B200 box by the hour and then compiling
# Atlas on it burns the expensive resource on work that needs no GPU:
# `nvcc --ptx -arch=sm_100a` is a cross-compile, and the kernels ship as PTX
# embedded in the binary. Everything below runs on any x86_64 Linux host with
# Docker. The rental box only pulls.
#
# Build from repo root (no GPU required, ~1 h cold):
#   docker build -f docker/b200/Dockerfile -t atlas-b200:latest .
#
#   # one model instead of all five (much faster; the binary then only serves it)
#   docker build -f docker/b200/Dockerfile \
#     --build-arg ATLAS_TARGET_MODEL=nemotron-super-120b-a12b \
#     -t atlas-b200:nemotron .
#
# Run — FIRST BOOT SHOULD BE `--check-kernels`. It resolves every kernel the
# model needs against the PTX actually compiled in, prints the unresolved
# ones, and exits with that count. Do this before timing anything:
#   docker run --gpus all --ipc=host --network host \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-b200:latest \
#     serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --check-kernels --no-tui
#
# Then serve. On a multi-GPU node use the single-node launcher rather than N
# hand-written `docker run` lines — it pins rank i to GPU i and ships NO NCCL
# environment, which is the correct configuration on an NVLink box:
#   NGPUS=8 IMAGE=atlas-b200:latest \
#     scripts/start-node-ep.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
#
# sm_100a IS NOT A STEP UP FROM sm_121. The two Blackwell architectures are
# siblings: datacentre Blackwell has tcgen05 and `redux.sync.max.abs.f32`,
# consumer/GB10 Blackwell has the warp-level
# `mma … .kind::mxf4nvf4.block_scale` that sm_100a lacks. Neither arch's PTX
# runs on the other, and B300 / GB300 are sm_103a — a different target again.
# The table is in the B200 section of docs/HARDWARE.md.
#
# THE PTX IS sm_100a AND ONLY sm_100a. `spark`'s arch preflight refuses to run
# it on a GB10, an H100 or a B300, before the driver would. That is the point:
# an image that cannot serve is better than one that JITs garbage.
#
# This file is deliberately parallel to docker/hopper/Dockerfile — the only
# functional difference is the ATLAS_TARGET_HW default. docker/
# datacenter_dockerfiles_test.sh holds both to the same assertions.

# ── Build stage ──────────────────────────────────────────────────
# Same digests as docker/gb10/Dockerfile. These are MULTI-ARCH INDEX digests,
# not the aarch64 child manifests — verified against Docker Hub on 2026-09-04:
#   13.0.0-devel   sha256:1e8ac7a5…  -> amd64 child sha256:435220c0…
#   13.0.0-runtime sha256:95318efe…  -> amd64 child sha256:4757e647…
# so `FROM …@sha256:1e8ac7a5…` on an x86_64 builder resolves the amd64 child
# and the GB10, Hopper and B200 images stay on one pinned base.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder

# Which kernel targets get compiled into the binary. `ATLAS_TARGET_HW` is an
# ARG rather than a bare ENV so this file and docker/hopper/Dockerfile differ
# in their default and nothing else; override it and you get the other image.
ARG ATLAS_TARGET_HW=b200
# '*' = every model under kernels/<hw>/ — deepseek-v4-flash,
# nemotron-3-nano-30b-a3b, nemotron-super-120b-a12b, qwen3.6-35b-a3b,
# qwen3-next-80b-a3b. Narrow it to one slug for a faster build.
ARG ATLAS_TARGET_MODEL="*"
# Named, not '*': nvfp4 is the only quant directory these targets have, and
# PCND says state it rather than let a wildcard imply a choice.
ARG ATLAS_TARGET_QUANT=nvfp4

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
# Override rust-toolchain.toml's pin: a transitive dep (libloading 0.9)
# requires >= 1.88. `stable` defers to what rustup installed above.
ENV RUSTUP_TOOLCHAIN=stable
# The vendored cudarc tops out at 13.1 in its nvcc-version table and panics on
# anything newer; every CI workflow in this repo pins the same value.
ENV CUDARC_CUDA_VERSION=13000

# CUTLASS_HOME / FLASHINFER_HOME are DELIBERATELY UNSET. Both side objects are
# optional (crates/spark-runtime/build.rs skips them when the vars are absent),
# and the CUTLASS NVFP4 wrappers would contribute nothing: they are gated on
# CUTLASS_ARCH_MMA_SM120_SUPPORTED || …SM121…, so they compile to nothing for
# sm_100a exactly as they do for sm_90a. An Sm100 collective port would change
# that; nothing here does it. docker/gb10/Dockerfile.builder is the image that
# wires those side objects up.

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/
# quick-speed-bench (atlas-plugin) include_str!'s its ISL prompt fixtures from
# tests/fixtures — without this line the release build fails in-container while
# every full-tree local build succeeds. Copy just the fixtures, not all tests/.
COPY tests/fixtures/ tests/fixtures/

ENV ATLAS_TARGET_HW=${ATLAS_TARGET_HW}
ENV ATLAS_TARGET_MODEL=${ATLAS_TARGET_MODEL}
ENV ATLAS_TARGET_QUANT=${ATLAS_TARGET_QUANT}

# `--no-default-features --features cuda,nccl` is the same set spark-server's
# `default` names today, spelled out so a change to that default cannot
# silently drop NCCL out of a multi-GPU image. NCCL links here against the
# libnccl-dev the CUDA devel base ships (2.27.7 on 13.0.0) — link-time only;
# the shared object actually loaded is the runtime stage's, which the gate
# below holds at >= 2.28.
RUN cargo build --release -p spark-server --bin spark \
      --no-default-features --features cuda,nccl

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.description="Atlas Spark, kernels compiled for NVIDIA datacentre Blackwell (sm_100a) — B200 / GB200"

# Stamp the source commit into the image so ":latest" provenance is answerable:
#   docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' <tag>
ARG ATLAS_GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${ATLAS_GIT_SHA}"

# NCCL for multi-GPU (EP/TP), RDMA userspace for RoCE transport.
# NCCL >= 2.28 is required for `ncclMemAlloc`/`ncclMemFree` symmetric memory
# windows. The base image has the NVIDIA CUDA APT repo enabled and pins
# libnccl2 with `apt-mark hold`, so the upgrade needs
# --allow-change-held-packages; the version is then verified rather than
# assumed.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" >&2; exit 1; }

COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE /LICENSE
COPY README.md /README.md

# NO NCCL_* ENVIRONMENT IS BAKED IN, on purpose. scripts/start-ep2.sh's block
# (NCCL_SOCKET_IFNAME=enp1s0f0np0, NCCL_ALGO=Ring, NCCL_PROTO=Simple,
# NCCL_NVLS_ENABLE=0, …) is correct for two GB10 chassis over RoCE and wrong
# on an NVLink node: it names a NIC that does not exist, disables NVLink
# SHARP, and forces the slowest algorithm pair onto an intra-node transport.
# On one node with NVLink the right NCCL configuration is no NCCL
# configuration. scripts/start-node-ep.sh agrees, and its NCCL_PROFILE knob is
# where a deliberate override belongs.
ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
