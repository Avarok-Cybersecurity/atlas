# Atlas Spark — All models on GB10 (multi-target binary)
#
# Build from repo root:
#   docker build -f docker/gb10/Dockerfile -t atlas-gb10 .
#
# Run (auto-selects kernels based on model config):
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-gb10 serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4
#
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-gb10 serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
# Override rust-toolchain.toml's 1.85 pin: a transitive dep (libloading 0.9)
# now requires >= 1.88. Using `stable` defers to the toolchain that rustup
# installed above (currently 1.93).
ENV RUSTUP_TOOLCHAIN=stable

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=*
ENV ATLAS_TARGET_QUANT=*

RUN cargo build --release -p spark-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3

LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

# Stamp the source commit into the image so ":latest" provenance is answerable:
#   docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' <tag>
# Passed by scripts/release.sh and the atlas-release skill via --build-arg ATLAS_GIT_SHA.
ARG ATLAS_GIT_SHA=unknown
LABEL org.opencontainers.image.revision="${ATLAS_GIT_SHA}"

# NCCL for multi-GPU (EP=2), RDMA userspace for RoCE transport.
# NCCL ≥ 2.28 is required for `ncclMemAlloc`/`ncclMemFree` symmetric memory
# windows (substrate for device-API fused allreduce kernels). The base
# `nvidia/cuda:13.0.0-runtime-ubuntu24.04` image has the NVIDIA CUDA APT
# repo enabled, which carries libnccl2 2.29+ (>=2.28 satisfied). Install
# the latest available, then verify the resolved version meets the floor.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    # Sanity-check the linked NCCL is ≥ 2.28 — fail-fast if a future repo
    # change downgrades the available version below the symmetric-memory floor.
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" >&2; exit 1; }

COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE /LICENSE
COPY README.md /README.md

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
