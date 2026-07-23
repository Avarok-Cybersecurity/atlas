# Atlas Spark — Qwen3.5-397B-A17B-NVFP4 on GB10 (4-node EP=4)
#
# Build from repo root:
#   docker build -f docker/gb10/qwen3.5-397b-a17b/nvfp4/Dockerfile -t atlas-397b .
#
# Run — this is a 4-node deployment. The NVFP4 weights (~200 GB) exceed a single
# GB10's 120 GB, and num_key_value_heads=2 cannot shard across 4 TP ranks, so the
# only viable topology is expert-parallel EP=4 / TP=1 across all four nodes:
#
#   /home/cluster/launch-atlas-ep4.sh          # orchestrates ranks 0..3 on the 4 nodes
#
# (launch-atlas-ep4.sh passes `--tp-size 1 --ep-size 4 --world-size 4` plus the
# RoCE/NCCL env — NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 — to each rank.) A
# single-node `docker run` will OOM at weight-load preflight.

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev libibverbs-dev && \
    # Pin the builder's NCCL to match the runtime stage (2.30.4). Not required
    # to link — the devel image's bundled 2.27.7 already exports the symbols
    # spark-comm uses (ncclMemAlloc/ncclMemFree, verified) and `-lnccl` is a
    # dynamic link — but keeping builder == runtime avoids any symmetric-memory
    # (NCCL >= 2.28) ABI surprises and keeps the image reproducible.
    apt-get install -y -qq --allow-change-held-packages libnccl-dev libnccl2 && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=qwen3.5-397b-a17b
ENV ATLAS_TARGET_QUANT=nvfp4

RUN cargo build --release -p spark-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

# NCCL for multi-node EP=4, RDMA userspace for the RoCE transport.
# NCCL >= 2.28 is required for ncclMemAlloc/ncclMemFree symmetric memory.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers \
      libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    echo "Linked NCCL: $NCCL_VER" && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      (echo "ERROR: NCCL $NCCL_VER < 2.28 — symmetric memory APIs unavailable" && exit 1)

COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY LICENSE /LICENSE
COPY README.md /README.md

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
