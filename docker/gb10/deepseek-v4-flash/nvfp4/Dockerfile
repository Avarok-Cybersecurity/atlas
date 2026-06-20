# Atlas Spark — DeepSeek-V4-Flash-NVFP4 on GB10
#
# Build from repo root:
#   docker build -f docker/gb10/deepseek-v4-flash/nvfp4/Dockerfile -t atlas-deepseek-v4 .
#
# Run (single node):
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-deepseek-v4 serve deepseek-ai/DeepSeek-V4-Flash
#
# Run (EP=2 — see scripts/start-deepseek-ep2.sh):
#   docker run --gpus all --ipc=host --network host \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-deepseek-v4 serve deepseek-ai/DeepSeek-V4-Flash \
#       --rank 0 --world-size 2 --master-addr <head-node-ip> \
#       --kv-cache-dtype bf16
#
# Note: DeepSeek-V4 uses num_key_value_heads=1 (MQA), so TP>1 is
# architecturally impossible.  Multi-spark deployments MUST use
# pure EP (tp-size 1, ep-size 2/4/...) or EP-only topology.

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /build

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY kernels/ kernels/

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=deepseek-v4-flash
ENV ATLAS_TARGET_QUANT=nvfp4

RUN cargo build --release -p spark-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

# NCCL for multi-GPU (EP=2), RDMA userspace for RoCE transport
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

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
