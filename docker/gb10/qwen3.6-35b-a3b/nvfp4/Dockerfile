# Atlas Spark — Qwen3.6-35B-A3B-FP8 (Dense Hybrid SSM+Attn) on GB10
#
# Build from repo root:
#   docker build -f docker/gb10/qwen3.6-35b-a3b/nvfp4/Dockerfile -t atlas-gb10:qwen3.6-35b-a3b .
#
# Run (target only, no DFlash):
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-gb10:qwen3.6-35b-a3b serve Qwen/Qwen3.6-35B-A3B-FP8 \
#       --max-seq-len 16384
#
# Run (with DFlash drafter pairing):
#   docker run --gpus all --ipc=host -p 8888:8888 \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-gb10:qwen3.6-35b-a3b serve Qwen/Qwen3.6-35B-A3B-FP8 \
#       --dflash --draft-model z-lab/Qwen3.6-35B-A3B-DFlash --max-seq-len 16384
#
# Run (proven agentic webserver_ok config — dgx2, 10/10 webserver_ok + 10/10
#      followed_directions, opencode Axum ping/pong suite, Σwall ~1650s):
#   docker run --gpus all --ipc=host --network host \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     atlas-gb10:qwen3.6-35b-a3b serve Qwen/Qwen3.6-35B-A3B-FP8 \
#       --model-name qwen36-35b-a3b --max-seq-len 32768 --max-batch-size 2 \
#       --gpu-memory-utilization 0.70 --kv-cache-dtype bf16 \
#       --lm-head-dtype bf16 --scheduling-policy slai \
#       --speculative --num-drafts 1 --mtp-quantization bf16 \
#       --enable-prefix-caching --ssm-cache-slots 256 --ssm-checkpoint-interval 16
#   Notes: default tool-call parser (qwen3_coder) — do NOT pass --tool-call-parser
#   hermes (mangles multi-line file writes, #37). --enable-prefix-caching is the
#   master switch for BOTH KV reuse and Marconi SSM-snapshot restore on this
#   attn+GDN hybrid. lm_head bf16 (NVFP4 lm_head collapses). fp8 KV cache is a
#   ~3x decode REGRESSION on SM121 here — keep KV at bf16.

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
COPY vendor/ vendor/

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=qwen3.6-35b-a3b
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
COPY README.md /README.md

ENV RUST_LOG=info
EXPOSE 8888

ENTRYPOINT ["spark"]
