# Atlas Spark — GLM-5.3-Flash-NVFP4 on GB10 (2-node, TP=2 + EP=2)
#
# Build from repo root:
#   docker build -f docker/gb10/glm-5.3-flash/nvfp4/Dockerfile -t atlas-glm53:<sha> .
#
# Run — this is a 2-node deployment. The text model is ~176 GiB against a ~121 GB
# node, so a single-node run is PHYSICALLY IMPOSSIBLE and there is no TP=1 config
# to fall back to. The topology is fixed at world=2, TP=2, EP=2:
#
#   TP=2 + EP=2   read/rank 10,665 MB   roofline 23.8 tok/s   88.713 GiB/rank
#   EP=2 only     read/rank 18,952 MB   roofline 13.4 tok/s   (rejected, measured)
#
# GLM is attention-heavy and expert-light (18B active of 320B) — KDA alone is
# 9,366 MB/token of BF16 weights, 2x all routed-expert traffic — so under EP-only
# 87 % of decode bandwidth is replicated. The TP=2 roofline matches our own frozen
# vLLM oracle (24.856 median) and Tony's public 2x Spark run.
#
# Launch both ranks with scripts/launch-atlas-glm53-tp2ep2.sh (rank 0 on n1,
# rank 1 on n2). Verify NET/IB dual-rail RoCE in rank 0's NCCL log before trusting
# ANY number; NET/Socket means the fabric fell back to sockets — STOP.

# ── Build stage ──────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq curl build-essential pkg-config git cmake libclang-dev libibverbs-dev && \
    # Builder NCCL pinned to match the runtime stage, as in the 397B image: not
    # required to link (the devel image's bundled NCCL exports what spark-comm
    # uses, and `-lnccl` is dynamic) but builder == runtime avoids symmetric-memory
    # ABI surprises and keeps the image reproducible.
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
# 🪤 REQUIRED, and every other gb10 Dockerfile is missing it. `atlas-plugin` pulls the
# benchmark prompts in with `include_str!("../../../../tests/fixtures/bench_prompt_*.txt")`,
# so a build without repo-root `tests/` dies at COMPILE time with four "couldn't read"
# errors — after the CUDA kernels have already built, which makes it look like a kernel
# problem. 784 KB.
COPY tests/ tests/

ENV ATLAS_TARGET_HW=gb10
# 🪤 This must be the kernel-target directory name, `glm-5.3-flash` — the same
# string `all_ptx_sets()` matches on. Two of the kernels the composite layer
# resolves (`glm5next_dsa_mla_decode`, `glm5next_mla_latent_write`) live ONLY in
# this target; building any other target yields an image whose GLM layer cannot
# construct, and the failure is at first layer build, not at compile time.
ENV ATLAS_TARGET_MODEL=glm-5.3-flash
ENV ATLAS_TARGET_QUANT=nvfp4

RUN cargo build --release -p spark-server

# ── Runtime stage ────────────────────────────────────────────────
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

# NCCL for the 2-node collectives, RDMA userspace for the RoCE transport.
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
