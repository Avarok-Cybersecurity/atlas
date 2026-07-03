# Atlas reproducible BUILD environment — pins every native/FFI dependency so a
# `cargo build` "just works" without remembering CUTLASS_HOME / FLASHINFER_HOME /
# CUDA-13.2 / the GDN AOT libs. This is the env behind the hand-built
# `atlas-holo:cuda13.2-fp4test` image, captured as code.
#
# The recurring failures this fixes:
#   • "CUTLASS support was not built; set CUTLASS_HOME" — build.rs silently drops
#     the native NVFP4 GEMM (`atlas_cutlass` cfg) when CUTLASS_HOME is unset.
#   • FlashInfer FA2 ragged-prefill wrapper needs FLASHINFER_HOME + its PINNED CCCL.
#   • GDN-FlashInfer (ATLAS_GDN_FLASHINFER=1) needs libatlasgdn.so + the CuTe-DSL
#     runtime + the cuda-13.2 compat driver for sm_121a.
#
# Two ways to use it:
#   1. As a BUILD SANDBOX (mount the repo, run any cargo cmd — all env preset):
#        docker build -f docker/gb10/Dockerfile.builder --target builder -t atlas-gb10:build .
#        docker run --rm --gpus all -v "$PWD":/build -w /build atlas-gb10:build \
#          cargo build --release -p spark-model --example nvfp4_gemm_bench \
#            --no-default-features --features "cuda gpu-examples"
#   2. As a full SERVE image (compiles spark-server, bundles the GDN runtime):
#        docker build -f docker/gb10/Dockerfile.builder -t atlas-gb10:cuda13.2-fp4 .
#
# Pinned versions (override with --build-arg):
ARG CUDA_VER=13.2.0
ARG CUTLASS_SHA=cf064d2e6bad2886238ac565b3b49007764f4939
ARG FLASHINFER_SHA=a671c02ee2fbcdde7cc991f5a01c7cf5eb4a8972
# CuTe-DSL runtime (libcute_dsl_runtime.so) for the GDN AOT kernel — matches the
# floor pinned by FlashInfer @ ${FLASHINFER_SHA} (requirements.txt: >=4.5.0, cu13
# extra). Bump deliberately and re-validate gdn_fla_vs_fi after.
ARG CUTLASS_DSL_VER=4.5.0

# ── Build stage: toolchain + all pinned native deps ──────────────────────────
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu24.04 AS builder
ARG CUTLASS_SHA
ARG FLASHINFER_SHA
ARG CUTLASS_DSL_VER

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      curl ca-certificates build-essential pkg-config git cmake libclang-dev \
      python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Rust (stable — overrides rust-toolchain.toml's 1.85 pin; libloading 0.9 needs >=1.88).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_TOOLCHAIN=stable
ENV CUDA_HOME=/usr/local/cuda

# CUTLASS (header-only; build.rs compiles cutlass_nvfp4_gemm.cu against it).
ENV CUTLASS_HOME=/opt/cutlass
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git ${CUTLASS_HOME} && \
    git -C ${CUTLASS_HOME} checkout ${CUTLASS_SHA}

# FlashInfer + its PINNED CCCL (libcudacxx/cub) for the FA2 ragged-prefill wrapper.
# build.rs puts $FLASHINFER_HOME/3rdparty/cccl/{libcudacxx/include,cub} ahead of
# the CUDA-13 toolkit CCCL via -isystem (toolkit CCCL lacks cuda::fast_mod_div).
ENV FLASHINFER_HOME=/opt/flashinfer
RUN git clone --filter=blob:none https://github.com/flashinfer-ai/flashinfer.git ${FLASHINFER_HOME} && \
    git -C ${FLASHINFER_HOME} checkout ${FLASHINFER_SHA} && \
    git -C ${FLASHINFER_HOME} submodule update --init --depth 1 3rdparty/cccl

# CuTe-DSL runtime for the GDN AOT kernel (provides libcute_dsl_runtime.so).
# Discover its location and expose it on the linker/loader path.
RUN pip3 install --no-cache-dir --break-system-packages "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VER}" || true; \
    CUTE_RT=$(find / -name libcute_dsl_runtime.so 2>/dev/null | head -1); \
    if [ -n "$CUTE_RT" ]; then ln -sf "$CUTE_RT" /usr/local/lib/libcute_dsl_runtime.so && echo "cute runtime: $CUTE_RT"; \
    else echo "WARN: libcute_dsl_runtime.so not found; stubbing (FI-GDN disabled)" && : > /usr/local/lib/libcute_dsl_runtime.so; fi
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/compat:${LD_LIBRARY_PATH}
ENV CUTE_DSL_ARCH=sm_121a

WORKDIR /build

# ── Optional: compile a release spark-server (skip when used as a build sandbox) ─
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/
COPY 3rdparty_patches/ 3rdparty_patches/

ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=*
ENV ATLAS_TARGET_QUANT=*
# Native FP4 GEMM + cuBLASLt BF16 prefill projections on by default (matches prod).
ENV ATLAS_CUTLASS_NVFP4_GEMM=1

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends libnccl2 libnccl-dev && rm -rf /var/lib/apt/lists/*
ENV CUDARC_CUDA_VERSION=12000
ENV RUSTFLAGS="-L /usr/local/cuda/lib64 -L /usr/local/cuda/targets/sbsa-linux/lib -C link-arg=-lcudart"
RUN cargo build --release -p spark-server

# Re-link the GDN AOT shared lib from committed artifacts (gdn_holo_0.o is the
# AOT-exported bf16 kernel; gdn_transpose.o is the k<->v state transpose). No
# python/torch needed here — the .o is pre-exported and version-controlled.
RUN cd 3rdparty_patches/gdn_aot && \
    { nvcc -arch=sm_121a -Xcompiler -fPIC -c gdn_transpose.cu -o gdn_transpose.o && \
      g++ -O2 -fPIC -shared gdn_shim.cpp gdn_transpose.o gdn_holo_0.o \
        -o /usr/local/lib/libatlasgdn.so \
        -I. -I/usr/local/cuda/include -lcudart \
        -L/usr/local/lib -lcute_dsl_runtime -Wl,-rpath,/usr/local/lib; } || \
    { echo "WARN: libatlasgdn build failed; stubbing (FI-GDN disabled)" && : > /usr/local/lib/libatlasgdn.so; }

# ── Runtime stage: serve image on CUDA 13.2 + GDN runtime bundled ─────────────
FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu24.04
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"

# NCCL >= 2.28 (ncclMemAlloc symmetric-memory windows) + RDMA userspace.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" || \
      { echo "ERROR: NCCL $NCCL_VER < 2.28" >&2; exit 1; }

COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
# GDN-FlashInfer runtime libs (only loaded when ATLAS_GDN_FLASHINFER=1).
COPY --from=builder /usr/local/lib/libatlasgdn.so /usr/local/lib/libatlasgdn.so
COPY --from=builder /usr/local/lib/libcute_dsl_runtime.so /usr/local/lib/libcute_dsl_runtime.so
COPY LICENSE /LICENSE
COPY README.md /README.md

ENV RUST_LOG=info
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/compat:/usr/local/cuda/lib64
ENV CUTE_DSL_ARCH=sm_121a
# GDN-FlashInfer is opt-in (FLA recurrence is the validated default).
ENV ATLAS_GDN_FLASHINFER=0
EXPOSE 8888
ENTRYPOINT ["spark"]
