# SPDX-License-Identifier: AGPL-3.0-only
# K3-only bring-up image. Build natively for the destination host ISA.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder
ARG AVAROK_TARGET_HW
RUN case "$AVAROK_TARGET_HW" in gb10|b300) ;; *) echo 'Set AVAROK_TARGET_HW=gb10 or b300' >&2; exit 1;; esac
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends curl ca-certificates build-essential pkg-config git cmake libclang-dev libibverbs-dev && \
    rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain 1.93.1
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_TOOLCHAIN=1.93.1 CUDA_HOME=/usr/local/cuda CARGO_BUILD_JOBS=4
ENV AVAROK_TARGET_HW=${AVAROK_TARGET_HW} AVAROK_TARGET_MODEL=kimi-k3 AVAROK_TARGET_QUANT=*
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY vendor/ vendor/
COPY kernels/ kernels/
COPY jinja-templates/ jinja-templates/
COPY tests/fixtures/ tests/fixtures/
# Preserve include_str! inputs for K3 compilation/tests; this Dockerfile's
# own ignore file admits these despite the repository-wide docs exclusion.
COPY docs/k3/fixtures/ docs/k3/fixtures/
COPY docs/k3/official-weight-classes.tsv docs/k3/official-weight-classes.tsv
RUN cargo build --locked --release -p spark-server --bin spark \
      --no-default-features --features cuda,nccl
RUN mkdir -p /build/receipt && rustc --version > /build/receipt/rustc.txt && \
    nvcc --version > /build/receipt/nvcc.txt && \
    dpkg-query -W > /build/receipt/builder-packages.txt && \
    sha256sum /build/target/release/spark > /build/receipt/spark.sha256

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3
ARG AVAROK_GIT_SHA=unknown
ARG AVAROK_TARGET_HW
LABEL org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.revision="${AVAROK_GIT_SHA}" \
      org.avarok.target.hardware="${AVAROK_TARGET_HW}"
# Same runtime transport packages/floor as the existing hardware images.
# No Spark-specific NCCL environment is baked into the B300 image.
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
      libnccl2 libibverbs1 librdmacm1 ibverbs-providers libnl-3-200 libnl-route-3-200 && \
    rm -rf /var/lib/apt/lists/* && \
    NCCL_VER=$(dpkg-query -W -f='${Version}' libnccl2) && \
    dpkg --compare-versions "$NCCL_VER" ge "2.28" && \
    mkdir -p /usr/local/share/atlas-k3 && \
    dpkg-query -W > /usr/local/share/atlas-k3/runtime-packages.txt
COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/receipt/ /usr/local/share/atlas-k3/
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE /LICENSE
ENV RUST_LOG=info
EXPOSE 8888
ENTRYPOINT ["spark"]
