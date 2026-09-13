#!/bin/bash
# Build a GLM-5.3 iteration image and put it on BOTH ranks.
#
# 🪤 `RUSTFLAGS="-L .ncclstub"` is not optional. Only NCCL *stubs* exist on this host (the real
# library lives inside the container), so without it the link fails with
# `/usr/bin/ld: cannot find -lnccl`. Changing RUSTFLAGS invalidates the whole build cache, so
# always build through this script rather than a bare `cargo build`.
#
# 🪤 The three ATLAS_TARGET_* vars pick which kernel set is compiled in. Unset, the build prints
# "not set. Hardware targets available: ..." and silently takes target 0 — another model's
# kernels, in a binary that still runs.
#
# 🪤 Each node builds its OWN one-layer image from the same binary rather than `docker save |
# docker load` — the base is ~10 GB and only the COPY layer differs. Never INFER the base: pass
# BASE explicitly or read it off the running container (`docker ps --format '{{.Image}}'`).
# A stale base silently changes what is in the image below your one changed layer.
set -euo pipefail
TAG="${1:?usage: build-atlas-glm53-image.sh <tag>   e.g. t48}"
BASE="${BASE:-atlas-glm53:t47}"
NODES="${NODES:-10.10.10.1 10.10.10.2}"

cd "$(dirname "$0")/.."
RUSTFLAGS="-L $PWD/.ncclstub" \
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server

for n in $NODES; do
  echo "=== $n"
  sudo -n -u cluster ssh -o BatchMode=yes "cluster@$n" "rm -rf ~/glm53-build && mkdir -p ~/glm53-build"
  sudo -n -u cluster scp -q -o BatchMode=yes target/release/spark "cluster@$n:~/glm53-build/spark"
  sudo -n -u cluster ssh -o BatchMode=yes "cluster@$n" "cd ~/glm53-build && \
    printf 'FROM %s\nCOPY spark /usr/local/bin/spark\n' '$BASE' > Dockerfile && \
    docker build -q -t 'atlas-glm53:$TAG' ."
done
echo "atlas-glm53:$TAG built on: $NODES   (base $BASE)"
