# K3 bring-up container

From the repository root, build **on the destination CPU architecture**:

```bash
# ARM64 DGX Spark host, GB10 device code.
docker build --no-cache --platform linux/arm64 -f docker/k3/Dockerfile \
  --build-arg AVAROK_TARGET_HW=gb10 \
  --build-arg AVAROK_GIT_SHA="$(git rev-parse HEAD)" -t atlas-k3:gb10-prep .

# Native x86_64 builder for an x86_64 B300 rental.
docker build --no-cache --platform linux/amd64 -f docker/k3/Dockerfile \
  --build-arg AVAROK_TARGET_HW=b300 \
  --build-arg AVAROK_GIT_SHA="$(git rev-parse HEAD)" -t atlas-k3:b300-prep .
```

An ARM64 Spark executable is **not portable to an x86_64 rental**, even if its
embedded PTX targets B300. Build the rental image natively; setting `--platform`
on a different host may invoke slow emulation and is not evidence of a native
build. The required hardware argument accepts only `gb10` or `b300`. Only K3's
registered BF16/MXFP4/NVFP4 targets compile; names do not prove each format works.

Both CUDA 13 base images are pinned by digest and Rust is pinned to 1.93.1.
Cargo honors the lockfile. APT resolves current packages, so retain the built
image ID/digest and package receipts; these commands are not fully reproducible
byte-for-byte dependency pins. Receipts live at `/usr/local/share/atlas-k3/`.
The source revision label describes HEAD; disclose a dirty source tree and
retain its patch/source archive instead of presenting the label as a clean pin.

This image contains no model weights or credentials. Mount the separately
verified checkpoint read-only and use the reviewed rank map and explicit serve
arguments from the [launch harness](../../scripts/k3/LAUNCH.md). Keep the host's
matching driver/NVIDIA Container Toolkit and required network devices available.
Do not copy dual-Spark RoCE overrides onto a single-node NVSwitch rental.
A container build or `serve --help` is not a GPU inference receipt. Start with
a bounded small-fixture canary before attempting official TP8 weights.
