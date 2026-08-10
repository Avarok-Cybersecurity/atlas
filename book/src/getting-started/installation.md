# Installation

Atlas ships as a single Docker image that contains the release binary plus every compiled `(GB10, model, quant)` PTX module — **22 target sets** today, one per `kernels/gb10/<model>/<quant>/` directory. There is no "install Atlas + download kernels" step — the kernels are baked in.

## Hardware prerequisites

Atlas is designed for broad hardware support — the engine is vendor-agnostic above the kernel layer (`ComputeTarget` at build time, `GpuBackend` at runtime, `CommBackend` for collectives) and new hardware plugs in at the trait layer. The first shipped target is **NVIDIA GB10 (SM121)** — the Grace-Blackwell Superchip in the NVIDIA DGX Spark workstation. To run the shipped image you need:

- A DGX Spark (or any GB10-based system) with 119.7 GB of unified GPU memory
- NVIDIA driver supporting CUDA 13.0 or later
- `docker` with `--gpus all` support (recent `nvidia-container-toolkit`)
- Internet access for the first model download; models are cached under `~/.cache/huggingface` after that

Other NVIDIA GPUs (H100, B200) and other vendors (AMD, Apple, Intel) are on the roadmap rather than in the shipped image. The PTX that ships today is compiled with `-arch=sm_121` using SM121-specific tile shapes and a software E2M1 conversion — none of that is architectural, it's just the first target we hyperoptimized. Adding a new hardware target is two trait impls plus kernel source; the [Adding a new hardware target](https://github.com/Avarok-Cybersecurity/atlas/blob/main/README.md#adding-a-new-hardware-target) guide in the README walks through an Apple Metal example end to end.

## Pull the image

```bash
docker pull avarok/atlas-gb10:latest
```

The image contains the Rust release binary, all 22 PTX module sets, tokenizer dependencies, and the `nvidia-container-runtime` library surfaces. No Python, no CUDA toolkit.

## Bring your own weights

Atlas loads HuggingFace `safetensors` directly. The image does **not** ship model weights. On first run, the binary resolves a HuggingFace model ID (e.g. `Sehyo/Qwen3.5-35B-A3B-NVFP4`) against `~/.cache/huggingface/hub` — download the weights once with the `hf` CLI or let the server download-on-miss:

```bash
pip install -U huggingface_hub
hf download Sehyo/Qwen3.5-35B-A3B-NVFP4
```

The command is `hf`, not `huggingface-cli`: `huggingface_hub` 1.0 renamed the
binary, and on 1.16+ the old name is gone entirely. Older docs and scripts still
say `huggingface-cli download …`, which now fails with "command not found".

Mount the cache directory into the container:

```bash
-v ~/.cache/huggingface:/root/.cache/huggingface
```

## Build from source (optional)

You only need to build from source if you are modifying Atlas. The `rust-toolchain.toml` pins `stable`; CUDA 13.0+ with `nvcc` on `PATH` (or `CUDA_HOME` set) is required for a real build. Clippy and fmt can run without CUDA via `ATLAS_SKIP_BUILD=1`.

```bash
git clone https://github.com/Avarok-Cybersecurity/atlas.git
cd atlas

# Full build — compiles every (gb10, model, quant) target (~6 min)
docker build -f docker/gb10/Dockerfile -t atlas-gb10 .

# Rust-only check (no CUDA). CUDARC_CUDA_VERSION is needed alongside
# ATLAS_SKIP_BUILD: without it cudarc's build script shells out to
# `nvcc --version` and panics on a host that has no CUDA toolkit.
# This pair is exactly what ci.yml exports. Deny-warnings comes from
# [workspace.lints], so `-- -Dwarnings` is not needed and CI does not pass it.
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo clippy --workspace --tests
cargo fmt --all -- --check

# Unit tests (uses MockGpuBackend; no GPU required)
cargo test --release

# Integration tests (require GPU + weights)
cargo test -p spark-server --release -- --ignored
```

The build system reads `kernels/gb10/HARDWARE.toml` for architecture flags, enumerates every `(model, quant)` subdirectory that matches the `ATLAS_TARGET_*` wildcards, compiles each `.cu` source file through `nvcc`, and emits a single `target_ptx.rs` that the `atlas-kernels` crate embeds in the final binary. Zero runtime compilation.

## Verify the install

```bash
docker run --rm --gpus all avarok/atlas-gb10:latest --version
# → spark 1.0.0-beta-preview   (the workspace version in Cargo.toml)
docker run --rm --gpus all avarok/atlas-gb10:latest --help | head -20
```

If `--version` errors with "no compatible GPU", the `nvidia-container-toolkit` is not picking up the device. Check `docker info | grep -i runtime` and `nvidia-smi` on the host.

You are now ready for the [Quickstart](./quickstart.md).
