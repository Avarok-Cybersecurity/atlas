# Atlas Docker Guide

Atlas provides per-model Dockerfiles organized by `(Hardware, Model, Quantization)` tuple.

## Directory Structure

```
docker/
  gb10/                              # NVIDIA GB10 (DGX Spark), sm_121f
    Dockerfile                       # all models, one multi-target binary
    qwen3-next-80b-a3b/
      nvfp4/
        Dockerfile                   # 80B model, NVFP4 quantization
    qwen3.5-35b-a3b/
      nvfp4/
        Dockerfile                   # 35B model, NVFP4 quantization
  hopper/                            # NVIDIA H100 / H200, sm_90a
    Dockerfile
  b200/                              # NVIDIA B200 / GB200, sm_100a
    Dockerfile
```

The three hardware sets are not interchangeable. Each image carries PTX for
one architecture and `spark`'s arch preflight refuses to start on any other
GPU — see [Hopper / B200 images](#hopper--b200-images) below.

## Prerequisites

- NVIDIA GPU with CUDA 13.0+ drivers
- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- Model weights downloaded via the `hf` CLI (`pip install -U huggingface_hub`; the
  binary was called `huggingface-cli` before huggingface_hub 1.0 and no longer exists)

### Download Model Weights

Use `--local-dir` to download weights as real files (no symlinks). This is the recommended approach for Docker — it avoids broken symlinks when mounting volumes.

```bash
# 80B model (~47 GB)
hf download nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  --local-dir /models/qwen3-next-80b

# 35B model (~20 GB) — both base + extra MTP weights
hf download Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
  --local-dir /models/qwen3.5-35b
```

## Build

All builds run from the **repository root**:

```bash
# 80B model
docker build -f docker/gb10/qwen3-next-80b-a3b/nvfp4/Dockerfile -t atlas-80b .

# 35B model
docker build -f docker/gb10/qwen3.5-35b-a3b/nvfp4/Dockerfile -t atlas-35b .
```

Build takes ~2-3 minutes (Rust compilation + CUDA kernel PTX compilation).

## Run

### Recommended: `--model-from-path` with local directory

This is the most portable approach — mount the model directory and pass the path directly.

```bash
# 80B with speculative decoding (~106 tok/s counting, ~99 tok/s diverse)
docker run --gpus all --ipc=host -p 8888:8888 \
  -v /models/qwen3-next-80b:/model \
  atlas-80b serve --model-from-path /model --speculative --num-drafts 1

# 35B with speculative decoding (~131 tok/s counting, ~127 tok/s diverse)
docker run --gpus all --ipc=host -p 8888:8888 \
  -v /models/qwen3.5-35b:/model \
  atlas-35b serve --model-from-path /model --speculative --num-drafts 1
```

### Non-speculative mode

```bash
# 80B (~82 tok/s)
docker run --gpus all --ipc=host -p 8888:8888 \
  -v /models/qwen3-next-80b:/model \
  atlas-80b serve --model-from-path /model

# 35B (~102 tok/s)
docker run --gpus all --ipc=host -p 8888:8888 \
  -v /models/qwen3.5-35b:/model \
  atlas-35b serve --model-from-path /model
```

### Alternative: HuggingFace cache mount

If you use the default HuggingFace cache (`~/.cache/huggingface/`), mount it and pass the model ID:

```bash
docker run --gpus all --ipc=host -p 8888:8888 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-80b serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 --speculative --num-drafts 1
```

> **Note:** The 35B model's `extra_weights.safetensors` is a symlink that may break with HF cache mounts. Use `--local-dir` download or `--model-from-path` instead.

## Hopper / B200 images

`docker/hopper/Dockerfile` (H100 / H200, `sm_90a`) and
`docker/b200/Dockerfile` (B200 / GB200, `sm_100a`) are the datacentre
counterparts of `docker/gb10/Dockerfile`. They exist so that **renting the GPU
and building on it are two different things**: `nvcc --ptx -arch=sm_90a` is a
cross-compile and the kernels ship as PTX embedded in the binary, so both
images build end to end on any x86_64 Linux host with Docker and no NVIDIA
hardware at all. The rented box only pulls.

### Build

From the repository root, on any x86_64 Docker host:

```bash
docker build -f docker/hopper/Dockerfile -t atlas-hopper:latest .
docker build -f docker/b200/Dockerfile   -t atlas-b200:latest .
```

Roughly an hour cold — the Rust release build plus one nvcc invocation per
kernel per model target. To build one model instead of all five:

```bash
docker build -f docker/hopper/Dockerfile \
  --build-arg ATLAS_TARGET_MODEL=nemotron-super-120b-a12b \
  -t atlas-hopper:nemotron .
```

Build args: `ATLAS_TARGET_HW` (defaults to `hopper` / `b200` per file),
`ATLAS_TARGET_MODEL` (`*` by default — `deepseek-v4-flash`,
`nemotron-3-nano-30b-a3b`, `nemotron-super-120b-a12b`, `qwen3.6-35b-a3b`,
`qwen3-next-80b-a3b`), `ATLAS_TARGET_QUANT` (`nvfp4`), and `ATLAS_GIT_SHA`
(stamped into `org.opencontainers.image.revision`).

Neither image builds the optional CUTLASS or FlashInfer side objects
(`CUTLASS_HOME` / `FLASHINFER_HOME` are deliberately unset). The CUTLASS
NVFP4 wrappers are gated on SM120/SM121 support and compile to nothing for
both of these architectures, so there is nothing to gain;
`docker/gb10/Dockerfile.builder` is the image that wires them up.

### First boot: `--check-kernels`

Run this before serving anything, and before timing anything. It resolves
every kernel the model needs against the PTX actually compiled into the
binary, prints the ones it cannot resolve, and exits with that count:

```bash
docker run --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-hopper:latest \
  serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --check-kernels --no-tui
```

A non-zero exit names kernels that are missing for this architecture. Some
absences are expected and declared: `kernels/<hw>/qwen3.6-35b-a3b/MODEL.toml`
lists the two W4A4 MoE entry points under `[expected_absent.moe_w4a16]`,
because the warp-level block-scaled MMA they use exists on neither `sm_90a`
nor `sm_100a`. The `[expected_absent]` tables in these trees were harvested on
GB10 and have **not** been re-harvested on datacentre silicon, so the first
real `--check-kernels` run on an H100 or a B200 is also the thing that
validates them.

### Serve

Single GPU:

```bash
docker run --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-hopper:latest \
  serve nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 --no-tui
```

N GPUs on one node — use the launcher rather than N hand-written `docker run`
lines. It pins rank `i` to GPU `i` and ships **no** NCCL environment, which is
the correct configuration on an NVLink box:

```bash
NGPUS=4 IMAGE=atlas-hopper:latest \
  scripts/start-node-ep.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8

NGPUS=8 IMAGE=atlas-b200:latest \
  scripts/start-node-ep.sh deepseek-ai/DeepSeek-V4-Flash
```

Neither image bakes in any `NCCL_*` variable. `scripts/start-ep2.sh`'s block
is tuned for two GB10 chassis over RoCE and is actively wrong here: it names a
NIC these machines do not have, disables NVLink SHARP, and forces the slowest
protocol/algorithm pair onto an intra-node transport. `start-node-ep.sh`'s
`NCCL_PROFILE` knob is where a deliberate override belongs.

Both runtime stages enforce **NCCL >= 2.28** at build time (`ncclMemAlloc` /
`ncclMemFree` symmetric-memory windows); the build fails rather than shipping
an image that would lose them.

### These images are architecture-locked, on purpose

`atlas-hopper` carries `sm_90a` PTX and nothing else; `atlas-b200` carries
`sm_100a` and nothing else. Point either at the wrong GPU and `spark`'s arch
preflight refuses to start, before the driver would. **That is the feature.**
PTX built for an `a`-suffixed architecture does not run forward, and the two
Blackwell architectures are siblings rather than a ladder — `sm_100a` has
tcgen05 and `redux.sync.max.abs.f32`, `sm_120a`/`sm_121` have the warp-level
`mma … .kind::mxf4nvf4.block_scale` that `sm_100a` lacks. B300 / GB300 are
`sm_103a` and are served by neither image. The measured table is in the B200
section of [`docs/HARDWARE.md`](../docs/HARDWARE.md).

Hopper is also **FP8 / BF16 only**: it has no NVFP4 datapath, and
`kernels/hopper/<model>/nvfp4/` is a directory name rather than a promise —
an NVFP4-built kernel bundle also serves FP8 and BF16 checkpoints, which is
why the directory is called that.

### Just want the binary?

`.github/workflows/datacenter-binaries.yml` builds the same `spark` on a
GitHub-hosted runner and uploads `spark-hopper-x86_64` / `spark-b200-x86_64`.
See [`docs/DEPLOYMENT.md`](../docs/DEPLOYMENT.md#a-prebuilt-binary-for-a-rental-box).

## API

Atlas serves an OpenAI-compatible API on the configured port.

```bash
# Check server status
curl http://localhost:8888/v1/models

# Chat completion
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'

# Streaming
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256,
    "stream": true
  }'
```

## Serve Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-from-path` | — | Direct filesystem path to model weights |
| `--port` | `8888` | HTTP listening port |
| `--max-seq-len` | `32768` | Maximum sequence length (tokens) |
| `--gpu-memory-utilization` | `0.90` | GPU memory fraction (0.0-1.0) |
| `--speculative` | `false` | Enable MTP speculative decoding |
| `--num-drafts` | `1`, then per-model | Draft tokens per speculative step (K = N+1). While the value is still `1`, the model's `MODEL.toml` `default_num_drafts` replaces it — `3` on qwen3.6-27b |
| `--max-batch-size` | `8` | Max concurrent sequences per decode step |
| `--kv-cache-dtype` | `fp8` | KV cache precision — `bf16`, `fp8`, `nvfp4`, `turbo2/3/4/8`, plus nine asymmetric K/V pairings |

## Performance (NVIDIA GB10 / DGX Spark)

| Model | Mode | Counting | Diverse |
|-------|------|:--------:|:-------:|
| **35B** | Speculative (K=2) | **131 tok/s** | **127 tok/s** |
| **35B** | Non-speculative | 102 tok/s | 102 tok/s |
| **80B** | Speculative (K=2) | **106 tok/s** | **99 tok/s** |
| **80B** | Non-speculative | 82 tok/s | 82 tok/s |

## Troubleshooting

### "No MTP weights found" with speculative decoding
The 35B model stores MTP weights in `extra_weights.safetensors`. If using HF cache mounts, the file may be a broken symlink. Fix: download with `--local-dir` or use `--model-from-path`.

### Model not found
Ensure the model path is correctly mounted inside the container. With `--model-from-path`, the path must be valid **inside** the container (not the host path).

### Out of memory
- Lower `--gpu-memory-utilization` (e.g., `0.85`)
- Reduce `--max-seq-len` (e.g., `2048`)

### Slow startup
Normal — model loading takes 30-90 seconds depending on model size and storage speed.
