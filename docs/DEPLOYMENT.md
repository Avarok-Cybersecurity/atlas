# Deploying Atlas

Three deployment modes, in increasing complexity:

1. **Single-GPU Docker** — one model, one node. Easiest for evaluation.
2. **Multi-rank EP=2 / TP=2** — sharded models that don't fit on one GPU.
3. **NVMe-backed swap** — long-context with KV cache eviction to disk.

For end-to-end recipes per supported model see [`QUICKSTART.md`](../QUICKSTART.md).

## 1. Single-GPU Docker

```bash
docker run -d \
  --name atlas \
  --gpus all --ipc=host -p 8888:8888 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve <hf-model-id> \
    --max-seq-len 16384 \
    --max-batch-size 1 \
    --gpu-memory-utilization 0.85
```

Required:
- NVIDIA Container Toolkit installed on the host.
- HuggingFace cache mounted (model weights are pulled on first run).
- `--gpus all` to expose the GPU.
- `--ipc=host` so CUDA shared-memory IPC works.

Then:
```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<hf-model-id>","messages":[{"role":"user","content":"hi"}]}'
```

### Memory tuning

`--gpu-memory-utilization 0.85` reserves 85% of GPU memory for weights + KV
cache. The rest is left for CUDA workspace, NCCL buffers, and OS overhead.
Drop to `0.70` if you see allocation failures during boot; raise to `0.92`
if you want more KV headroom and have nothing else competing for the GPU.

`--max-seq-len 16384` caps the context window. Longer requires either:
- more KV memory (lower batch size)
- NVMe swap (`--high-speed-swap`)
- a smaller model

## 2. Multi-rank EP=2 / TP=2

Models that don't fit on one GB10 (122B-A10B, MiniMax-M2 / M2.7,
Mistral-Small-4, Nemotron-Super-120B) shard across two nodes via NCCL +
RoCEv2 (or fast Ethernet). The two ranks share one OpenAI endpoint exposed
on rank 0.

That is the **two-chassis** case, and everything down to "Critical: MTP /
DFlash flag symmetry" describes it. For **one host with N GPUs** — an
H100/H200/B200 box — skip to [Single node, N
GPUs](#single-node-n-gpus-hopper--b200): the launcher is different and the
GB10 NCCL settings below are a pessimization there.

### Topology options

| Mode | `--tp-size` | `--ep-size` | Use when |
|---|---|---|---|
| Pure EP=2 | 1 | 2 | MoE expert sharding only (122B, MiniMax) |
| Pure TP=2 | 2 | 1 | Dense / attention sharding only (rare) |
| TP+EP overlap | 2 | 2 | Both attention and experts sharded; the two NCCL groups share one comm |

Run via the canonical launcher (single-node default; override with env
for cross-node):

```bash
# Single-node EP=2 (both ranks on this machine)
bash scripts/start-ep2.sh Sehyo/Qwen3.5-122B-A10B-NVFP4

# Cross-node EP=2: head and worker on different machines
HEAD_IP=10.0.0.1 WORKER_IP=10.0.0.2 \
  bash scripts/start-ep2.sh Sehyo/Qwen3.5-122B-A10B-NVFP4
```

What the launcher does:
- Forces `NCCL_SOCKET_IFNAME=enp1s0f0np0` (GB10 RDMA NIC) — change for
  non-DGX-Spark hardware.
- Sets `NCCL_NVLS_ENABLE=0` (GB10 lacks NVLink).
- Sets `NCCL_NET_GDR_LEVEL=0`, `NCCL_NET_GDR_C2C=0`, `NCCL_DMABUF_ENABLE=0`
  (GDS not supported on GB10).
- Pins `NCCL_PROTO=Simple`, `NCCL_ALGO=Ring`.
- Starts rank 0 on `HEAD_IP:8888` and rank 1 on `WORKER_IP:8889`, both
  pointing `--master-addr` to `HEAD_IP:29500`.

### Critical: MTP / DFlash flag symmetry

**Rank 0 and rank 1 must launch with the same `--speculative` / `--mtp-quantization` / `--num-drafts` flags.** Otherwise rank 0's verify
command lands on a layer rank 1 didn't allocate intermediate buffers
for, and you get an SSM intermediate-buffer error.

The launcher mirrors them automatically; if you write your own
two-`docker run` invocation, copy the flags verbatim.

### Single node, N GPUs (Hopper / B200)

`scripts/start-ep2.sh` is for **two GB10 chassis over RoCE**. On a single
H100 / H200 / B200 box with N GPUs, use `scripts/start-node-ep.sh` instead —
it starts N `spark` processes on one host, rank `i` on GPU `i`, bootstrapping
NCCL over `127.0.0.1`.

```bash
# 4×H100, pure EP, local ./target/release/spark, NCCL defaults
NGPUS=4 scripts/start-node-ep.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8

# same, from the published image
NGPUS=4 IMAGE=avarok/atlas-gb10:latest \
  scripts/start-node-ep.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8

# print the commands without launching anything
NGPUS=8 scripts/start-node-ep.sh --dry-run <model>

# stop the ranks this script started (pid files, never `pkill -f`)
scripts/start-node-ep.sh --stop
```

**Which script.**

| You have | Script | Interconnect |
|---|---|---|
| 2 (or 4) DGX Sparks, 1 GB10 each | `start-ep2.sh` | RoCEv2 over `enp1s0f0np0` |
| 2 Sparks, DeepSeek V4-Flash | `start-deepseek-ep2.sh` | RoCEv2, EP-only |
| 1 host, N NVLink GPUs | `start-node-ep.sh` | NVLink / PCIe P2P, intra-node |

#### NCCL profile: start from nothing

`NCCL_PROFILE` selects one of three environments, and the default is the
empty one on purpose.

| `NCCL_PROFILE` | What it sets | When |
|---|---|---|
| `default` | **nothing** | every NVLink run, including all benchmark numbers |
| `debug` | `NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,NET` | first boot on a new box |
| `gb10-roce` | the full `start-ep2.sh` block | only to A/B against the two-Spark deployment |

The GB10 block is a **pessimization on an NVLink node**, not a safe default:

- `NCCL_SOCKET_IFNAME=enp1s0f0np0` and `NCCL_IB_HCA=rocep1s0f0` name a NIC
  that does not exist on an H100/H200/B200 chassis. NCCL then binds the wrong
  interface, or hangs.
- `NCCL_NVLS_ENABLE=0` exists because GB10 has no NVLink (and because of an
  aarch64 Blackwell bug). Setting it on a machine that *does* have NVLink
  throws away NVLink SHARP.
- `NCCL_NET_GDR_LEVEL=0`, `NCCL_NET_GDR_C2C=0`, `NCCL_DMABUF_ENABLE=0` work
  around GB10's missing GDS support. Intra-node transfers do not use the net
  plugin at all, so these only remove options.
- `NCCL_PROTO=Simple`, `NCCL_ALGO=Ring`, `NCCL_MAX_NCHANNELS=2` force the
  slowest protocol/algorithm pair and cap channel count. NCCL's own topology
  detection picks better on one node, every time.

So: **run with `default`, boot once with `debug` to read the `NET/` lines and
confirm which transport NCCL chose, then go back to `default` for anything you
intend to report.** Record the profile in the run's artifact either way.

#### Port layout

Rank `i` is given `--port PORT_BASE+i` (default `8888`, so `8888..8888+N-1`),
but **only rank 0 listens**. That is the code, not a convention: for `rank > 0`
`maybe_run_ep_worker` returns before the HTTP router is ever built
(`crates/spark-server/src/main_modules/serve_load.rs:752`) — an EP worker runs
its command loop and exits when the head does. The distinct ports on workers
only guarantee that a future bind cannot collide. Point every client, and the
readiness poll, at `PORT_BASE`.

`--bind` (default `127.0.0.1`) is passed to rank 0 only. Pass `BIND=0.0.0.0`
if the benchmark client is on another machine — and read the warning in
`--bind`'s help before you do.

#### Topology

The launcher enforces the same rule `resolve_topology` does
(`serve_phases/topology.rs`): `world_size == tp_size × ep_size` (orthogonal
mesh) **or** `world_size == tp_size == ep_size` (overlapping groups). It
refuses a bad combination in the shell rather than letting N processes each
load weights and then bail. `NGPUS=4 EP_SIZE=2 TP_SIZE=1` is rejected;
`EP_SIZE=4 TP_SIZE=1` and `TP_SIZE=2 EP_SIZE=2` are both accepted.

**DeepSeek V4-Flash is EP-only.** The checkpoint has
`num_key_value_heads = 1` (MQA), so there is nothing to shard across TP ranks
and `--tp-size > 1` is impossible. On 8×H200 that means
`NGPUS=8 EP_SIZE=8 TP_SIZE=1`, matching the EP=2 recipe in
`scripts/start-deepseek-ep2.sh` scaled to one node.

#### GPU pinning

Ranks pin with `--gpu-ordinal i`, **not** `CUDA_VISIBLE_DEVICES=i`. The
ordinal already reaches the backend
(`serve_phases/preflight.rs` → `AtlasCudaBackend::new(ordinal)`), and leaving
every GPU visible to every rank keeps NCCL's view of the node complete so it
can pick NVLink/P2P between peers. Masking would also work — the rank would
then need `--gpu-ordinal 0`, since the mask renumbers devices — but it hides
topology for no gain here. Use one mechanism or the other, never both, or
rank `i` lands on the wrong die.

#### Speculative-flag parity

The rule from §2 applies unchanged: **every rank must launch with identical
`--speculative` / `--mtp-quantization` / `--num-drafts` flags**, or rank 0's
verify command lands on a layer the worker never allocated buffers for. A
worker started without them refuses at boot with that message
(`serve_phases/build.rs`). `EXTRA_ARGS` exists for exactly this — it is
appended verbatim to every rank, so parity is structural rather than
remembered:

```bash
NGPUS=4 EXTRA_ARGS="--speculative --mtp-quantization nvfp4 --num-drafts 2" \
  scripts/start-node-ep.sh <model>
```

Keep topology flags out of `EXTRA_ARGS`; the launcher owns those.

#### First boot on a new box

```bash
# 1. Does every kernel resolve? Single-rank, no NCCL, exits with the
#    unresolved count. (The kernel audit runs AFTER the NCCL bootstrap, so a
#    multi-rank --check-kernels would hang waiting for peers instead of
#    reporting — the launcher forces --world-size 1 here.)
scripts/start-node-ep.sh --check-kernels <model>

# 2. First real boot, with NCCL logging, minimal flags.
NGPUS=4 NCCL_PROFILE=debug EXTRA_ARGS="--max-batch-size 1" \
  scripts/start-node-ep.sh <model>

# 3. Then the recipe flags, back on NCCL defaults.
```

Readiness is polled at `http://127.0.0.1:$PORT_BASE/health` once a second
until 200 or `BOOT_TIMEOUT_S` (default 1800 s — the campaign's 30-minute boot
cap). A 503 is a *loading* state, not a failure. On timeout the tail of every
rank log is printed and the ranks are **left running** for inspection unless
you pass `--stop-on-timeout`. Logs and pid files live in
`${ATLAS_NODE_RUN_DIR:-/tmp/atlas-node-ep}`.

The final line prints the exact rank-0 command, ready to paste into a
campaign artifact's `serve_command`.

## 3. NVMe-backed high-speed swap

For long contexts (>32K tokens) the on-device KV cache fills fast. Atlas
can evict cold blocks to NVMe and stream them back as needed:

```bash
# High-speed swap uses io_uring — it REQUIRES the two container flags below
# (--security-opt seccomp=unconfined --ulimit memlock=-1). Without them the
# io_uring setup fails and swap silently does nothing.
docker run -d --gpus all --ipc=host -p 8888:8888 \
  --security-opt seccomp=unconfined --ulimit memlock=-1 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/fast-nvme/atlas-kv:/mnt/fast-nvme/atlas-kv \
  avarok/atlas-gb10:latest \
  serve <model> \
    --max-seq-len 65536 \
    --high-speed-swap \
    --high-speed-swap-cache-blocks-per-seq 64 \
    --high-speed-swap-dir /mnt/fast-nvme/atlas-kv
```

How it works:
- Each sequence keeps a fixed number of "hot" KV blocks on-GPU
  (`--high-speed-swap-cache-blocks-per-seq`, default 64 = 1024 tokens at
  block_size=16).
- Cold blocks evict via `io_uring` async writes through a pinned-host
  bounce buffer (GB10 lacks GDS, so direct NVMe→GPU isn't possible).
- The radix tree tracks `disk_block_id` and reads back on demand when a
  cold block is referenced again.

### Image orientation

EXIF orientation is **applied on decode**. A photo stored sideways with an
`Orientation` tag — which is how most phone cameras write them — is rotated
upright before it reaches the encoder, matching what every ordinary viewer
shows the user. Files with no tag, including all PNGs, are unchanged.

### Optional host dependency: `ffmpeg` (video input only)

**Serving video requires `ffmpeg` on the host.** Text and image serving need
nothing beyond the container. Animated GIF decodes in-process in pure Rust,
but every other container — MP4/MOV, WebM/Matroska, AVI, covering H.264,
H.265, VP9 and AV1 — is decoded by invoking `ffmpeg` as a subprocess.

* Install: `apt install ffmpeg` (Debian/Ubuntu), `dnf install ffmpeg`
  (Fedora/RHEL). The official image ships it.
* Enable: `--video-allow-ffmpeg`. It is **off by default**, because it makes
  the server execute another program per video request.
* Pin a specific build with `--video-ffmpeg-path /usr/bin/ffmpeg`.
* Verify: the startup log says `Video decoding ENABLED via … (ffmpeg version …)`.
  If it says `Video decoding was ENABLED … but the decoder is NOT USABLE`, the
  flag is set and the binary is missing — text and images still serve, every
  video request fails.

The decode is bounded on every axis a caller controls: no shell, no temp
file, capped frame count (`--video-max-frames`), capped output size, and a
wall-clock cap (`--video-decode-timeout-s`) that kills a hung decoder.

Disk requirements:
- **Sequential write bandwidth**: ≥3 GB/s (NVMe gen4 SSD).
- **Free space**: `(num_seqs × max_seq_len × num_layers × kv_dim × 2)` bytes,
  rounded to block size. For Qwen3.6-35B at 64K context with 8 sequences
  ≈ 100 GB.
- **Mount on a different filesystem than `/tmp/atlas-swap/`** — that path
  is for the OS-level CPU swap (`--swap-space-gb`), distinct from
  high-speed swap.

## Health check + observability

```bash
# Liveness
curl http://localhost:8888/health

# Loaded model info
curl http://localhost:8888/v1/models

# Metrics (Prometheus exposition)
curl http://localhost:8888/metrics
```

Logs go to stdout (`docker logs <container>`). The `RUST_LOG` env var
controls verbosity (`info` default, `debug` for kernel call traces, `warn`
for production).

## Kubernetes (community-maintained)

No official manifest yet. The Docker image is self-contained, so a basic
Deployment + Service is sufficient. Open a PR with a working example if
you build one — happy to merge.

## See also

- [`QUICKSTART.md`](../QUICKSTART.md) — copy-paste recipes for each supported model.
- [`docs/GB10_DEPLOYMENT_GUIDE.md`](GB10_DEPLOYMENT_GUIDE.md) — §7 diagnoses multi-rank (EP=2) issues; §2 is the model×quant compatibility matrix; §4 is the OOM / context ladder.
- [`scripts/start-node-ep.sh`](../scripts/start-node-ep.sh) — the single-node,
  N-GPU launcher described above; `scripts/start_node_ep_test.sh` is its
  GPU-free test.
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — what's running inside the binary.
