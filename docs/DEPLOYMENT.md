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

## 3. NVMe-backed high-speed swap

For long contexts (>32K tokens) the on-device KV cache fills fast. Atlas
can evict cold blocks to NVMe and stream them back as needed:

```bash
docker run ... avarok/atlas-gb10:latest \
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
- [`docs/EP2-TROUBLESHOOTING.md`](EP2-TROUBLESHOOTING.md) — diagnosing multi-rank issues.
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — what's running inside the binary.
