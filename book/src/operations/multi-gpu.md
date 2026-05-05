# Multi-GPU & EP=2

Expert Parallelism across two GB10 nodes is the only way to run the largest MoE models (Qwen3.5-122B-A10B, Mistral-Small-4-119B, MiniMax-M2.7) — their experts don't fit on one node. Atlas's multi-GPU support is specifically EP=2 over RoCEv2; the scheduler and HTTP API run on rank 0.

## What "EP=2" means here

Two GB10 nodes connected by InfiniBand or RoCE. The model's MoE experts are split 50/50 (128 experts per node for a 256-expert model). Every other layer (attention, SSM, dense FFN, embeddings, LM head) is replicated on both ranks.

Per decode step:

1. Both ranks run the attention / SSM / dense layers on their local portion of the batch.
2. At each MoE layer, the gate runs on both ranks; top-k expert IDs are computed.
3. Tokens destined for remote experts are sent via `reduce_scatter`.
4. Local experts compute.
5. Results `all_gather` back.
6. Continue.

Only rank 0 runs the HTTP server and the scheduler. Rank 1 is a silent compute worker that joins at startup via NCCL rendezvous.

## Network layer

Atlas's production two-node setup uses InfiniBand RoCE over a Mellanox ConnectX HCA (`mlx5_0`). The two-node network is dedicated — the public/management interface is separate. Canonical IPs:

- Head: `<head-ip>`
- Worker: `<worker-ip>`

If you don't have InfiniBand, EP=2 works over plain Ethernet with a 3–4× throughput penalty. GB10's EP=2 numbers in this book assume RoCE.

## Launching — the canonical scripts

`scripts/start-ep2.sh` handles 99% of deployments. Defaults to Qwen3.5-122B-A10B-NVFP4. Usage:

```bash
# Default model
bash scripts/start-ep2.sh

# Explicit model
bash scripts/start-ep2.sh Sehyo/Qwen3.5-122B-A10B-NVFP4

# MiniMax (script auto-strips --speculative since MiniMax doesn't have MTP)
bash scripts/start-ep2.sh lukealonso/MiniMax-M2.7-NVFP4
```

On each node, the script does:

1. Cleans any stale containers.
2. Sets the NCCL + GLOO env vars (see below) for RoCE.
3. Runs `docker run ... atlas-gb10:latest serve <model> --rank {0|1} --world-size 2 --master-addr <head-ip> --master-port 29500 ...`.

## Manual launch

If you need custom flags, the manual flow is:

**Head (rank 0, node 0, <head-ip>):**

```bash
sudo docker run -d --name atlas-122b-r0 \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_HCA=mlx5_0 -e NCCL_IB_DISABLE=0 \
  -e NCCL_NET_GDR_LEVEL=5 -e NCCL_NVLS_ENABLE=0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  avarok/atlas-gb10:latest \
  serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --port 8888 \
    --rank 0 --world-size 2 \
    --master-addr <head-ip> --master-port 29500 \
    --max-seq-len 4096 \
    --max-batch-size 1 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.70 \
    --scheduling-policy slai \
    --speculative --mtp-quantization nvfp4
```

**Worker (rank 1, node 1, <worker-ip>):**

```bash
sudo docker run -d --name atlas-122b-r1 \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IB_HCA=mlx5_0 -e NCCL_IB_DISABLE=0 \
  -e NCCL_NET_GDR_LEVEL=5 -e NCCL_NVLS_ENABLE=0 \
  -e GLOO_SOCKET_IFNAME=enp1s0f0np0 \
  avarok/atlas-gb10:latest \
  serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --port 8889 \
    --rank 1 --world-size 2 \
    --master-addr <head-ip> --master-port 29500 \
    --max-seq-len 4096 \
    --max-batch-size 1 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.70 \
    --scheduling-policy slai \
    --speculative --mtp-quantization nvfp4
```

Start both; they rendezvous on `<head-ip>:29500`. Only the head serves HTTP.

## The critical MTP-flag symmetry rule

The worker **must** be started with the same `--speculative --mtp-quantization <value> --num-drafts <N>` flags as the head. Otherwise the head's MTP verify command arrives at the worker's SSM layer with no intermediate buffers allocated, and the step crashes with an SSM intermediate-buffer error.

`start-ep2.sh` enforces this. Manual launches where only the head has MTP enabled have bit multiple contributors; the in-code EP=2 MTP guard (wave-5) now produces a clearer error, but the rule is: **head and worker flags must match for speculative, num-drafts, and mtp-quantization**.

## NCCL env vars — what matters on GB10

| Variable | Value | Why |
|---|---|---|
| `NCCL_SOCKET_IFNAME=enp1s0f0np0` | the RoCE interface | A mis-set ifname falls back to the 1 GbE mgmt interface, dropping throughput 10× |
| `NCCL_IB_HCA=mlx5_0` | the Mellanox HCA | Pins the transport |
| `NCCL_IB_DISABLE=0` | IB enabled | Default, but explicit is safer |
| `NCCL_NET_GDR_LEVEL=5` | GPUDirect RDMA | Bypasses the host bounce buffer |
| `NCCL_NVLS_ENABLE=0` | NVLink-SHARP off | SHARP crashes GB10 — **mandatory** off |
| `GLOO_SOCKET_IFNAME=enp1s0f0np0` | same | Gloo fallback paths need the same ifname |

These are baked into the `scripts/start-*ep2.sh` scripts. If you bring up EP=2 from scratch, carry them forward — a missing `NCCL_NVLS_ENABLE=0` will crash silently mid-warmup.

## Testing

`scripts/test-minimax-ep2.sh` is the canonical end-to-end harness. It runs:

- Coherence check (3 prompts, 3 tokens each, verifies byte-identical output across 3 repeats)
- Fibonacci generation (tests long-output coherence)
- Tool calls (tests the EP=2 + tool-call + MTP interaction)
- TPS benchmark (tests decode throughput)

Last known-green run on alpha-2.35: MiniMax-M2.7 EP=2 scored 8/10 on the suite; the Qwen3.5-122B EP=2 equivalent hits ~46 tok/s on 600-token decodes with the flags shown above.

## Performance reference

| Model | EP=2 decode tok/s | Notes |
|---|---:|---|
| Qwen3.5-122B-A10B | 46 | MTP K=2, NVFP4, 600-tok sustained |
| Mistral-Small-4-119B | 33 | No MTP, NVFP4 |
| MiniMax-M2.7 | — | Achieved full PASS; throughput varies |

## Troubleshooting

- **Rendezvous timeout** — `master-addr` unreachable from worker, or `master-port` blocked by firewall.
- **NCCL stuck at "all ranks ready"** — `NCCL_SOCKET_IFNAME` wrong on one side. Both ranks need the *same* ifname (both `enp1s0f0np0`).
- **Silent NaN after first request** — `NCCL_NVLS_ENABLE=1` leaked through. Force off.
- **SSM intermediate buffer error on worker** — MTP flags mismatched between head and worker. See the symmetry rule above.
- **Health endpoint on head returns 200 but requests hang** — worker failed to come up and the head's initial barrier is blocking. Check the worker's logs.

## Why not tensor parallel?

Tensor parallelism (TP) was the traditional approach for splitting dense models across GPUs. Atlas on GB10 does not use TP because:

1. GB10 is unified memory — there's no NVLink island to exploit for intra-node TP.
2. The models Atlas targets are MoE-dominant beyond a single GB10's memory. EP is the natural split — one expert per rank is cleaner than splitting a weight matrix.
3. TP requires per-layer all-reduces. EP requires per-MoE-layer token dispatch. At the model shapes we care about, EP is lower collective traffic than TP.

TP for dense models on future multi-GPU hardware is on the roadmap; today there's no supported use case where it would win.

## Files to read

- `scripts/start-ep2.sh`, `scripts/start-minimax-ep2.sh`, `scripts/test-minimax-ep2.sh`
- `crates/spark-comm/src/nccl_backend.rs` — NCCL impl.
- `crates/spark-model/src/layers/minimax_moe.rs` (EP=2 dispatch example).
- `kernels/gb10/minimax-m2-229b/nvfp4/moe_dispatch.cu` — kernel.
- `docs/design/ep2-token-dispatch-design.md` — design note.
- `docs/EP2-TROUBLESHOOTING.md` — field-tested troubleshooting.
