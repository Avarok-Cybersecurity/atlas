# Qwen3.5-397B-A17B-NVFP4 on GB10 — Quickstart (4-node EP=4)

The largest Qwen3.5 MoE checkpoint (397B total / ~17B active, 512 experts,
top-10). The NVFP4 weights are ~200 GB, so this model **only** runs across all
four GB10 nodes in expert-parallel (EP=4, TP=1). Tensor parallelism is not an
option — `num_key_value_heads = 2` cannot shard across 4 TP ranks.

## 1. Build the image (from the repo root)

```bash
docker build -f docker/gb10/qwen3.5-397b-a17b/nvfp4/Dockerfile -t atlas-397b .
```

The builder stage compiles only the `qwen3.5-397b-a17b` kernel target
(`ATLAS_TARGET_MODEL=qwen3.5-397b-a17b`, `ATLAS_TARGET_QUANT=nvfp4`); the runtime
stage carries `libnccl2 (>= 2.28)` + RDMA userspace for the RoCE transport.

Make the image available on all four nodes (build on each, or `docker save | ssh
nodeN docker load`).

## 2. Launch 4-node EP=4

```bash
/home/cluster/launch-atlas-ep4.sh          # orchestrates ranks 0..3 across the 4 nodes
```

The launcher starts rank 0 (HTTP + scheduler) on the head node and ranks 1-3 as
EP workers, passing `--tp-size 1 --ep-size 4 --world-size 4` plus the cluster's
RoCE/NCCL env:

```
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0      # both ConnectX-7 halves
```

Weights are read from the shared HF cache:
`~/.cache/huggingface/hub/models--nvidia--Qwen3.5-397B-A17B-NVFP4`.

## 3. Verify

```bash
curl http://localhost:8888/v1/models
curl http://localhost:8888/v1/chat/completions -d \
  '{"model":"nvidia/Qwen3.5-397B-A17B-NVFP4",
    "messages":[{"role":"user","content":"What is 2+2?"}]}'
```

Rank-0 logs should show the NCCL ring forming (`NCCL INFO Channel X/Y`) and the
kernel target selected as `qwen3.5-397b-a17b`. Thinking is **off by default**
(server default); opt in per request with `thinking_token_budget` — 128 is the
tested sweet spot (256 is non-monotonically worse). MTP is omitted (it regresses
throughput on the NVFP4 checkpoint).

## Notes

- A single-node `docker run` will OOM at the weight-load preflight (weights > 120 GB).
- Driver must stay on 580.x — 590.x triggers a CUDAGraph deadlock on GB10.
- vLLM and Atlas are mutually exclusive on the cluster (both reserve unified memory).
