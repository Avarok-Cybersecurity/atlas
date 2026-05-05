# Atlas Spark — Alpha 2

**Image:** `docker pull avarok/atlas-alpha2`
**Hardware:** NVIDIA DGX Spark GB10

## Models

| Model | HuggingFace ID | Type | MTP | tok/s |
|-------|---------------|------|-----|-------|
| **35B MoE** | `Kbenkhaled/Qwen3.5-35B-A3B-NVFP4` | SSM+Attn+MoE | Yes | ~130 |
| **80B MoE** | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | SSM+Attn+MoE | Yes | ~100 |
| **VL-30B** | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` | Attn+MoE (Vision) | No | ~100 |
| **Nemotron-H 30B** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | Mamba-2+MoE+Attn | No | ~100 |
| **27B Dense** | `Kbenkhaled/Qwen3.5-27B-NVFP4` | SSM+Attn (Dense) | No | ~14 |
| **122B MoE** | `Sehyo/Qwen3.5-122B-A10B-NVFP4` | SSM+Attn+MoE | Yes | ~50 |

Download models: `huggingface-cli download <HuggingFace ID>`

## Run Commands

**35B (recommended):**
```bash
sudo docker run -d --name atlas --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-alpha2 serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
    --port 8888 --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.88 \
    --scheduling-policy slai --max-seq-len 8192 --max-batch-size 16 \
    --speculative --mtp-quantization nvfp4
```

**80B:**
```bash
sudo docker run -d --name atlas --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-alpha2 serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
    --port 8888 --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.88 \
    --scheduling-policy slai --max-seq-len 8192 --max-batch-size 16 \
    --speculative --mtp-quantization nvfp4
```

**VL-30B / Nemotron-H / 27B** (no `--speculative`):
```bash
sudo docker run -d --name atlas --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-alpha2 serve <MODEL_ID> \
    --port 8888 --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.88 \
    --scheduling-policy slai --max-seq-len 8192 --max-batch-size 16
```

**122B (single node):**

The 122B model fits on a single GB10 (~77 GB weights). Use `--gpu-memory-utilization 0.80` to leave headroom for the OS. KV cache will be limited (~19 GB), so `--max-seq-len 4096` is recommended.

```bash
sudo docker run -d --name atlas --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-alpha2 serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --port 8888 --kv-cache-dtype nvfp4 --gpu-memory-utilization 0.80 \
    --scheduling-policy slai --max-seq-len 4096 --max-batch-size 1 \
    --speculative --mtp-quantization nvfp4
```

**122B (2x GB10 nodes, EP=2):**

With two GB10 nodes connected via RoCE InfiniBand, each node loads half the experts (128 each). This frees more memory for KV cache and allows `--max-seq-len 8192`.

Prerequisites: passwordless SSH from head to worker, model weights on both nodes, RDMA support (`/dev/infiniband`), `avarok/atlas-alpha2` pulled on both nodes.

```bash
HEAD_IP="<your-head-ip>"     # rank 0 node
WORKER_IP="<your-worker-ip>" # rank 1 node
MODEL="Sehyo/Qwen3.5-122B-A10B-NVFP4"
IMAGE="avarok/atlas-alpha2"

RDMA_FLAGS="--device=/dev/infiniband --cap-add=IPC_LOCK --ulimit memlock=-1"
NCCL_ENV="-e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=0 -e NCCL_NET_GDR_LEVEL=0 \
  -e NCCL_PROTO=Simple -e NCCL_ALGO=Ring \
  -e NCCL_MIN_NCHANNELS=1 -e NCCL_MAX_NCHANNELS=2 -e NCCL_DEBUG=INFO"

# Rank 0 (head — runs HTTP server)
sudo docker run -d --name atlas-ep0 --gpus all --ipc=host --network host \
  $RDMA_FLAGS $NCCL_ENV \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  $IMAGE serve $MODEL \
    --rank 0 --world-size 2 --master-addr $HEAD_IP --master-port 29500 \
    --port 8888 --max-batch-size 1 --gpu-memory-utilization 0.55 \
    --kv-cache-dtype nvfp4 --speculative --mtp-quantization nvfp4

# Rank 1 (worker — run via SSH)
ssh $WORKER_IP "sudo docker run -d --name atlas-ep1 --gpus all --ipc=host --network host \
  $RDMA_FLAGS $NCCL_ENV \
  -v \\\${HOME}/.cache/huggingface:/root/.cache/huggingface \
  $IMAGE serve $MODEL \
    --rank 1 --world-size 2 --master-addr $HEAD_IP --master-port 29500 \
    --port 0 --max-batch-size 1 --gpu-memory-utilization 0.55 \
    --kv-cache-dtype nvfp4 --speculative --mtp-quantization nvfp4"
```

Wait ~5 minutes for both ranks to load weights and complete NCCL init. Check with `curl http://$HEAD_IP:8888/health`.

## Test

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
       "messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

## API

OpenAI-compatible: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`

Supports: streaming, `enable_thinking`, vision inputs (VL-30B), `content: [array]` and `content: null`.

## What's New (vs Alpha 1)

- Concurrent batching (`--max-batch-size 16`) with SLO-aware scheduling
- Marconi SSM prefix caching (4x TTFT on multi-turn)
- NVFP4 KV cache (44% less memory vs FP8)
- Paged flash attention for chunked prefill
- Conv1d+L2norm fused kernel
