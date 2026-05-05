# Atlas Spark — Quickstart Guide

Run state-of-the-art language models on a single NVIDIA DGX Spark (GB10).

**Docker image:** `avarok/atlas-gb10:latest`
**API:** OpenAI-compatible (`/v1/chat/completions`, `/v1/models`)
**Default port:** 8888

---

## Prerequisites

**For the Docker quick-start path (most users):**

- NVIDIA DGX Spark GB10 (119.7 GB GPU memory)
- Docker with `--gpus all` support (the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- HuggingFace cache at `~/.cache/huggingface` (models download automatically on first run)

**For building from source (contributors):**

```bash
# Ubuntu 22.04 / 24.04
sudo apt-get update && sudo apt-get install -y \
    build-essential pkg-config git cmake clang libclang-dev
# CUDA 13.0 toolkit must already be installed; `nvcc --version` should report 13.0.
# The vendored xgrammar-rs build fetches https://github.com/mlc-ai/xgrammar.git
# at build time — set `XGRAMMAR_SRC_DIR=/path/to/local/clone` for air-gapped builds.
```

The first `cargo build --release -p spark-server` takes ~15-30 minutes (PTX
compilation across the supported model targets dominates). Expect ~3-5 GB
under `target/`.

---

## Network exposure

By default Atlas binds the HTTP listener to **`127.0.0.1` (localhost only)**.
This is the right default for an inference engine that talks to local
clients (Open WebUI, opencode, an OpenAI SDK pinned to
`base_url=http://localhost:8888/v1`).

**To accept connections from elsewhere on your network**, pass
`--bind 0.0.0.0` *and* enable bearer-token auth:

```bash
spark serve <model> \
  --bind 0.0.0.0 \
  --require-auth \
  --auth-tokens-file /etc/atlas/tokens.txt
```

Where `tokens.txt` is one bearer token per line (`#` comments allowed),
permissions `chmod 600`. Clients then send `Authorization: Bearer <token>`
on every request, exactly as they would to `api.openai.com`. For a
single-token quick demo, `--auth-token sk-demo-…` works but the token
leaks via `ps`/`/proc/<pid>/cmdline` — prefer the file form in production.

The server logs a warning when binding to `0.0.0.0` so this isn't a silent
exposure. The CORS layer is permissive (`Allow-Origin: *`) by default; if
you bind to `0.0.0.0` from a developer machine, treat the auth gate as
required, not optional.

---

## First-request cold-start

The server marks itself ready as soon as weights are loaded, but the
**first** `/v1/chat/completions` request will pause for **5-30 seconds**
while:

1. CUDA graphs are captured for the prevailing batch shape.
2. The kernel autotuner records timings and selects launch configs.
3. The radix prefix cache and SSM snapshot index initialize.

This is normal. Subsequent requests are 3-5× faster. If you see a stall on
the first request, *don't* Ctrl-C — wait the full minute. To eliminate
the cold-start entirely, pass `--warmup-prompt /path/to/prompt.txt`; the
server prefills it at startup and inserts the resulting KV/SSM state into
the prefix cache, so the first real request hits a warm graph.

---

## Models

| Model | Params | Active | Arch | Tok/s | MTP | Max Context |
|-------|--------|--------|------|-------|-----|-------------|
| Qwen3.5-27B | 27B | 27B (dense) | SSM+Attn | 14 | No | 8K |
| Qwen3-VL-30B | 30B | 3B (MoE) | Attn+MoE+Vision | 97 | No | 32K |
| Nemotron-H 30B | 30B | 3B (MoE) | Mamba-2+MoE+Attn | 98 | No | 8K |
| Qwen3.5-35B | 35B | 3B (MoE) | SSM+MoE+Attn | 133 | K=2 | 8K |
| Qwen3-Next-80B | 80B | 3B (MoE) | SSM+MoE+Attn | 104 | K=2 | 8K |
| Qwen3.5-122B | 122B | 10B (MoE) | SSM+MoE+Attn | 51 | K=2 | 4K |

Tok/s = server decode throughput (p50, ISL=128, concurrency=1, OSL=128).
122B shown with EP=2 (two nodes). Single-node 122B is also possible — see below.

---

## 1. Qwen3.5-27B Dense (27B params, 14 tok/s)

Dense hybrid SSM+Attention model. No MoE, no MTP. Best for quality-sensitive tasks
where throughput is less critical.

```bash
sudo docker run -d \
  --name atlas-27b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Kbenkhaled/Qwen3.5-27B-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai
```

---

## 2. Qwen3-VL-30B Vision (30B MoE, 97 tok/s)

Vision-language model. Accepts images in OpenAI content-parts format.
Pure attention (no SSM), no MTP support.

```bash
sudo docker run -d \
  --name atlas-vl-30b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4 \
    --port 8888 \
    --max-seq-len 32768 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai
```

**Send an image:**
```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64_DATA>"}}
      ]
    }],
    "max_tokens": 512
  }'
```

---

## 3. Nemotron-H 30B (30B MoE, 98 tok/s)

NVIDIA's hybrid Mamba-2 + MoE + Attention model with strong reasoning.
No MTP support.

```bash
sudo docker run -d \
  --name atlas-nemotron \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai
```

---

## 4. Qwen3.5-35B MoE + MTP (35B MoE, 133 tok/s)

Fastest model in the catalogue. MTP speculative decoding (K=2) gives +35% over
baseline. 3B active parameters per token.

```bash
sudo docker run -d \
  --name atlas-35b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai \
    --speculative \
    --mtp-quantization nvfp4
```

---

## 5. Qwen3-Next-80B MoE + MTP (80B MoE, 104 tok/s)

Largest single-node MoE model with MTP. Hybrid SSM+Attention+MoE architecture.
3B active parameters per token.

```bash
sudo docker run -d \
  --name atlas-80b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai \
    --speculative \
    --mtp-quantization nvfp4
```

---

## 6. Qwen3.5-122B MoE — Single Node (122B MoE, ~50 tok/s)

All 256 experts on one GB10. Weights are ~76 GB NVFP4, leaving ~33 GB for KV cache,
SSM state, and buffers. Must use `max-batch-size 1` and `max-seq-len 4096`.

```bash
sudo docker run -d \
  --name atlas-122b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --port 8888 \
    --max-seq-len 4096 \
    --max-batch-size 1 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai \
    --speculative \
    --mtp-quantization nvfp4
```

> **Note:** Single-node 122B has not been benchmarked yet. Expected ~50 tok/s based on
> EP=2 results. Memory is tight — if OOM occurs, lower `--gpu-memory-utilization` to 0.85
> or reduce `--max-seq-len` to 2048.

---

## 7. Qwen3.5-122B MoE — EP=2 (two DGX Sparks, 51 tok/s)

Expert Parallelism across two GB10 nodes connected via InfiniBand RoCE.
128 experts per node, NCCL all-reduce between ranks.

> **⚠ Critical EP=2 constraint:** the worker (rank > 0) **must** be started
> with the same `--speculative --mtp-quantization <value> --num-drafts <N>`
> flags as the head, otherwise the head's MTP verify command will land in
> the worker's SSM layer with no intermediate buffers allocated and you'll
> see an SSM intermediate-buffer error. The commands below mirror the flags
> correctly — copy them verbatim.

Replace `<HEAD_IP>` below with the IP of your head node (rank 0). Both
nodes need passwordless SSH between them and a fast interconnect (RoCE
or 100GbE+).

**Node 0 (head, `<HEAD_IP>`):**
```bash
sudo docker run -d \
  --name atlas-122b-r0 \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --port 8888 \
    --rank 0 --world-size 2 \
    --master-addr <HEAD_IP> --master-port 29500 \
    --max-seq-len 4096 \
    --max-batch-size 1 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.70 \
    --scheduling-policy slai \
    --speculative \
    --mtp-quantization nvfp4
```

**Node 1 (worker):**
```bash
sudo docker run -d \
  --name atlas-122b-r1 \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --port 8889 \
    --rank 1 --world-size 2 \
    --master-addr <HEAD_IP> --master-port 29500 \
    --max-seq-len 4096 \
    --max-batch-size 1 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.70 \
    --scheduling-policy slai \
    --speculative \
    --mtp-quantization nvfp4
```

> Start both nodes. They handshake via NCCL on `master-addr:master-port`.
> The API is served from Node 0 only (`http://<HEAD_IP>:8888`).

---

## Common Operations

### Test the API

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_ID",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Stream responses

```bash
curl -sN http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_ID",
    "messages": [{"role": "user", "content": "Explain quantum computing."}],
    "max_tokens": 500,
    "stream": true
  }'
```

### Enable thinking mode (chain-of-thought)

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_ID",
    "messages": [{"role": "user", "content": "Solve: 23 * 47"}],
    "max_tokens": 1024,
    "enable_thinking": true
  }'
```

### List models

```bash
curl http://localhost:8888/v1/models
```

### View logs

```bash
sudo docker logs -f <container-name>
```

### Stop and remove

```bash
sudo docker stop <container-name> && sudo docker rm <container-name>
```

---

## OpenAI SDK / Open WebUI

Atlas Spark exposes a standard OpenAI-compatible API. Use it with any client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8888/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

For **Open WebUI**: set the base URL to `http://<spark-ip>:8888/v1` and any API key.

---

## Notes

- **First startup** downloads the model from HuggingFace (~15-40 GB depending on model).
  Subsequent runs use the cached weights from `~/.cache/huggingface`.
- **Startup time** is typically 2-5 minutes (weight loading + CUDA graph compilation).
- **GPU memory** should be clean before starting. Check with `nvidia-smi`.
- **Tool calling** is supported via `--tool-call-parser hermes` (Qwen models) or
  `--tool-call-parser qwen3_coder` (Nemotron-H). See the Tool Calling section below.
- **NVFP4 quantization** uses 4-bit E2M1 weights with FP8 block scales. All models in
  this guide use NVFP4 for both weights and KV cache.

---

## Tool Calling

Add `--tool-call-parser <FORMAT>` to enable OpenAI-compatible function calling.

| Parser | Models |
|--------|--------|
| `hermes` | Qwen3-VL-30B, Qwen3-Next-80B (Qwen3 family — JSON format) |
| `qwen3_coder` | Qwen3.5-27B, Qwen3.5-35B, Nemotron-H 30B, Qwen3.5-122B (Qwen3.5 family — XML format) |

**Example: start 35B with tool calling enabled:**
```bash
sudo docker run -d \
  --name atlas-35b-tools \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Kbenkhaled/Qwen3.5-35B-A3B-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai \
    --speculative \
    --mtp-quantization nvfp4 \
    --tool-call-parser hermes
```

**Call a tool:**
```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 512
  }'
```

**Response with tool call:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_00000000",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"Paris\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

**Send tool result back (multi-turn):**
```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kbenkhaled/Qwen3.5-35B-A3B-NVFP4",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"},
      {"role": "assistant", "content": null, "tool_calls": [{"id": "call_00000000", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}}]},
      {"role": "tool", "tool_call_id": "call_00000000", "name": "get_weather", "content": "{\"temperature\": 15, \"condition\": \"cloudy\"}"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 512
  }'
```

**Tool choice options:**
- `"tool_choice": "auto"` — model decides (default)
- `"tool_choice": "none"` — disable tool calling for this request
- `"tool_choice": "required"` — force the model to call a tool
- `"tool_choice": {"type": "function", "function": {"name": "get_weather"}}` — force specific function

Streaming is fully supported — tool calls are detected in the token stream and emitted
as `delta.tool_calls` chunks in real-time.
