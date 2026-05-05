# Quickstart

Goal: first successful chat completion in under five minutes, against the flagship **Qwen3.5-35B-A3B** model running at 131 tok/s on a single GB10.

## 1. Start the server

```bash
sudo docker run -d \
  --name atlas-35b \
  --network host --gpus all --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  avarok/atlas-gb10:latest \
  serve Sehyo/Qwen3.5-35B-A3B-NVFP4 \
    --port 8888 \
    --max-seq-len 8192 \
    --kv-cache-dtype nvfp4 \
    --gpu-memory-utilization 0.88 \
    --scheduling-policy slai \
    --speculative \
    --mtp-quantization nvfp4
```

What's happening:

- The container binds the host's network (`--network host`) so port `8888` is reachable on the host directly. `--gpus all` grants the GB10, `--ipc=host` enables shared memory for larger KV buffers.
- `serve <model-id>` selects the model; the binary auto-detects `model_type` from `config.json` and picks the matching kernel target.
- `--kv-cache-dtype nvfp4` keeps the KV cache in 4-bit E2M1 — halves memory vs FP8, with no measurable coherence loss for Qwen3.5.
- `--speculative --mtp-quantization nvfp4` turns on Multi-Token Prediction speculative decoding with the NVFP4 MTP head that ships in the checkpoint. This is the change that takes throughput from ~70 tok/s to ~131 tok/s.
- `--scheduling-policy slai` enables SLO-aware scheduling — prioritises decode steps approaching their TBT deadline.

First start-up takes **2–5 minutes**: the loader reads 15–40 GB of safetensors through the `O_DIRECT` fast path, CUDA graphs are captured for every batch size, and the HTTP server binds. Watch the log:

```bash
sudo docker logs -f atlas-35b
```

You'll see `loaded 125000 tensors in 34s`, then `captured graph for batch=1`, then finally `listening on 0.0.0.0:8888`. The server does **not** accept requests before that last line appears.

## 2. Send a request

Once the server logs `listening`, a standard OpenAI request works:

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "Explain the key idea behind speculative decoding in one paragraph."}],
    "max_tokens": 256
  }'
```

Atlas accepts any `model` string — it serves exactly one model per container, so the field is ignored. Use the real HF id if you want round-tripping through OpenAI clients to feel natural.

## 3. Stream tokens

```bash
curl -sN http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "atlas",
    "messages": [{"role": "user", "content": "Write a short poem about kernels."}],
    "max_tokens": 200,
    "stream": true
  }'
```

Each chunk is a standard `data: {...}` SSE frame with `choices[0].delta.content`. Tool calls stream as `choices[0].delta.tool_calls` chunks in the same format OpenAI emits.

## 4. From Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8888/v1", api_key="unused")

stream = client.chat.completions.create(
    model="atlas",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=200,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

The same client works with Open WebUI — set `Base URL: http://<host>:8888/v1`, API key `sk-dummy`.

## 5. Stop

```bash
sudo docker stop atlas-35b && sudo docker rm atlas-35b
```

## Troubleshooting

- **`error: out of memory`** during start-up — drop `--gpu-memory-utilization` to `0.85`, or `--max-seq-len` to `4096`. 35B has the headroom; it's usually leaked GPU state from a previous container. `nvidia-smi` should show ~0 MB used before starting.
- **Server logs `loaded 0 tensors`** — your HF cache is empty or the path is wrong. Verify with `ls ~/.cache/huggingface/hub/models--Sehyo--Qwen3.5-35B-A3B-NVFP4`.
- **Connection refused on port 8888** — the server hasn't finished initialising. Watch the log; `listening` is the readiness marker.
- **Tokens are gibberish** — almost always a model/loader mismatch. Check that the HF model id in the command line matches the cached directory. If the kernel target the binary picked is wrong (unlikely — Atlas logs it on startup), open an issue; Atlas's house rule is *never blame the model, always find the Atlas bug*.

Next: pick a different model from [Supported Models](./models.md), or dive into the [Architecture](../architecture/philosophy.md).
