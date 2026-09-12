# Bake-off harness — Atlas K3

Lab comparison protocol. Numbers from spark1/spark2 dummy runs are **comms-bound**. Do not write them into certified `.benchmarks/*/BASELINE.json`. Do not compare them to published GB300-NVL72 vLLM K3 numbers.

Pin the rental vLLM K3 image the week you book. Lab vLLM on Sparks reuses the existing Spark / spark-recipes pins, not a K3 image.

## Endpoints

| engine | host | URL |
| --- | --- | --- |
| Atlas | spark1 | `http://spark1:8888/v1` (bind `0.0.0.0`; address not in git) |
| Atlas worker | spark2 | `http://spark2:8889/v1` (not the client endpoint) |
| vLLM (lab proxy) | spark2 | `http://spark2:8000/v1` |

## Fixed protocol

- Same weights (dummy or official)
- Same tokenizer
- `temperature=0`, `max_tokens` fixed
- Warmup 3 requests, then N=32
- ISL/OSL pairs: 512/128, 2048/256, 8192/256
- Concurrency 1 and 8 (8 only if mem allows)

## JSONL columns

```
engine, hardware, model, isl, osl, concurrency, ttft_p50_ms, ttft_p99_ms, itl_p50_ms, tok_s_per_user, tok_s_system, gpu_mem_gb, notes
```

One JSON object per line. Script: `bench_openai.py` (S0 deliverable; not in this commit).

## Forbidden

- Comparing spark1/spark2 Atlas dummy to published 16× GB300 vLLM K3 numbers
- Mixing speculative decode on one engine only without a second row
- Writing lab dummy numbers into certified `.benchmarks/*/BASELINE.json`

## Results

S0 think-off (2026-09-11), **not certified**, not a K3 kernel number. Sibling Qwen3.8-27B NVFP4 trees. Atlas `--dangerously-allow-unresolved-kernel-lookups`. vLLM `--attention-backend TRITON_ATTN --enforce-eager`.

`docs/k3/logs/bakeoff-thinkoff-2026-09-11.jsonl`

| engine | host | isl/osl | C | e2e p50 ms | tok/s (osl/e2e) | sample |
| --- | --- | --- | --- | --- | --- | --- |
| atlas | spark1 | 128/32 | 1 | 2032 | 15.7 | `ping` + 1..8 |
| vllm | spark2 | 128/32 | 1 | 2579 | 12.4 | `ping` + 1..8 |

Without `enable_thinking=false`, vLLM emits a think preamble and the row is not comparable. Default FlashInfer `plan()` 19 vs 20 args — see RST sheet.
