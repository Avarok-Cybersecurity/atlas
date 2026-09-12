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

_None yet. First JSONL is S0 exit: one already-shipping Atlas MoE vs vLLM on spark1+spark2._
