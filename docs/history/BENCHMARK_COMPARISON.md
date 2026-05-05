# Atlas vs vLLM — Qwen3.5-35B-A3B-NVFP4 on DGX Spark (GB10)

Single request, batch=1. Same model, same hardware, same benchmark script.

## Atlas (MTP K=2)

| Workload | ISL/OSL | TPOT p50 | tok/s |
|---|---:|---:|---:|
| Summarization short | 1024/128 | 8.99ms | 111.2 |
| RAG / document QA | 8192/1024 | 10.82ms | 92.5 |
| Short chat | 256/256 | 8.01ms | 124.8 |
| Standard chat | 1024/1024 | 8.31ms | 120.3 |
| Code generation | 128/1024 | 8.32ms | 120.2 |
| Long reasoning | 1024/8192 | 10.08ms | 99.2 |

## vLLM (no speculative decoding available)

| Workload | ISL/OSL | TPOT p50 | tok/s |
|---|---:|---:|---:|
| Summarization short | 1024/128 | 26.36ms | 37.9 |
| RAG / document QA | 8192/1024 | 27.17ms | 36.8 |
| Short chat | 256/256 | 26.62ms | 37.6 |
| Standard chat | 1024/1024 | 26.69ms | 37.5 |
| Code generation | 128/1024 | 26.99ms | 37.1 |
| Long reasoning | 1024/8192 | CRASH | |

> vLLM's engine dies after a few requests due to CUTLASS TMA grouped GEMM failures on SM120/SM121 (GB10), tracked upstream as [vllm#33857](https://github.com/vllm-project/vllm/issues/33857). MTP speculative decoding is not available in vLLM for this model.
> Used DGX "de facto standard" from [Eugr](https://github.com/eugr/spark-vllm-docker/tree/main) 

## Head-to-head

| Workload | Atlas tok/s | vLLM tok/s | Speedup |
|---|---:|---:|---:|
| Summarization short | 111.2 | 37.9 | 2.9x |
| RAG / document QA | 92.5 | 36.8 | 2.5x |
| Short chat | 124.8 | 37.6 | 3.3x |
| Standard chat | 120.3 | 37.5 | 3.2x |
| Code generation | 120.2 | 37.1 | 3.2x |
| Long reasoning | 99.2 | CRASH | — |
| **Average** | **111.4** | **37.5** | **3.0x** |
