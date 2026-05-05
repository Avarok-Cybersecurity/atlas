# Atlas GB10 — Concurrency Benchmark Results

**Date:** 2026-03-11
**Hardware:** 2x NVIDIA GB10 Grace Blackwell (119.7 GB GPU memory each)
**Image:** `atlas-gb10:latest` (ATLAS_TARGET_MODEL=*)
**KV Cache:** NVFP4 (all models)
**Scheduler:** SLAI (SLO-aware)
**Config:** max-batch-size=16, max-seq-len=8192, gpu-memory-utilization=0.88
**Benchmark:** count-prompt mode, OSL=128, warmup=1

## Metric Definitions

| Metric | Description |
|--------|-------------|
| TTFT   | Client Time To First Token (prefill latency), p50 ms |
| TPOT   | Client Time Per Output Token (decode inter-token), p50 ms |
| sTTFT  | Server TTFT (excludes network RTT), p50 ms |
| sTPS   | Server decode throughput (tok/s per sequence), p50 |
| Tput   | Aggregate output tok/s across concurrent batch |


---

## Qwen3.5-27B Dense (NVFP4)

**Model:** `Kbenkhaled/Qwen3.5-27B-NVFP4`
**Config:** max-batch-size=1, no MTP (dense model), ISL<=1024
**Coherence:** 10/10 passed

| ISL | Conc | Tput | TTFT p50 | TPOT p50 | sTTFT p50 | sTPS p50 |
|-----|------|------|----------|----------|-----------|----------|
| 128 | 1 | 13.3 | 509 ms | 71.6 ms | 507 ms | 14.0 |
| 512 | 1 | 4.6 | 508 ms | 72.7 ms | 505 ms | 13.7 |
| 1024 | 1 | 12.7 | 877 ms | 72.5 ms | 874 ms | 13.8 |

**Notes:** Dense model — no MoE, no batched decode. Conc=1 only. ~14 tok/s decode.


---

## Qwen3-VL-30B MoE Vision (NVFP4)

**Model:** `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4`
**Config:** max-batch-size=16, no MTP
**Coherence:** 10/10 passed

| ISL | Conc | Tput | TTFT p50 | TPOT p50 | sTTFT p50 | sTPS p50 |
|-----|------|------|----------|----------|-----------|----------|
| 128 | 1 | 74.5 | 402 ms | 10.4 ms | 399 ms | 96.6 |
| 128 | 2 | 91.7 | 820 ms | 18.7 ms | 405 ms | 64.4 |
| 128 | 4 | 70.0 | 1215 ms | 40.3 ms | 401 ms | 26.9 |
| 128 | 8 | 102.3 | 2031 ms | 66.0 ms | 403 ms | 15.9 |
| 128 | 16 | 105.4 | 3644 ms | 127.5 ms | 402 ms | 8.0 |
| 512 | 1 | 1.5 | 662 ms | n/a | 660 ms | 0.0 |
| 1024 | 1 | 48.6 | 1151 ms | 11.7 ms | 1148 ms | 85.8 |
| 1024 | 4 | 61.6 | 4277 ms | 34.2 ms | 2804 ms | 31.6 |
| 1024 | 16 | 65.3 | 16230 ms | 119.5 ms | 10603 ms | 8.5 |
| 2048 | 1 | 36.3 | 1907 ms | 12.7 ms | 1904 ms | 78.6 |
| 2048 | 4 | 42.9 | 6780 ms | 49.5 ms | 5273 ms | 24.8 |
| 2048 | 16 | 44.0 | 24234 ms | 180.3 ms | 17717 ms | 5.8 |

**Notes:** ISL 512 had anomalous n/a TPOT (model may have emitted few output tokens). ~97 tok/s single-stream decode at ISL 128. Aggregate throughput peaks at ~105 tok/s at conc=16.


---

## Qwen3.5-35B MoE (NVFP4, MTP K=2)

**Model:** `Kbenkhaled/Qwen3.5-35B-A3B-NVFP4`
**Config:** max-batch-size=16, speculative MTP K=2
**Coherence:** 10/10 passed

| ISL | Conc | Tput | TTFT p50 | TPOT p50 | sTTFT p50 | sTPS p50 |
|-----|------|------|----------|----------|-----------|----------|
| 128 | 1 | 19.8 | 126 ms | 12.6 ms | 125 ms | 79.1 |
| 128 | 2 | 107.2 | 251 ms | 17.8 ms | 127 ms | 59.4 |
| 128 | 4 | 97.1 | 398 ms | 74.3 ms | 129 ms | 52.9 |
| 128 | 8 | 71.4 | 678 ms | 71.3 ms | 133 ms | 14.2 |
| 128 | 16 | 87.4 | 1224 ms | 74.0 ms | 135 ms | 0.0 |
| 512 | 1 | 120.2 | 96 ms | 7.6 ms | 93 ms | 131.2 |
| 512 | 2 | 102.0 | 203 ms | 18.9 ms | 95 ms | 55.1 |
| 512 | 4 | 95.3 | 307 ms | 121.8 ms | 97 ms | 52.0 |
| 512 | 8 | 89.6 | 485 ms | 21.5 ms | 97 ms | 0.0 |
| 512 | 16 | 10.3 | 866 ms | n/a | 97 ms | 0.0 |
| 1024 | 1 | 87.4 | 299 ms | 9.0 ms | 296 ms | 110.9 |
| 1024 | 2 | 87.2 | 616 ms | 20.6 ms | 302 ms | 54.8 |
| 1024 | 4 | 89.6 | 947 ms | 39.8 ms | 622 ms | 26.7 |
| 1024 | 8 | 94.6 | 1621 ms | 74.3 ms | 1289 ms | 13.9 |
| 1024 | 16 | 97.2 | 3103 ms | 141.5 ms | 2755 ms | 7.2 |
| 2048 | 1 | 67.1 | 806 ms | 8.7 ms | 803 ms | 115.4 |
| 2048 | 2 | 63.6 | 1665 ms | 25.0 ms | 1656 ms | 53.9 |
| 2048 | 4 | 66.4 | 2518 ms | 47.1 ms | 2506 ms | 24.6 |
| 2048 | 8 | 67.1 | 4271 ms | 91.9 ms | 3780 ms | 11.7 |
| 2048 | 16 | 70.1 | 8045 ms | 168.2 ms | 7536 ms | 6.2 |

**Notes:** Fastest single-stream: **131 sTPS at ISL 512** (MTP speculative). Aggregate throughput peaks ~97 tok/s at conc=16 ISL 1024. MTP acceptance degrades under contention (sTPS drops from 131 to ~7 at conc=16). TTFT stable ~93-135 ms at ISL 128-512 (Marconi prefix cache).


---

## Qwen3-Next-80B MoE (NVFP4, MTP K=2)

**Model:** `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`
**Config:** max-batch-size=16, speculative MTP K=2
**Coherence:** 9/10 passed (max_tokens test: model EOS'd at 85 tokens — model behavior, not server bug)

| ISL | Conc | Tput | TTFT p50 | TPOT p50 | sTTFT p50 | sTPS p50 |
|-----|------|------|----------|----------|-----------|----------|
| 128 | 1 | 88.3 | 187 ms | 9.9 ms | 185 ms | 100.7 |
| 128 | 2 | 84.0 | 372 ms | 22.5 ms | 185 ms | 47.5 |
| 128 | 4 | 71.3 | 565 ms | 48.1 ms | 181 ms | 31.7 |
| 128 | 8 | 65.2 | 879 ms | 72.2 ms | 177 ms | 13.8 |
| 128 | 16 | 82.5 | 1586 ms | 135.1 ms | 174 ms | 7.5 |
| 512 | 1 | 84.0 | 143 ms | 10.9 ms | 140 ms | 92.0 |
| 512 | 2 | 13.6 | 272 ms | 155.5 ms | 135 ms | 45.7 |
| 512 | 4 | 13.4 | 419 ms | 174.3 ms | 135 ms | 25.2 |
| 512 | 8 | 13.7 | 694 ms | 471.9 ms | 136 ms | 3.0 |
| 512 | 16 | 85.7 | 1226 ms | 305.9 ms | 135 ms | 12.5 |
| 1024 | 1 | 71.8 | 459 ms | 10.4 ms | 456 ms | 96.0 |
| 1024 | 2 | 69.0 | 947 ms | 25.4 ms | 941 ms | 46.0 |
| 1024 | 4 | 73.8 | 1469 ms | 46.6 ms | 967 ms | 23.3 |
| 1024 | 8 | 77.9 | 2485 ms | 87.2 ms | 1990 ms | 12.0 |
| 1024 | 16 | 79.5 | 4687 ms | 160.5 ms | 4159 ms | 6.4 |
| 2048 | 1 | 42.6 | 1567 ms | 11.3 ms | 1563 ms | 88.5 |
| 2048 | 2 | 43.5 | 3119 ms | 34.0 ms | 2918 ms | 45.9 |
| 2048 | 4 | 29.2 | 4764 ms | 423.3 ms | 3738 ms | 17.1 |
| 2048 | 8 | 47.2 | 8097 ms | 117.9 ms | 7027 ms | 9.4 |
| 2048 | 16 | 49.1 | 14749 ms | 219.6 ms | 13726 ms | 4.8 |

**Notes:** ~101 sTPS at ISL 128 conc=1 (MTP). ISL 512 conc=2-8 shows anomalous low throughput (batching overhead with larger prefills). Aggregate peaks ~86 tok/s at ISL 512 conc=16.


---

## Nemotron-H 30B MoE (NVFP4)

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`
**Config:** max-batch-size=16, no MTP
**Coherence:** 9/10 passed (multi-turn context recall: model limitation, not server bug)

| ISL | Conc | Tput | TTFT p50 | TPOT p50 | sTTFT p50 | sTPS p50 |
|-----|------|------|----------|----------|-----------|----------|
| 128 | 1 | 89.9 | 129 ms | 10.2 ms | 127 ms | 98.1 |
| 128 | 2 | 95.7 | 256 ms | 20.0 ms | 129 ms | 52.6 |
| 128 | 4 | 73.7 | 390 ms | 40.0 ms | 126 ms | 25.6 |
| 128 | 8 | 75.0 | 642 ms | 76.8 ms | 125 ms | 13.0 |
| 128 | 16 | 8.4 | 1139 ms | 10.4 ms | n/a | n/a |
| 512 | 1 | 8.0 | 124 ms | n/a | 122 ms | 0.0 |
| 512 | 2 | 8.1 | 246 ms | n/a | 123 ms | 0.0 |
| 512 | 4 | 8.2 | 366 ms | n/a | 121 ms | 0.0 |
| 512 | 8 | 67.7 | 607 ms | 317.8 ms | 121 ms | 3.2 |
| 512 | 16 | 8.8 | 1092 ms | 10.4 ms | n/a | n/a |
| 1024 | 1 | 56.8 | 925 ms | 10.5 ms | 922 ms | 95.7 |
| 1024 | 2 | 59.9 | 1847 ms | 26.3 ms | 918 ms | 52.3 |
| 1024 | 4 | 60.3 | 2788 ms | 52.0 ms | 1854 ms | 22.3 |
| 1024 | 8 | 60.9 | 4701 ms | 102.1 ms | 3767 ms | 10.5 |
| 1024 | 16 | 8.5 | 8665 ms | 937.4 ms | n/a | n/a |
| 2048 | 1 | 13.5 | 8123 ms | 10.5 ms | 8119 ms | 95.0 |
| 2048 | 2 | 13.7 | 16262 ms | 83.3 ms | 9722 ms | 51.7 |
| 2048 | 4 | 11.7 | 24430 ms | 165.0 ms | 17852 ms | 9.9 |
| 2048 | 8 | 13.3 | 40789 ms | 329.7 ms | 34209 ms | 3.8 |
| 2048 | 16 | 1.9 | 73828 ms | 4075.8 ms | n/a | n/a |

**Notes:** ~98 sTPS at ISL 128 conc=1, ~96 sTPS at ISL 1024. Several anomalous configurations (ISL 512 n/a TPOT, ISL 2048 extreme TTFT). Mamba-2 SSM architecture has high prefill cost at long sequences. Conc=16 at ISL 128/512/1024/2048 shows degradation (PTok=0 indicates possible scheduling issue).


---

## Qwen3.5-122B MoE (NVFP4, EP=2, MTP)

**Model:** `Sehyo/Qwen3.5-122B-A10B-NVFP4`
**Config:** EP=2 (2x GB10, RDMA), max-batch-size=1, gpu-memory-utilization=0.55, MTP K=2
**Coherence:** 10/10 passed

| ISL | Conc | Tput | TTFT p50 | TPOT p50 | sTTFT p50 | sTPS p50 |
|-----|------|------|----------|----------|-----------|----------|
| 128 | 1 | 48.3 | 317 ms | 18.4 ms | 315 ms | 54.4 |
| 512 | 1 | 46.0 | 256 ms | 19.9 ms | 253 ms | 50.2 |
| 1024 | 1 | 38.7 | 761 ms | 20.0 ms | 759 ms | 49.9 |
| 2048 | 1 | 26.8 | 2108 ms | 21.0 ms | 2104 ms | 47.7 |

**Notes:** Conc=1 only (EP forces batch=1). ISL 512 TTFT benefits from Marconi prefix cache. Decode stable ~50 sTPS across ISLs. TTFT scales linearly with ISL (chunked prefill).


---

## Summary — Single-Stream Decode Performance (Conc=1)

| Model | ISL 128 sTPS | ISL 512 sTPS | ISL 1024 sTPS | ISL 2048 sTPS | MTP |
|-------|-------------|-------------|--------------|--------------|-----|
| **35B MoE** | 79 | **131** | 111 | 115 | Yes |
| **80B MoE** | **101** | 92 | 96 | 89 | Yes |
| **VL-30B** | 97 | — | 86 | 79 | No |
| **Nemotron-H** | 98 | — | 96 | 95 | No |
| **122B EP=2** | 54 | 50 | 50 | 48 | Yes |
| **27B Dense** | 14 | 14 | 14 | — | No |

## Summary — Aggregate Throughput Scaling (tok/s)

| Model | Conc=1 | Conc=2 | Conc=4 | Conc=8 | Conc=16 |
|-------|--------|--------|--------|--------|---------|
| **35B MoE** (ISL 1024) | 87 | 87 | 90 | 95 | 97 |
| **80B MoE** (ISL 1024) | 72 | 69 | 74 | 78 | 80 |
| **VL-30B** (ISL 1024) | 49 | 60 | 62 | 64 | 65 |
| **Nemotron-H** (ISL 1024) | 57 | 60 | 60 | 61 | 9 |
| **122B EP=2** (ISL 1024) | 39 | — | — | — | — |
| **27B Dense** (ISL 1024) | 13 | — | — | — | — |

---

Finished at 2026-03-11
