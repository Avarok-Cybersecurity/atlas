# Atlas GB10 — Full Model Benchmark Results

**Date:** 2026-03-10
**Hardware:** 2× NVIDIA GB10 Grace Blackwell (119.7 GB GPU memory each)
**Image:** `atlas-gb10:latest` (ATLAS_TARGET_MODEL=*)
**KV Cache:** NVFP4 (all models)
**Scheduler:** SLAI (SLO-aware: shortest-prompt-first prefill, decode-priority near TBT deadline)
**Benchmark:** count-prompt mode, OSL=128, warmup=1

## Metric Definitions

| Metric | Description |
|--------|-------------|
| TTFT   | Client Time To First Token (prefill latency), ms |
| TPOT   | Client Time Per Output Token (decode inter-token), ms |
| E2E    | Client end-to-end latency (start → last token), ms |
| sTTFT  | Server TTFT (server-side, excludes network RTT), ms |
| sTPS   | Server decode throughput (tok/s per sequence) |
| Tput   | Aggregate output tok/s across concurrent batch |

All latency metrics: **p50 / p90 / p99**.


---

## Qwen3.5-27B Dense (NVFP4)

**Model:** `Kbenkhaled/Qwen3.5-27B-NVFP4`  
**Config:** NVFP4 KV cache, max-seq-len=8192, no MTP (hybrid SSM), ISL≤1024  
**Coherence:** 9/9 passed

### Coherence Test Output

```
Atlas Spark — Coherence Test Suite
  Model : Kbenkhaled/Qwen3.5-27B-NVFP4
  URL   : http://localhost:8888

  PASS  factual: 2+2=4
       [got: '\n\n4']
  PASS  factual: capital of France
       [got: '\n\nParis']
  PASS  temperature diversity (5 runs)
  PASS  streaming format (SSE + usage chunk)
       [[DONE]=True content=True usage_chunk=True ttft_field=True]
  PASS  thinking mode (enable_thinking=True)
       [empty response]
  PASS  content: array-of-parts (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9bb2f94e7524', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9bb35c7c22c0', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  PASS  Korean/emoji no garble (streaming)

  All 9 tests passed.
```

### Concurrency Benchmark

```
Atlas Spark — Concurrency Benchmark
  Model  : Kbenkhaled/Qwen3.5-27B-NVFP4
  URL    : http://localhost:8888
  OSL    : 128 max output tokens per request
  Warmup : 1 request(s) per configuration
  Prompt : count (forces full OSL)

  TTFT  = client Time To First Token  (prefill latency)
  TPOT  = client Time Per Output Token (decode inter-token latency)
  E2E   = client end-to-end latency    (start → last token)
  sTTFT = server TTFT (server-side, excludes network RTT)
  sTPS  = server decode tok/s
  Tput  = aggregate output tok/s across concurrent batch

   ISL    OSL  Conc   PTok      Tput   TTFT p50/p90/p99 ms     TPOT p50/p90/p99 ms    E2E p50 ms  sTTFT p50 ms  sTPS p50
------------------------------------------------------------------------------------------------------------------------
   128    128     1    158     12.4t  1280.2 / 1280.2 / 1280.2  71.10 / 71.10 / 71.10        10310        1277.6      14.1
   128    128     4    158     13.1t  3911.3 / 5186.4 / 5186.4  286.92 / 295.42 / 295.42       39073        1278.1       3.6
   512    128     1    584      9.9t  3822.1 / 3822.1 / 3822.1  71.66 / 71.66 / 71.66        12924        3819.6      14.0
   512    128     4    584      8.1t  11530.1 / 15344.7 / 15344.7  325.83 / 355.39 / 355.39       49091        3820.0       3.4
  1024    128     1   1151      8.0t  6912.2 / 6912.2 / 6912.2  72.32 / 72.32 / 72.32        16097        6908.6      13.8
  1024    128     4   1151      6.0t  26979.6 / 27923.8 / 27923.8  129.54 / 9126.35 / 9126.35       43432       19053.8       8.2
```


---

## Qwen3-VL-30B MoE Vision (NVFP4)

**Model:** `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4`  
**Config:** NVFP4 KV cache, max-seq-len=8192, no MTP (no MTP weights in this checkpoint)  
**Coherence:** 9/9 passed

### Coherence Test Output

```
Atlas Spark — Coherence Test Suite
  Model : ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4
  URL   : http://localhost:8888

  PASS  factual: 2+2=4
       [got: '4']
  PASS  factual: capital of France
       [got: 'Paris']
  PASS  temperature diversity (5 runs)
  PASS  streaming format (SSE + usage chunk)
       [[DONE]=True content=True usage_chunk=True ttft_field=True]
  PASS  thinking mode (enable_thinking=True)
       [empty response]
  PASS  content: array-of-parts (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9c094276d34a', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9c095ca49fe3', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  PASS  Korean/emoji no garble (streaming)

  All 9 tests passed.
```

### Concurrency Benchmark

```
Atlas Spark — Concurrency Benchmark
  Model  : ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4
  URL    : http://localhost:8888
  OSL    : 128 max output tokens per request
  Warmup : 1 request(s) per configuration
  Prompt : count (forces full OSL)

  TTFT  = client Time To First Token  (prefill latency)
  TPOT  = client Time Per Output Token (decode inter-token latency)
  E2E   = client end-to-end latency    (start → last token)
  sTTFT = server TTFT (server-side, excludes network RTT)
  sTPS  = server decode tok/s
  Tput  = aggregate output tok/s across concurrent batch

   ISL    OSL  Conc   PTok      Tput   TTFT p50/p90/p99 ms     TPOT p50/p90/p99 ms    E2E p50 ms  sTTFT p50 ms  sTPS p50
------------------------------------------------------------------------------------------------------------------------
   128    128     1    156     74.6t  403.8 / 403.8 / 403.8   10.33 / 10.33 / 10.33         1716         401.7      96.8
   128    128     4    156     67.4t  1200.0 / 1652.1 / 1652.1  40.96 / 44.10 / 44.10         6004         401.6      26.4
   128    128    16    156     88.8t  11856.9 / 14331.9 / 14731.9  78.24 / 87.48 / 87.74        23000         400.6      12.8
   512    128     1    582      1.5t  667.2 / 667.2 / 667.2            n/a                   667         665.0       0.0
   512    128     4    582      1.5t  1993.1 / 2651.9 / 2651.9           n/a                  1993         664.2       0.0
   512    128    16    582     21.1t  5971.6 / 9965.6 / 10644.9  362.41 / 362.41 / 362.41        7314         663.7       0.0
  1024    128     1   1149      2.5t  1168.1 / 1168.1 / 1168.1  12.81 / 12.81 / 12.81         1194        1165.4      77.8
  1024    128     4   1149      2.4t  4271.7 / 4598.6 / 4598.6  180.68 / 226.02 / 226.02        4633        2626.6      28.5
  1024    128    16   1149      2.7t  15304.6 / 18178.0 / 18519.1  179.08 / 583.41 / 590.36       17185        3464.0       5.9
  4096    128     1   4307      0.8t  3580.7 / 3580.7 / 3580.7  18.70 / 18.70 / 18.70         3618        3575.0      53.8
  4096    128     4   4307     25.5t  11900.0 / 14878.0 / 14878.0  86.32 / 107.66 / 107.66       20028       10247.4      15.6
  4096    128    16   4307     16.0t  35800.7 / 65018.4 / 68985.5  309.26 / 644.52 / 705.04       69072        6719.3       3.3
  8192    128     1  1 error(s): HTTP Error 400: Bad Request
  8192    128     4  4 error(s): HTTP Error 400: Bad Request
  8192    128    16  16 error(s): HTTP Error 400: Bad Request
```


---

## Qwen3.5-35B MoE (NVFP4, MTP K=2)

**Model:** `Kbenkhaled/Qwen3.5-35B-A3B-NVFP4`  
**Config:** NVFP4 KV cache, max-seq-len=8192, speculative MTP K=2  
**Coherence:** 9/9 passed

### Coherence Test Output

```
Atlas Spark — Coherence Test Suite
  Model : Kbenkhaled/Qwen3.5-35B-A3B-NVFP4
  URL   : http://localhost:8888

  PASS  factual: 2+2=4
       [got: '\n\n4']
  PASS  factual: capital of France
       [got: '\n\nParis']
  PASS  temperature diversity (5 runs)
  PASS  streaming format (SSE + usage chunk)
       [[DONE]=True content=True usage_chunk=True ttft_field=True]
  PASS  thinking mode (enable_thinking=True)
       [empty response]
  PASS  content: array-of-parts (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9c4ff1a3b7b1', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9c5002c561ad', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  PASS  Korean/emoji no garble (streaming)

  All 9 tests passed.
```

### Concurrency Benchmark

```
Atlas Spark — Concurrency Benchmark
  Model  : Kbenkhaled/Qwen3.5-35B-A3B-NVFP4
  URL    : http://localhost:8888
  OSL    : 128 max output tokens per request
  Warmup : 1 request(s) per configuration
  Prompt : count (forces full OSL)

  TTFT  = client Time To First Token  (prefill latency)
  TPOT  = client Time Per Output Token (decode inter-token latency)
  E2E   = client end-to-end latency    (start → last token)
  sTTFT = server TTFT (server-side, excludes network RTT)
  sTPS  = server decode tok/s
  Tput  = aggregate output tok/s across concurrent batch

   ISL    OSL  Conc   PTok      Tput   TTFT p50/p90/p99 ms     TPOT p50/p90/p99 ms    E2E p50 ms  sTTFT p50 ms  sTPS p50
------------------------------------------------------------------------------------------------------------------------
   128    128     1    158    100.4t  315.4 / 315.4 / 315.4     7.55 / 7.55 / 7.55          1274         313.6     132.4
   128    128     4    158     94.7t  961.3 / 1277.6 / 1277.6  37.49 / 39.78 / 39.78         5407         316.7      28.6
   128    128    16    158     94.4t  11046.2 / 13021.3 / 13337.8  75.71 / 82.36 / 83.28        21624         315.3      13.3
   512    128     1    584     90.4t  440.8 / 440.8 / 440.8     7.68 / 7.68 / 7.68          1416         438.1     130.2
   512    128     4    584     65.6t  1315.4 / 1752.0 / 1752.0  42.45 / 45.77 / 45.77         6275         435.3      25.6
   512    128    16    584     84.8t  12406.9 / 15095.7 / 15526.1  81.48 / 91.25 / 91.82        24070         434.0      12.3
  1024    128     1   1151     71.6t  796.8 / 796.8 / 796.8     7.80 / 7.80 / 7.80          1788         794.4     128.2
  1024    128     4   1151     49.8t  2951.1 / 3275.9 / 3275.9  40.15 / 54.16 / 54.16         7734        1821.1      26.5
  1024    128    16   1151     67.1t  15830.2 / 21114.2 / 21978.3  99.97 / 112.56 / 114.05       30126         804.0      10.2
  4096    128     1   4309     34.7t  2372.9 / 2372.9 / 2372.9  10.37 / 10.37 / 10.37         3690        2364.7      96.4
  4096    128     4   4309     21.6t  7723.4 / 9681.8 / 9681.8  21.86 / 62.94 / 62.94        11899        6711.8      46.9
  4096    128    16   4309     17.0t  24609.1 / 37386.4 / 39506.1  40.50 / 288.16 / 301.26       29571        4917.7      24.9
  8192    128     1  1 error(s): HTTP Error 400: Bad Request
  8192    128     4  4 error(s): HTTP Error 400: Bad Request
  8192    128    16  16 error(s): HTTP Error 400: Bad Request
```


---

## Qwen3-Next-80B MoE (NVFP4, MTP K=2)

**Model:** `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`  
**Config:** NVFP4 KV cache, max-seq-len=8192, speculative MTP K=2  
**Coherence:** 8/10 passed

### Coherence Test Output

```
Atlas Spark — Coherence Test Suite
  Model : nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4
  URL   : http://localhost:8888

  PASS  factual: 2+2=4
       [got: '4']
  PASS  factual: capital of France
       [got: 'Paris']
  FAIL  temperature diversity (5 runs)
       all outputs identical: '482'
       [all outputs identical: '482']
  PASS  streaming format (SSE + usage chunk)
       [[DONE]=True content=True usage_chunk=True ttft_field=True]
  PASS  thinking mode (enable_thinking=True)
       [empty response]
  PASS  content: array-of-parts (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9cab76260cea', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9cab8e46d1fc', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  PASS  Korean/emoji no garble (streaming)

  1/9 tests FAILED.
```

### Concurrency Benchmark

```
Atlas Spark — Concurrency Benchmark
  Model  : nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4
  URL    : http://localhost:8888
  OSL    : 128 max output tokens per request
  Warmup : 1 request(s) per configuration
  Prompt : count (forces full OSL)

  TTFT  = client Time To First Token  (prefill latency)
  TPOT  = client Time Per Output Token (decode inter-token latency)
  E2E   = client end-to-end latency    (start → last token)
  sTTFT = server TTFT (server-side, excludes network RTT)
  sTPS  = server decode tok/s
  Tput  = aggregate output tok/s across concurrent batch

   ISL    OSL  Conc   PTok      Tput   TTFT p50/p90/p99 ms     TPOT p50/p90/p99 ms    E2E p50 ms  sTTFT p50 ms  sTPS p50
------------------------------------------------------------------------------------------------------------------------
   128    128     1    156     73.4t  525.4 / 525.4 / 525.4     9.58 / 9.58 / 9.58          1742         522.8     104.3
   128    128     4    156     77.2t  1592.3 / 2111.4 / 2111.4  43.79 / 47.87 / 47.87         6634         520.7      25.2
   128    128    16    156     66.4t  4901.9 / 17279.4 / 17805.4  93.69 / 102.75 / 1281.41       17874         524.1      11.1
   512    128     1    582     56.1t  863.4 / 863.4 / 863.4   11.15 / 11.15 / 11.15         2280         861.0      89.7
   512    128     4    582     58.5t  2592.1 / 3445.2 / 3445.2  55.35 / 61.76 / 61.76         8755         861.5      20.6
   512    128    16    582     62.2t  17306.7 / 22471.9 / 23323.9  109.41 / 122.71 / 123.23       32837         850.5       9.2
  1024    128     1   1149     44.0t  1528.7 / 1528.7 / 1528.7  10.85 / 10.85 / 10.85         2906        1526.3      92.2
  1024    128     4   1149     44.0t  5747.7 / 6258.6 / 6258.6  49.89 / 78.42 / 78.42        11619        3693.2      21.6
  1024    128    16   1149     43.0t  14419.5 / 34716.9 / 35256.8  116.66 / 139.96 / 1270.28       32195        6214.1       8.7
  4096    128     1   4307     20.4t  4702.6 / 4702.6 / 4702.6  12.31 / 12.31 / 12.31         6266        4696.6      81.2
  4096    128     4   4307     20.8t  15598.0 / 19416.2 / 19416.2  99.41 / 128.11 / 128.11       24591       13239.3      14.1
  4096    128    16   4307     20.1t  44073.9 / 83520.2 / 87428.1  312.10 / 325.76 / 904.22       84688        8862.5       3.2
  8192    128     1  1 error(s): HTTP Error 400: Bad Request
  8192    128     4  4 error(s): HTTP Error 400: Bad Request
  8192    128    16  16 error(s): HTTP Error 400: Bad Request
```


---

## Nemotron-H 30B MoE (NVFP4)

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`  
**Config:** NVFP4 KV cache, max-seq-len=8192, no MTP  
**Coherence:** 9/9 passed

### Coherence Test Output

```
Atlas Spark — Coherence Test Suite
  Model : nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
  URL   : http://localhost:8888

  PASS  factual: 2+2=4
       [got: '4']
  PASS  factual: capital of France
       [got: 'The capital of France is Paris.']
  PASS  temperature diversity (5 runs)
  PASS  streaming format (SSE + usage chunk)
       [[DONE]=True content=True usage_chunk=True ttft_field=True]
  PASS  thinking mode (enable_thinking=True)
       [empty response]
  PASS  content: array-of-parts (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9d09adf87a34', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9d09d252c018', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  PASS  Korean/emoji no garble (streaming)

  All 9 tests passed.
```

### Concurrency Benchmark

```
Atlas Spark — Concurrency Benchmark
  Model  : nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
  URL    : http://localhost:8888
  OSL    : 128 max output tokens per request
  Warmup : 1 request(s) per configuration
  Prompt : count (forces full OSL)

  TTFT  = client Time To First Token  (prefill latency)
  TPOT  = client Time Per Output Token (decode inter-token latency)
  E2E   = client end-to-end latency    (start → last token)
  sTTFT = server TTFT (server-side, excludes network RTT)
  sTPS  = server decode tok/s
  Tput  = aggregate output tok/s across concurrent batch

   ISL    OSL  Conc   PTok      Tput   TTFT p50/p90/p99 ms     TPOT p50/p90/p99 ms    E2E p50 ms  sTTFT p50 ms  sTPS p50
------------------------------------------------------------------------------------------------------------------------
   128    128     1    158     55.0t  1030.3 / 1030.3 / 1030.3  10.20 / 10.20 / 10.20         2326        1028.4      98.0
   128    128     4    158     57.9t  3120.1 / 4151.8 / 4151.8  53.20 / 61.33 / 61.33         8841        1032.1      22.2
   128    128    16    158     58.1t  18585.9 / 24867.3 / 25903.4  106.36 / 130.59 / 130.82       35200        1033.9       9.4
   512    128     1    589      6.7t  3779.8 / 3779.8 / 3779.8  10.38 / 10.38 / 10.38         4050        3777.5      96.4
   512    128     4    589     20.7t  11341.8 / 15113.9 / 15113.9  126.61 / 328.10 / 328.10       19879        3775.6      14.9
   512    128    16    589     18.9t  34440.9 / 66126.8 / 69903.6  253.10 / 371.44 / 371.97       54807        3775.8       4.5
  1024    128     1   1162     14.6t  7442.2 / 7442.2 / 7442.2  10.49 / 10.49 / 10.49         8775        7440.0      95.3
  1024    128     4   1162     14.8t  28845.8 / 29793.8 / 29793.8  52.15 / 212.90 / 212.90       34561       15869.4      22.2
  1024    128    16   1162     13.0t  67459.9 / 128002.5 / 128992.4  368.08 / 682.95 / 682.95      124259       23369.2       3.1
  4096    128     1   4354      4.4t  27780.9 / 27780.9 / 27780.9  11.09 / 11.09 / 11.09        29190       27775.8      90.1
  4096    128     4   4354      0.3t  89926.1 / 111262.3 / 111262.3  3050.10 / 4650.80 / 4650.80      111283       70402.5      25.6
  4096    128    16   4354      4.0t  252328.8 / 434045.5 / 455566.4  1500.76 / 2312.61 / 3081.64      456249       56188.6       0.7
  8192    128     1  1 error(s): HTTP Error 400: Bad Request
  8192    128     4  4 error(s): HTTP Error 400: Bad Request
  8192    128    16  16 error(s): HTTP Error 400: Bad Request
```


---

## Qwen3.5-122B MoE (NVFP4, EP=2, MTP K=2)

**Model:** `Sehyo/Qwen3.5-122B-A10B-NVFP4`  
**Config:** EP=2, NVFP4 KV, MTP K=2, max-seq-len=4096  
**Coherence:** 9/9 passed

### Coherence Test Output

```
Atlas Spark — Coherence Test Suite
  Model : Sehyo/Qwen3.5-122B-A10B-NVFP4
  URL   : http://localhost:8888

  PASS  factual: 2+2=4
       [got: '\n\n4']
  PASS  factual: capital of France
       [got: '\n\nParis']
  PASS  temperature diversity (5 runs)
  PASS  streaming format (SSE + usage chunk)
       [[DONE]=True content=True usage_chunk=True ttft_field=True]
  PASS  thinking mode (enable_thinking=True)
       [empty response]
  PASS  content: array-of-parts (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9e442a6e1cb6', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189b9e44695850c2', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  PASS  Korean/emoji no garble (streaming)

  All 9 tests passed.
```

### Concurrency Benchmark

```
Atlas Spark — Concurrency Benchmark
  Model  : Sehyo/Qwen3.5-122B-A10B-NVFP4
  URL    : http://localhost:8888
  OSL    : 128 max output tokens per request
  Warmup : 1 request(s) per configuration
  Prompt : count (forces full OSL)

  TTFT  = client Time To First Token  (prefill latency)
  TPOT  = client Time Per Output Token (decode inter-token latency)
  E2E   = client end-to-end latency    (start → last token)
  sTTFT = server TTFT (server-side, excludes network RTT)
  sTPS  = server decode tok/s
  Tput  = aggregate output tok/s across concurrent batch

   ISL    OSL  Conc   PTok      Tput   TTFT p50/p90/p99 ms     TPOT p50/p90/p99 ms    E2E p50 ms  sTTFT p50 ms  sTPS p50
------------------------------------------------------------------------------------------------------------------------
   128    128     1    158     40.4t  668.4 / 668.4 / 668.4   19.68 / 19.68 / 19.68         3167         666.6      50.8
   128    128     2    158     40.2t  3862.8 / 3862.8 / 3862.8  19.76 / 19.76 / 19.76         6366         680.6      50.7
   128    128     4    158     40.2t  7042.7 / 10223.1 / 10223.1  19.70 / 19.72 / 19.72         9542         679.5      50.8
   512    128     1    584     36.4t  993.1 / 993.1 / 993.1   19.90 / 19.90 / 19.90         3521         991.0      50.2
   512    128     2    584     36.3t  4516.2 / 4516.2 / 4516.2  19.91 / 19.91 / 19.91         7043         993.6      50.3
   512    128     4    584     36.3t  8050.1 / 11568.2 / 11568.2  19.91 / 19.95 / 19.95        10578         995.4      50.2
  1024    128     1   1151     23.8t  1964.4 / 1964.4 / 1964.4  26.83 / 26.83 / 26.83         5371        1961.0      37.3
  1024    128     2   1151     19.2t  4038.5 / 4038.5 / 4038.5  37.42 / 37.42 / 37.42         6891        1965.3      44.5
  1024    128     4   1151     12.6t  6088.4 / 10881.4 / 10881.4  35.59 / 36.57 / 36.57         8916        1968.1      34.0
  2048    128     1   2263     21.9t  3348.7 / 3348.7 / 3348.7  19.72 / 19.72 / 19.72         5854        3345.2      50.7
  2048    128     2   2263     20.1t  9365.4 / 9365.4 / 9365.4  26.71 / 26.71 / 26.71        12758        3360.0      47.9
  2048    128     4   2263      8.3t  13876.8 / 17423.6 / 17423.6  29.90 / 38.64 / 38.64        14070        3361.7      34.9
```


---

## Sweep Complete

Finished at 2026-03-10 19:24:36
