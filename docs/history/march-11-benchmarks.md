# Atlas GB10 — Full Model Benchmark Results

**Date:** 2026-03-11
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
**Coherence:** 10/10 passed

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
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bcde2c37f7e83', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bcde33c4fe350', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  PASS  multi-turn context recall
       [got: '\n\nZephyr']
  PASS  Korean/emoji no garble (streaming)

  All 10 tests passed.
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
   128    128     1    158     13.4t  509.6 / 509.6 / 509.6   71.45 / 71.45 / 71.45         9584         507.7      14.0
   128    128     4    158     14.1t  1656.9 / 2161.9 / 2161.9  276.52 / 280.51 / 280.51       36260         507.1       3.7
   512    128     1    584     13.2t  512.1 / 512.1 / 512.1   72.05 / 72.05 / 72.05         9662         510.3      13.9
   512    128     4    584     10.7t  1596.1 / 2102.0 / 2102.0  275.92 / 279.50 / 279.50       36131         507.3       3.7
  1024    128     1   1151     12.7t  873.9 / 873.9 / 873.9   72.38 / 72.38 / 72.38        10066         871.3      13.8
  1024    128     4   1151     10.8t  2832.6 / 3972.0 / 3972.0  281.39 / 287.94 / 287.94       37575        1871.9       3.6
```


---

## Qwen3-VL-30B MoE Vision (NVFP4)

**Model:** `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4`  
**Config:** NVFP4 KV cache, max-seq-len=8192, no MTP (no MTP weights in this checkpoint)  
**Coherence:** 10/10 passed

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
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bce2f5d24c006', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bce2f77861d80', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  PASS  multi-turn context recall
       [got: 'Zephyr']
  PASS  Korean/emoji no garble (streaming)

  All 10 tests passed.
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
   128    128     1    156      4.8t  402.3 / 402.3 / 402.3   11.28 / 11.28 / 11.28          414         400.8      88.1
   128    128     4    156     67.2t  1233.6 / 1638.3 / 1638.3  35.48 / 41.90 / 41.90         5740         404.6      31.0
   128    128    16    156    101.9t  10319.8 / 12785.8 / 13191.1  68.83 / 75.14 / 76.57        20043         402.6      14.9
   512    128     1    582      1.5t  666.0 / 666.0 / 666.0            n/a                   666         664.4       0.0
   512    128     4    582      1.5t  1982.6 / 2645.6 / 2645.6           n/a                  1983         664.1       0.0
   512    128    16    582     36.7t  5945.3 / 9899.5 / 10586.1  65.60 / 102.06 / 102.06        7274         658.4       0.0
  1024    128     1   1149     48.4t  1153.9 / 1153.9 / 1153.9  11.71 / 11.71 / 11.71         2642        1150.9      85.4
  1024    128     4   1149     61.3t  4264.6 / 4611.6 / 4611.6  34.60 / 56.16 / 56.16         8338        2789.6      31.2
  1024    128    16   1149     61.1t  17444.9 / 24789.8 / 25990.7  101.50 / 121.55 / 123.91       33182        1165.0      10.3
  4096    128     1   4307     23.2t  3566.4 / 3566.4 / 3566.4  15.26 / 15.26 / 15.26         5505        3559.4      65.5
  4096    128     4   4307     26.5t  11883.6 / 14848.7 / 14848.7  80.39 / 102.02 / 102.02       19258       10329.9      17.2
  4096    128    16   4307     26.1t  41750.5 / 65421.9 / 69337.8  231.75 / 271.18 / 275.02       76678        7780.7       4.5
  8192    128     1  1 error(s): HTTP Error 400: Bad Request
  8192    128     4  4 error(s): HTTP Error 400: Bad Request
  8192    128    16  16 error(s): HTTP Error 400: Bad Request
```


---

## Qwen3.5-35B MoE (NVFP4, MTP K=2)

**Model:** `Kbenkhaled/Qwen3.5-35B-A3B-NVFP4`  
**Config:** NVFP4 KV cache, max-seq-len=8192, speculative MTP K=2  
**Coherence:** 10/10 passed

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
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bce7e9bf5f06f', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bce7ead24ef3e', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  PASS  multi-turn context recall
       [got: '\n\nZephyr']
  PASS  Korean/emoji no garble (streaming)

  All 10 tests passed.
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
   128    128     1    158    115.3t  127.2 / 127.2 / 127.2     7.73 / 7.73 / 7.73          1110         124.9     129.3
   128    128     4    158     94.1t  407.6 / 535.4 / 535.4   40.20 / 43.45 / 43.45         3560         128.2      37.7
   128    128    16    158    104.1t  9208.1 / 10071.1 / 10203.0  69.86 / 325.46 / 325.46        9551         130.8      14.3
   512    128     1    584    120.6t    93.1 / 93.1 / 93.1      7.62 / 7.62 / 7.62          1061          91.2     131.3
   512    128     4    584    108.2t  295.1 / 389.5 / 389.5   35.65 / 36.20 / 36.20         4730          94.1      28.6
   512    128    16    584    100.7t  939.7 / 10132.6 / 10229.4  73.23 / 74.28 / 74.28         9645          96.2      13.5
  1024    128     1   1151     97.5t  303.7 / 303.7 / 303.7     7.94 / 7.94 / 7.94          1313         300.5     125.9
  1024    128     4   1151     90.8t  931.1 / 1261.6 / 1261.6  39.31 / 41.51 / 41.51         5629         612.0      27.0
  1024    128    16   1151     92.6t  11069.9 / 13288.1 / 13655.4  82.68 / 83.92 / 84.56        21809         302.6      12.1
  4096    128     1   4309     32.4t  1805.3 / 1805.3 / 1805.3    8.95 / 8.95 / 8.95          2530        1797.3     111.7
  4096    128     4   4309     42.5t  5611.0 / 7556.7 / 7556.7  64.67 / 77.89 / 77.89        12006        5112.8      19.9
  4096    128    16   4309     40.8t  18680.2 / 37065.3 / 39158.8  170.34 / 184.53 / 184.71       40313        2043.2       6.4
  8192    128     1  1 error(s): HTTP Error 400: Bad Request
  8192    128     4  4 error(s): HTTP Error 400: Bad Request
  8192    128    16  16 error(s): HTTP Error 400: Bad Request
```


---

## Qwen3-Next-80B MoE (NVFP4, MTP K=2)

**Model:** `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4`  
**Config:** NVFP4 KV cache, max-seq-len=8192, speculative MTP K=2  
**Coherence:** 10/10 passed

### Coherence Test Output

```
Atlas Spark — Coherence Test Suite
  Model : nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4
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
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bcecdcadee23f', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bcecde30f905a', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=347 (max_tokens=512, old cap was 256)]
  PASS  multi-turn context recall
       [got: 'Zephyr']
  PASS  Korean/emoji no garble (streaming)

  All 10 tests passed.
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
   128    128     1    156     69.5t  191.9 / 191.9 / 191.9   12.98 / 12.98 / 12.98         1840         190.2      77.0
   128    128     4    156     76.2t  550.8 / 728.1 / 728.1   43.72 / 46.21 / 46.21         3521         178.2      27.2
   128    128    16    156     85.8t  3421.2 / 8785.8 / 10335.4  84.68 / 102.31 / 103.34       10818         182.5      12.9
   512    128     1    582     76.2t  139.1 / 139.1 / 139.1   12.13 / 12.13 / 12.13         1680         136.2      82.4
   512    128     4    582     80.8t  432.4 / 570.4 / 570.4   44.66 / 45.74 / 45.74         5948         137.4      22.6
   512    128    16    582     86.0t  5389.8 / 11977.6 / 12109.2  86.03 / 339.89 / 474.47       12183         135.5      11.7
  1024    128     1   1149     67.9t  455.6 / 455.6 / 455.6   11.26 / 11.26 / 11.26         1886         452.3      88.8
  1024    128     4   1149     66.6t  1414.2 / 1904.8 / 1904.8  48.62 / 55.91 / 55.91         7143         930.5      22.2
  1024    128    16   1149     76.6t  13774.6 / 17067.8 / 17607.0  101.00 / 102.61 / 102.61       26394         478.5      10.1
  4096    128     1   4307     24.1t  3736.1 / 3736.1 / 3736.1  12.39 / 12.39 / 12.39         5310        3730.8      80.7
  4096    128     4   4307     25.2t  11137.4 / 14867.3 / 14867.3  99.83 / 126.93 / 126.93       20243       10122.0      13.9
  4096    128    16   4307     21.6t  34816.0 / 66323.0 / 70165.5  279.86 / 849.36 / 897.53       59642        7837.2       3.6
  8192    128     1  1 error(s): HTTP Error 400: Bad Request
  8192    128     4  4 error(s): HTTP Error 400: Bad Request
  8192    128    16  16 error(s): HTTP Error 400: Bad Request
```


---

## Nemotron-H 30B MoE (NVFP4)

**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`  
**Config:** NVFP4 KV cache, max-seq-len=8192, no MTP  
**Coherence:** 9/11 passed

### Coherence Test Output

```
Atlas Spark — Coherence Test Suite
  Model : nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
  URL   : http://localhost:8888

  PASS  factual: 2+2=4
       [got: '4']
  PASS  factual: capital of France
       [got: 'The user asks: "What is the capital of France?" It\'s a straightforward factual question. The answer: Paris. Provide one word: "Paris". Should respond with just "Paris" or "The capital is Paris". Probably just "Paris". Ensure it\'s appropriate.\n</think>\nParis.']
  PASS  temperature diversity (5 runs)
  PASS  streaming format (SSE + usage chunk)
       [[DONE]=True content=True usage_chunk=True ttft_field=True]
  PASS  thinking mode (enable_thinking=True)
       [empty response]
  PASS  content: array-of-parts (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bcf14f81a4cc8', 'object': 'chat.completion',]
  PASS  null content / tool role (no 422)
       [HTTP 200: {'id': 'chatcmpl-0000000000000000189bcf151ceeb068', 'object': 'chat.completion',]
  PASS  default max_tokens > 256
       [completion_tokens=512 (max_tokens=512, old cap was 256)]
  FAIL  multi-turn context recall
       got: 'I’m sorry, but I can’t determine your name from the information
       provided. If you’d like to share it, feel free to let me know!'
       [got: 'I’m sorry, but I can’t determine your name from the information provided. If you’d like to share it, feel free to let me know!']
  PASS  Korean/emoji no garble (streaming)

  1/10 tests FAILED.
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
   128    128     1    158     89.3t  128.8 / 128.8 / 128.8   10.27 / 10.27 / 10.27         1434         127.1      97.3
   128    128     4    158     96.8t  388.4 / 511.7 / 511.7   39.55 / 40.32 / 40.32         5286         125.3      25.9
   128    128    16    158     92.1t  10476.9 / 11307.9 / 11431.9  78.27 / 81.01 / 81.28        11584         125.0      12.8
   512    128     1    589     89.1t  123.1 / 123.1 / 123.1   10.34 / 10.34 / 10.34         1436         121.6      96.7
   512    128     4    589     88.0t  376.0 / 494.2 / 494.2   22.00 / 22.00 / 22.00         2918         121.0      45.5
   512    128    16    589     86.2t  10446.4 / 11249.3 / 11365.7  78.16 / 80.83 / 80.83        10524         121.1      12.8
  1024    128     1   1162      1.1t  918.8 / 918.8 / 918.8            n/a                   919         916.1       0.0
  1024    128     4   1162     44.7t  2785.8 / 3740.3 / 3740.3  56.44 / 56.44 / 56.44         8087        1852.9      20.3
  1024    128    16   1162     60.7t  17529.5 / 23479.7 / 24474.2  124.04 / 125.10 / 125.11       33419         917.3       8.1
  4096    128     1   4354      5.6t  21356.5 / 21356.5 / 21356.5  11.07 / 11.07 / 11.07        22762       21351.5      90.3
  4096    128     4   4354      4.4t  64177.6 / 85655.1 / 85655.1  374.03 / 4945.85 / 4945.85       90322       57624.4       4.9
  4096    128    16   4354      5.7t  201355.4 / 331293.1 / 352959.9  1249.37 / 1253.38 / 1253.39      360493       21599.4       0.8
  8192    128     1  1 error(s): HTTP Error 400: Bad Request
  8192    128     4  4 error(s): HTTP Error 400: Bad Request
  8192    128    16  16 error(s): HTTP Error 400: Bad Request
```


---

## Qwen3.5-122B MoE (NVFP4, EP=2, MTP K=2)

**Model:** `Sehyo/Qwen3.5-122B-A10B-NVFP4`  
**Config:** EP=2, NVFP4 KV, MTP K=2, max-seq-len=4096  
**Coherence:** 0/0 passed

### Coherence Test Output

```
SKIPPED — server did not start
```

### Concurrency Benchmark

```
SKIPPED — server did not start\n\nDocker log tail:\n
```


---

## Sweep Complete

Finished at 2026-03-11 10:37:51
