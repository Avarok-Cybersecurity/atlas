<!-- provenance-id: 526f6e616c6420522e205374657369616b -->
# DeepSeek-V4.1 Flash on one DGX Spark: the decode retune

Starting point, measured on Blackbird (GB10) at the decode-fix binary, warm page cache,
chat endpoint, 100-token completions (receipts `~/dflash-logs/ds41_diag_serve_2026-09-19_runA.log`,
`~/dflash-logs/ds41_nsys_decode_*_2026-09-19_runB.csv`).

## The warm decode step

One token, every routed expert already resident (fetch 0):

| slice | ms | share |
|---|---|---|
| expert compute (240 GEMVs + shared) | 23.5 | 36% |
| other, not instrumented (HC mixes, norms, head, sampling, glue) | 17.4 | 27% |
| attention, 40 layers | 13.4 | 21% |
| routing, 40 layers incl. the host round-trip | 10.0 | 15% |
| engram | 0.9 | 1% |
| total | 65.3 | 15.3 tok/s |

Steps with cache misses (about a third of steps at 100 tokens): 24 of 240 experts missing on
average, 47 ms of fetch at about 6 GB/s, 117 ms per step.

## What the GPU does per token (nsys, one warm 60-token request)

- 3,184 kernel launches
- 234 `cuStreamSynchronize`
- 49 `cuMemcpyDtoHAsync`

| kernel | ms / token | launches / token | note |
|---|---|---|---|
| `kquant_mmvq_q2_k_w` (attention projections) | 13.0 | 596 | ~10 us each, one per tensor per layer |
| `kquant_mmvq_q2_k_experts_w` (gate, up) | 10.1 | 79 | ~175 GB/s on a 273 GB/s part |
| `kquant_mmvq_q3_k_experts_w` (down) | 8.9 | 39 | ~134 GB/s |
| `dense_gemv_bf16` | 7.9 | 22 | one 5.5 ms instance: the LM head resident as bf16, 1.3 GB read; the GGUF holds it as Q6_K, 0.5 GB |
| `moe_v41_router_gemv_f32out` | 5.4 | 40 | 134 us for a ~4 MB weight |
| HC chain (`hc_v41_mixes_finish` 4.2, `hc_post` 1.7, `hc_v41_mixes_dot` 1.0, `hc_v41_collapse` 0.5) | 7.4 | 320 | elementwise work at ~50 us per launch |
| attention proper (`attn_v41_sparse_attn` 2.1, `attn_v41_slice_cols` 1.0, `attn_v41_index_score` 0.9, `attn_v41_gemm_f32` 0.9) | 5.0 | 374 | `slice_cols` is 320 launches of 3 us |
| `kquant_q8_1_rows_bf16` | 0.85 | 714 | 1 us launches |

Kernel time sums to about 63 ms per token: the GPU is nearly saturated, with small
kernels rather than bandwidth. The expert reads are about 19 ms against an 11 ms floor
at LPDDR rates; the other 44 ms is launches, synchronizations, an unquantized head,
a mis-shaped router GEMV and elementwise chains.

## The levers, in order

1. The LM head stays Q6_K (5.5 ms per token today).
2. Grouped attention-projection GEMV (13.0 ms, 596 launches today).
3. Router GEMV retune, then top-k and the expert plan on the GPU (10.0 ms routing, 234 syncs, 49 D2H today), then the whole-step graph.
4. Fused hyper-connection chain (7.4 ms, 320 launches today).
5. Expert GEMV occupancy (19.0 ms today against an 11 ms floor).

Also folded in where a lever touches the file: batch `kquant_q8_1_rows_bf16` per layer
(714 launches) and `attn_v41_slice_cols` per layer (320 launches).

## Oracle

Pure refactors: the six suite outputs (MinHeap x3, Volvo x3, 300 tokens, temperature 0,
chat endpoint, one serve) byte-identical to the baseline. The head: top-1 agreement over
the same six outputs, differing positions counted and reported.

Numbers only as measured, medians of three, receipts on the PR. Results are appended below
as they land.

## Results

All five levers kept, every one byte-identical to the baseline oracle on all six suite
outputs (the head lever's near-tie allowance was not needed). Chat endpoint, 300 tokens,
temperature 0, one serve, medians of three, warm pass; receipts
`~/dflash-logs/ds41_suite_retune_<tag>_{minheap,volvo}_r{1,2,3}.json`.

| step | commit | MinHeap tok/s | MinHeap TTFT | Volvo tok/s | Volvo TTFT | launches / token |
|---|---|---|---|---|---|---|
| baseline | f13752b23 | 10.42 | 2048 ms | 10.77 | 1291 ms | 3,184 |
| L1 head stays Q6_K | a44e9fd6b | 10.71 | 2030 ms | 11.08 | 1273 ms | 3,185 |
| L2 wo_a groups in one launch | 763756050 | 10.99 | 2023 ms | 11.38 | 1248 ms | 1,985 |
| L3 router staged, one read-back | b38631920 | 11.37 | 2018 ms | 11.81 | 1283 ms | 1,985 |
| L4 HC chain in registers, wide | 2d6ea1c46 | 12.05 | 2033 ms | 12.53 | 1266 ms | 1,905 |
| L5 eight warps, gate+up merged | e1c30d867 | 12.39 | 2009 ms | 12.90 | 1262 ms | 1,866 |

Baseline to final: MinHeap +18.9%, Volvo +19.8%. The published 09-17 numbers were
10.6 / 10.9.

### The final profile (nsys, one warm 60-token request, same method as above)

Receipts `~/dflash-logs/ds41_nsys_decode_{cuda_gpu_kern_sum,cuda_api_sum,osrt_sum}_final.csv`
against the `*_2026-09-19_runB.csv` baseline set.

| per token | baseline | final |
|---|---|---|
| kernel launches | 3,184 | 1,865 |
| `cuStreamSynchronize` | 234 (49.8 ms blocked) | 186 (27.0 ms blocked) |
| `cuMemcpyDtoHAsync` | 49 | 49 |
| `cuMemcpyHtoDAsync` | 260 | 260 |
| GPU kernel time | 62.4 ms | 48.6 ms |

| kernel | baseline ms / launches | final ms / launches |
|---|---|---|
| attention projections (`kquant_mmvq_q2_k_w`, + `kquant_mmvq_q2_k_groups_w` for wo_a) | 13.04 / 596 | 10.03 / 276 + 2.43 / 40 |
| `kquant_mmvq_q2_k_experts_w` (gate, up) | 10.14 / 79 | 9.80 / 39 (`_w8`, gate+up in one) |
| `kquant_mmvq_q3_k_experts_w` (down) | 8.95 / 39 | 7.49 / 39 (`_w8`) |
| LM head (`dense_gemv_bf16` instance, then `kquant_mmvq_q6_k_w`) | 5.5 / 1 | 2.68 / 1 |
| other `dense_gemv_bf16` | 2.4 / 21 | 2.82 / 21 |
| router GEMV (`moe_v41_router_gemv_f32out`, then `_staged`) | 5.37 / 40 | 2.74 / 40 |
| HC chain (`mixes_finish` + `hc_post` + `mixes_dot` + `collapse`, then `mixes_dot` + `finish_collapse` + `post_wide`) | 7.41 / 320 | 2.75 / 240 |
| `attn_v41_slice_cols` + `attn_v41_scatter_cols` | 1.45 / 640 | 0 / 0 |
| `kquant_q8_1_rows_bf16` | 0.84 / 714 | 0.52 / 435 |
| `attn_v41_sparse_attn` | 2.07 / 40 | 2.08 / 40 |

What remains, in order of size: the expert reads (17.3 ms against the 11 ms floor), the
attention projections (12.5 ms over 316 launches, the wq_a / wkv pair and the indexer
projections still one launch per tensor), and the host round-trips (186 waits: the router
and indexer read-backs, the per-token engram row upload, the final collapse; every blocking
`copy_d2h` / `copy_h2d` in the CUDA backend is an async copy plus a stream sync). A
device-side expert plan needs a replay protocol for cache misses, which is why the top-k
stayed on the host and the whole-step graph was not re-tested.

## Phase 2 (PR #1148, branch `ds41-decode-retune-2`)

### S0, the byte floor

One decode token reads 6.09 GB of weights (routed experts 240 x 12.22 MiB = 2.93 GB;
attention q_b 550 + o_b 550 + o_a 440 + q_a 86 + kv 34 MB; shared experts 512 MB; LM head
543 MB; router bf16 157 MB; engram wkv 103 MB). Measured on this GB10 (`int4` streaming
read, best of five): device memory 249 GB/s, the GPU reading the page-locked expert arena
223 GB/s. Floor = 2.93/223 + 3.16/249 = 25.8 ms = 38.7 tok/s at 100% of the read ceiling
with no gaps, no waits, no compute. The single-token 40 tok/s target sits below it.

### S1, the misses (measured 09-19, `ATLAS_DS41_ROUTE_TRACE` on the standard)

At 88 GiB (7,376 slots) 1,792 of 1,794 decode steps miss: 22,191 misses = 12.4 a step =
5.2% of the 240 accesses; hit steps 53.6 ms, miss steps 79.8 ms. One 12.22 MiB expert reads
in 1.8-2.0 ms however it is split (the NVMe's 11 GB/s needs several experts in flight), so
12.4 x 2 ms is the whole gap to the hot number. The static top-7,376 set covers 95.2% of
accesses: the misses are the tail, not an LRU artifact (LFU, 2Q, LRU-2 no better; S3-FIFO
-26% at 88 GiB but worse at 100 GiB; Belady 2x better only because r1/r2/r3 repeat).

| configuration | MinHeap tok/s | Volvo tok/s | decode misses | hit / miss step |
|---|---|---|---|---|
| 88 GiB, 8 readers (phase 1 final, reproduced) | 12.38 | 12.85 | 22,191 | 53.6 / 79.8 ms |
| **100 GiB, 16 readers** (kept: the recipe) | **14.20** | **17.73** | 12,255 | 55.9 / 74.9 ms |
| 100 GiB, 16, reader pool, no prefetch | 13.79 | 17.78 | 12,255 | |
| 100 GiB, 16, pool + prefetch K=4 (d=1) | 12.50 | 16.67 | 8,240 | 60.7 / 80.7 ms |
| 100 GiB, 16, pool + prefetch K=6 (d=1) | 13.63 | 16.39 | | |
| 100 GiB, 16, pool + prefetch K=12 (d=1) | killed | | | ~1.8 s a step |

All byte-identical to the oracle (6/6). The 100 GiB arena (8,380 slots; MemAvailable ~10 GB
during load, ~17 GiB of page cache given up, hit steps +2.3 ms from the engram row reads)
halves the misses. Prediction (the next layer's router on this layer's MoE input, d=1):
top-6 covers 67.6% of the picks and 48.5% of the misses, top-12 81.8% / 69.1%; but the COLD
part of a prediction, the reads a prefetch issues, has 36% precision at top-4 (6 reads a
token, 2.3 misses caught), 23% at top-6 (15 reads), 8% at top-12 (58 reads). Built (reader
pool with urgent / background lanes and a reserve, per-slot tickets, speculative slots
demoted when unused) and measured: the predictor's router launch costs 2.7 ms a token of
GPU time and the background reads slow the remaining misses on the shared disk, so every
step got ~5 ms slower while the misses fell 12,255 -> 8,240. Shipped off by default
(`ATLAS_DS41_READER_POOL=1 ATLAS_DS41_PREFETCH_K=4` to enable); worth revisiting once the
router GEMV is cheap (S3) and with a cap on background reads in flight.

### S4, the speculative ceiling, measured before any engine work

`~/code/atlas-notes/bin/ds41_lookup_sim.py` replays the oracle texts through an n-gram
lookup drafter (the longest match of the last 2..6 tokens in prompt + generated proposes
the K tokens that followed it; the greedy-matching prefix is accepted; a step emits
accepted + 1). MinHeap: 1.07 / 1.10 / 1.10 / 1.11 tokens a step at K = 1 / 2 / 4 / 8
(drafts fire on 41-51 of ~270 steps); with 1..4-token matches 1.13-1.20 (drafts on 125
steps, 5-24% of drafted tokens accepted). Volvo: 1.03-1.09 at every setting. The texts are
fresh code and prose with almost no repeated n-grams. A K+1-token verify step reads the
union of the tokens' experts (up to (K+1) x 6 a layer out of 384, little overlap), so the
MoE half of the step grows with K while only the dense 3.16 GB is amortised: at 1.1-1.2
accepted tokens a step the verify costs more than it returns. No MTP / next-n tensors ship
in the GGUF (S0), so there is no free draft head. Lookup-draft speculation cannot carry this
suite toward 40 tok/s; the number is stated here so nobody builds it for that reason.

### S1 addendum: the working-set edge, the policy, and long outputs

Per request, decode misses a step at 100 GiB (8,380 slots), strict LRU: MinHeap r1 17.6
(a cold cache), r2 5.54, r3 5.54; Volvo r1 12.3, r2 0, r3 0. MinHeap's working set is 8,418
distinct (layer, expert) pairs (prefill 3,151 + decode), 38 over the cache; Volvo's is 8,005
and fits. Replaying a working set 0.5% larger than the cache in the same order is LRU's
pathological case: at every miss it evicts exactly the expert needed next. That, not the
kernels, is why MinHeap trailed Volvo (hit steps are the same 56-57 ms on both). The union of
the two suites is 11,223 pairs with 5,200 in common, so every switch between suites costs
about 3,000 misses (r1 of each: 17.6 and 12.3 a step); the standard's medians of three absorb
that by design.

Two answers, both measured:

* **102 GiB (8,548 slots)**: MinHeap r2/r3 fall to 0 misses and 17.59 / 18.16 tok/s, but the
  box swaps (16 GB swap file, `pswpout` climbing): Volvo r2 ran at 14.07 tok/s with zero
  misses and MinHeap's TTFT median was 11.7 s. 100 GiB is the safe maximum here; this is the
  working-set edge of one prompt, not a general result.
* **Policy** (simulated on the 100 GiB trace, then built): a random victim among the oldest 5%
  of the LRU list (`ATLAS_DS41_EVICT_RANDOM_PCT`, default 5; 0 = strict LRU) breaks the
  lockstep: MinHeap r2/r3 1.17 / 1.06 misses a step, Volvo unchanged at 0 / 0, total decode
  misses 12,262 -> 9,627. The oldest 10%: 0.75 / 0.74 but Volvo r2 0.20. CLOCK 4.79, SLRU worse,
  pure RANDOM 0.34 but Volvo 9.0 / 6.0, Belady 0.13 / 0.13. The policy never touches the math.
  Measured on the standard: decode misses 12,255 -> 9,914, MinHeap r2 / r3 misses a step 1.90 /
  1.19, step-wall medians 61.8 / 58.7 ms (were 69.3 / 69.6), Volvo unchanged (0 / 0, 17.87 tok/s),
  6/6 byte-identical. Kept (`ATLAS_DS41_EVICT_RANDOM_PCT`, default 5).

**The stalls.** The suite's tok/s is a mean, and at 100 GiB the runs carry single steps of 1.9
to 17.6 s with only 3-11 misses in them (one 17.65 s step in MinHeap r2 of the 100 GiB run;
2.9 + 1.9 + 10.5 s across the three MinHeap requests of the policy run), always in the first
three requests, while `pswpout` climbs by ~100k pages a run: the kernel reclaiming page cache
into the 16 GB swap file as the arena's pages get touched, not the miss path. Without them
MinHeap r3 of the policy run is 300 x 59 ms = ~17 tok/s. Fix: `posix_fadvise(DONTNEED)` on the
expert range after every pread, so the arena's reads never leave page cache for reclaim to
fight over. Measured: the longest MinHeap r2 / r3 step fell from 1.9 / 5.3 s to 107 / 115 ms
(the 26 steps over 150 ms left are all in r1, the cold cache); MinHeap 12.76 -> 15.79 tok/s.

**Long outputs** (one MinHeap request, 1,500 requested, EOS at 993 tokens, 100 GiB, cold
cache): cumulative distinct experts 3,151 after the prefill, 8,423 after 300 tokens, 9,700
after 600, 10,385 after 900, 10,650 at 992, still growing ~3 a token toward the 15,360;
misses a step per 300-token window 17.6 (cold) / 5.4 / 4.9 / 8.7 (last 92). One Spark keeps
missing on long outputs: the all-resident case is two Sparks with 187.7 GB of experts split
94 GB each into a 100 GiB arena, zero misses.

**The miss path itself**: one 12.22 MiB expert costs 1.8-2.0 ms buffered (8 parts); O_DIRECT
at 4 parts reads it in 1.76 ms in isolation (-12%), but every expert stack sits at file offset
160 mod 512 (GGUF alignment 32), so O_DIRECT needs a padded slot layout: deferred. GPUDirect
Storage: the `nvidia-fs` module is present but not loaded and there is no system `libcufile`:
not tested. A device-memory arena (249 vs 223 GB/s on the 2.93 GB of expert reads a token,
1.4 ms) is not reachable by `pread`: `cudaMalloc` memory is not CPU-accessible on this GB10
(SIGSEGV), so it would need a pinned staging ring and an H2D per miss: deferred.

### What passes 40 (the S0 arithmetic on other hardware)

* **Two GB10s, 2-way expert parallel** (each Spark holds half the routed experts, the dense
  weights replicated): routed bytes per GPU 2.93 / 2 = 1.465 GB at 223 GB/s = 6.6 ms, dense
  3.16 GB at 249 GB/s = 12.7 ms, **19.3 ms = 51.9 tok/s** ceiling before the interconnect. The
  exchange is one 5120-wide bf16 row (10 KB) per layer per direction, 40 layers a token: to
  stay above 40 tok/s (25.0 ms a step) the 5.7 ms left buys **~140 us per layer round trip**
  (~70 us a one-way message), against ~10 us for an RDMA message on the ConnectX-7 link: the
  interconnect is not the limit if the step overlaps the exchange with the dense half and
  keeps one message per layer per direction. And with 187.7 GB of experts split 94 GB a Spark
  into a 100 GiB arena, every expert is resident: zero misses on any prompt length.
* **One B200**: 6.09 GB at ~8 TB/s HBM3e = 0.76 ms a token: the step is launch-bound (1,866
  launches x ~3 us = 5.6 ms eager, ~180 tok/s; a whole-step graph puts it in the hundreds).
  That is the #1140 / #1141 campaign (one GPU, then the two-GPU expert split), where the
  bit-exact device-side selection prototyped here (S2, `route_select_dev.cu`: glibc's `expf` /
  `log1pf` ported and verified over all 2^32 inputs, 20,000 tokens of picks, weights and plan
  identical) is what makes a whole-step graph possible.

### S2 and S3, time-boxed

* **S2.** The per-layer CUDA graph path (`ATLAS_DS41_GRAPH=1`, phase 1's segments A / host span /
  B) loses on the S1 binary: MinHeap 9.77 (7.08 / 12.97 / 9.77), Volvo 15.19 (11.91 / 15.19 /
  17.63) against eager 14.20 / 17.73, and erratically. A whole-step graph with one miss-flag
  read-back a step does not pay on one Spark either: 65% of decode steps miss, and a miss at
  layer L wastes the 40 - L layers replayed after it. The piece that IS needed for a whole-step
  graph on hardware where everything is resident, a bit-exact device-side selection, is
  prototyped and verified in `docs/perf/ds41_prototypes/` (glibc's `expf` / `log1pf` on the
  device, 0 mismatches over all 2^32 inputs; the top-6, weights and plan identical to the host
  on 20,000 tokens). The host span stays as it is: the syncs remain one per layer.
* **S3.** Two kernel attempts inside the time box, both out: the single-token K-quant GEMV
  (`kq_mmvq_warp_s`, the attention projections' 316 launches a token at 133-152 GB/s) with
  its super-block loop unrolled eight deep on one accumulator passed the oracle tests but
  measured nothing (`kquant_mmvq_q2_k_w` 10.82 ms a token against 10.03, kernel time 49.6
  against 48.6 ms: the launches are not waiting on dependent loads), reverted; the router
  chain double-buffered across trips faulted (`CUDA_ERROR_ILLEGAL_ADDRESS` in the MoE oracle
  test), reverted. The attention projections' bandwidth (1.66 GB in ~13 ms) remains the
  largest kernel inefficiency, and it needs a shape change (rows per block, block reads
  vectorised across the 84-byte Q2_K super-blocks), not an unroll.

### Phase 2 results

Chat, 300 tokens, temperature 0, one serve, medians of three, every text byte-identical to
the oracle (6/6 on every row). Recipe from S1 on: `ATLAS_DS41_EXPERT_CACHE_GIB=100
ATLAS_DS41_READER_THREADS=16` on a Spark running nothing else.

| step | commit | MinHeap tok/s (r1/r2/r3) | MinHeap TTFT | Volvo tok/s (r1/r2/r3) | Volvo TTFT | decode misses |
|---|---|---|---|---|---|---|
| baseline (09-17 binary, 88 GiB, 8 readers) | f13752b23 | 10.42 | 2048 ms | 10.77 | 1291 ms | |
| phase 1, five levers (#1147) | c5ca8e2d5 | 12.39 | 2009 ms | 12.90 | 1262 ms | 22,191 |
| phase 2, reproduced with the route trace | ece8acb34 | 12.38 (10.77/12.42/12.38) | 2013 ms | 12.85 (11.43/12.85/12.89) | 1245 ms | 22,191 |
| S1: 100 GiB arena, 16 readers | 7866373ea | 14.20 (10.76/14.21/14.20) | 4560 ms | 17.73 (12.03/17.73/18.45) | 432 ms | 12,255 |
| S1: + random victim among the oldest 5% | | 12.76 (10.79/16.07/12.76), swap stalls | 4086 ms | 17.87 (12.02/17.87/18.32) | 433 ms | 9,914 |
| **S1: + page cache dropped after every expert read** | 998bf8986 | **15.79** (9.99/15.79/16.40) | 1569 ms | **17.67** (11.22/17.67/17.97) | 399 ms | 9,914 |

Baseline to phase 2: MinHeap +51.5%, Volvo +64.1%; phase 1 to phase 2: +27.4% / +36.9%.
Hit steps 56-57 ms (17.6-17.9 tok/s) on both suites; MinHeap's median still carries 1.2-1.9
misses a step at 100 GiB (the working-set edge).

Profile of the final binary (nsys, one warm 60-token request, 19.86 tok/s hot): GPU kernel
time 49.6 ms a token over 1,865 launches; 186 `cuStreamSynchronize` (28.6 ms blocked, most
of it the GPU finishing queued work), 49 D2H, 260 H2D; experts 17.5 ms (`q2_k_experts_w8`
10.53 + `q3_k_experts_w8` 6.96), attention projections 10.82 + 2.34 + 1.95, head 2.88, router
2.73, sparse attention 2.06, HC chain 2.54. Unchanged from phase 1 within the run, as S2 and
S3 landed no kernel.

## Phase 3 (PR #TBD, branch `ds41-decode-retune-3`): the hit step

### What the microbench and the trace said before any lever

The brief's P1/P2 (the K-quant GEMVs latency-bound on dependent super-block loads; issue the
loads first) was tested first, in a scratchpad microbench that includes `kquant_moe.cu`,
rotates the weights through a ring larger than L2 and compares every variant byte for byte
against the shipped kernel at the exact V4.1 decode shapes. Register-prefetch variants of
`kq_mmvq_warp` (4 / 8 / 16 super-blocks a lane in flight, 1 / 2 / 4 / 8 warps a block) were
bit-identical and slower at every shape (+4 to +55%); the shipped kernels run at 150-215 GB/s
isolated (wq_b 179, wo_b 203, groups 186, experts 213 / 206 GB/s; the small grids wkv 100 and
wq_a 154) against the 249 GB/s ceiling. The thesis is dead; nothing of it landed.

The per-launch trace of the phase 2 hit step (30 warm tokens of the final nsys, one token =
2,213 GPU ops) placed the time instead:

| per warm token, phase 2 final | ms |
|---|---|
| wall / GPU busy / idle (the host span, ~67 us a layer) | 50.6 / 47.9 / 2.7 |
| routed experts (`q2_k_experts_w8` 267 us + `q3_k_experts_w8` 176 us a layer) | 17.7 |
| attention projections (`q2_k_w` wq_b 89.5, wo_b 76.9, wq_a 16, wkv 7.5 us; `groups_w` 57) | 9.9 + 2.3 |
| LM head `q6_k_w` | 2.9 |
| engram `wkv` as bf16 (`dense_gemv_bf16`, 25600 x 6144, 315 MB a layer x 2, 245 GB/s) | 2.6 |
| router `_staged` (64.7 us for a 4 MB weight: a 5,120-step sequential chain on 2 lanes a block) | 2.6 |
| sparse attention | 2.4 |
| shared expert (w1 + w3 1.85, w2 1.2) | 3.1 |
| HC `finish_collapse` (19.3 us: the Sinkhorn finish on one thread) + `mixes_dot` (12.4 us) | 1.5 + 1.0 |
| `index_score` (one block, 122 us x 8) + compressor `gemm_f32` (32 blocks, 144 us x 6) | 1.0 + 0.9 |

The routed experts read the page-locked arena: the same kernel reads a device-memory ring at
213 GB/s, a default pinned ring at 178-180 and a write-combined pinned ring at 184-196 (the
in-step 267 us a layer is the pinned number exactly). Managed memory reads at 166; pageable
memory faults (no HMM on this GB10); a device arena would cost 213 us of H2D per 12 MiB miss
(59 GB/s copy engine).

### The levers, kept or reverted, each on a full standard with 6/6 texts byte-identical

| step | commit | MinHeap tok/s (r1/r2/r3) | MinHeap TTFT | Volvo tok/s (r1/r2/r3) | Volvo TTFT | oracle |
|---|---|---|---|---|---|---|
| p3base = #1148 rebased on main | 9b0748d04 | 16.19 (10.29/16.19/16.74) | 1212 ms | 18.21 (11.43/18.21/18.37) | 423 ms | 6/6 |
| L1 arena write-combined, REVERTED (neutral: A/B on the L3 binary 17.75/20.14 vs 17.58/19.75) | 689787c4f, 4a30b6b2c | 16.37 (10.30/16.37/16.47) | 1209 ms | 17.82 (11.46/18.18/17.82) | 462 ms | 6/6 |
| L2 shared expert on a side stream under the router and the host span | 758df0dde | 17.11 (10.63/17.11/17.62) | 1213 ms | 19.34 (11.92/19.34/19.40) | 406 ms | 6/6 |
| L3 compressor product staged at m = 1; index score in registers | c87a65510 | 17.75 (10.86/17.75/18.39) | 1361 ms | 20.14 (12.24/20.14/20.24) | 422 ms | 6/6 |
| L4 engram `wkv` on raw Q2_K (q8_1 rows), DROPPED | (not committed) | 19.62 (11.29/19.62/20.66) | 1098 ms | 21.55 (12.70/21.55/21.72) | **0/6** |
| L5 HC finish on 16 lanes + L6 router and compressor chains from precomputed products | c1fcb64a4 | **18.43** (11.06/18.43/18.94) | 1214 ms | **20.81** (12.46/20.81/20.92) | 399 ms | 6/6 |

Phase 3 against its baseline: MinHeap +13.8%, Volvo +14.3%; against phase 2 as published
(15.79 / 17.67): +16.7% / +17.8%; against the 09-19 morning baseline (10.42 / 10.77): +77% /
+93%. The phase target of 22 / 25 was not reached; what stands between 18.4 / 20.8 and it is
below.

L1: `cuMemHostAlloc(DEVICEMAP | WRITECOMBINED)` for the arena. Faster isolated, inside the
band on the standard in both directions; reverted so the branch carries only levers that moved
the number (the allocator entry point stays in the history).

L2: at decode the shared expert depends only on the MoE input; it now runs on a side stream
(its own q8_1 scratch) from an event recorded before the router launch, and the main stream
waits on its event before the same `accumulate` + `finish`. The order of additions into `acc`
is unchanged (routed rows in plan order, then the shared expert): phase 2's bits.
`ATLAS_DS41_SHARED_SIDE=0` restores the serial order.

L3: `attn_v41_gemm_f32` (the 16x16 tile at m = 1: 32 blocks, 144 us) becomes one output a
block with the strict k chain in the tiled kernel's order; `attn_v41_index_score` (one block,
122 us) keeps the key row in registers and reads the query 16 bytes at a time when `ihd` is
128. Same products, same order, same bf16 roundings.

L4: the engram projection ships as Q2_K (52 MB a layer) and is expanded to bf16 (315 MB a
layer, 2.6 ms a token to read). The K-quant GEMV on the raw blocks needs the 25,600-wide
engram row quantised to q8_1 first, and that rounding is a different model: 0/6 texts
identical (MinHeap diverges at character 100, Volvo at 51). Out of the branch. What would pass
the oracle is a bf16-activation Q2_K GEMV (the weight block dequantised in registers, f32
products in the tiled kernel's order), ~2 ms a token, not built here.

L5: the Sinkhorn finish of each HC site ran on one thread (17 us a call, 80 sites a token);
`hcv_finish_lanes<HC>` puts one element on each of the 16 lanes, gathers row and column sums by
shuffles in the serial order, keeps the max, `expf` and every division per element: the serial
form's bits, 19.3 -> 4.5 us a launch.

L6: the router's strict 5,120-step chain ran on two lanes a block (64.7 us for a 4 MB weight).
The block's 256 threads now write the products into shared memory (bf16 x bf16 is exact in
f32) and one lane adds them in k order eight float4 at a time from registers: no load on the
FADD chain, 64.7 -> 41.2 us in the step, bit-identical. The compressor product of L3 rewritten
the same way: 95 -> 64 us.

### The final profile (nsys, one warm 60-token request, same method; receipts `ds41_nsys_decode_final3*`)

| per token | phase 2 final | phase 3 final |
|---|---|---|
| profiled request | 19.86 tok/s | 22.79 tok/s |
| wall / GPU busy / idle gaps (30 mid-request tokens) | 50.6 / 47.9 / 2.7 ms | 44.2 / 45.0 / 1.6 ms |
| kernel launches | 1,865 | 1,865 |
| `cuStreamSynchronize` | 186 (28.6 ms blocked) | 186 (25.3 ms blocked) |
| `cuMemcpyDtoHAsync` / `HtoDAsync` | 49 / 260 | 49 / 260 |
| routed experts (`q2_k_experts_w8` + `q3_k_experts_w8`) | 10.68 + 7.05 | 10.28 + 7.18 |
| attention projections (`q2_k_w` wq_b + wo_b + wq_a + wkv, `groups_w`) | 3.58 + 3.08 + 0.64 + 0.30, 2.29 | 3.11 + 2.82 + 0.60 + 0.36, 2.75 |
| shared expert (`q2_k_w` x2 + `q3_k_w`) | 1.85 + 1.20 | 1.82 + 1.68 (now overlapped with the host span) |
| LM head `q6_k_w` | 2.88 | 2.78 |
| engram `wkv` (`dense_gemv_bf16` 6400x1) | 2.57 | 2.63 |
| sparse attention | 2.37 | 2.41 |
| router (`_staged` -> `_products`) | 2.59 | 1.65 |
| HC `mixes_dot` + `finish_collapse` | 0.99 + 1.54 | 1.00 + 0.36 |
| compressor (`gemm_f32` -> `gemv_f32_staged`) + `index_score` | 0.86 + 0.97 | 0.38 + 0.16 |

GPU busy exceeds wall because the side stream overlaps the main one. The step is now 44 ms of
GPU time, 41 of it in seven bandwidth-bound GEMV families reading 6.09 GB a token at 150-215
GB/s of the 249 measured, plus 1.6 ms of host span. The honest gap to 22 / 25 on the standard
is (a) those families at their present efficiency (the routed experts alone are 17.5 ms for
2.93 GB, 167 GB/s from the pinned arena; the same kernel reads device memory at 213), (b) the
1-2 residual misses a step on MinHeap at 100 GiB (~2 ms each), and (c) the 186 host waits a
step, whose cure (device routing, a whole-step graph) phase 2 showed does not pay while 65% of
steps miss. None of the three is a kernel rewrite of the kind tried here.

## Phase 4 (PR #1155, branch `ds41-decode-retune-4`): the bytes and the waits

Baseline `p4base` (f1aacfc01, the phase 3 head rebuilt and re-measured): MinHeap 18.27 (11.05 / 18.27 / 18.94), Volvo 20.76 (12.46 / 20.76 / 20.89), 6/6 byte-identical.

### What was measured before the first lever (Nsight Compute, standalone microbench)

The real expert kernels at the real per-layer shapes, one layer's six experts (76.9 MB), a ring of eight layers larger than L2, from three kinds of memory: the page-locked arena as allocated today 176 / 169 GB/s (gate-up / down), `cudaMalloc` 215 / 209, `cuMemHostAlloc` DEVICEMAP 174 / 168. Under `ncu` the pinned and device runs show the same L2 hit rate (62.6 vs 63.0%), the same L1 hit (89.7 vs 89.6%), the same achieved occupancy (87.9 vs 92.6% of a theoretical 100), the same stall reason (long scoreboard, 82 vs 78% of stall cycles) and differ only in duration (265.8 vs 232.2 us): the path, not the kernel. One 12.22 MiB expert copies pinned-to-device in 217 us (59 GB/s) on its own stream; back-to-back copies slow a concurrent layer GEMV by 23-27%. A 100 GiB `cudaMalloc` arena plus 3 GiB and a 16-slot pinned ring leaves 14.3 GiB `MemAvailable` with swap flat: the same footprint as the pinned arena.

### P3, the expert cache in device memory (96641e79c): KEPT

`ATLAS_DS41_ARENA_DEVICE` (default on, `=0` restores the page-locked arena exactly), `ATLAS_DS41_STAGING_SLOTS` (16). Device slots off the allocation ledger (the KV budget reads the ledger as Atlas-own memory); misses read by the scoped reader threads into a half of the pinned ring, then one `cuMemcpyHtoDAsync` per miss into the device slot on the compute stream ahead of the token's launches (stream order is the synchronisation; a ring half is rewritten only after the event behind its last copies). The GEMV reads the same bytes at another address.

| | MinHeap | TTFT | Volvo | TTFT | oracle | hot probe |
|---|---|---|---|---|---|---|
| p4base | 18.27 (11.05 / 18.27 / 18.94) | 1209 | 20.76 (12.46 / 20.76 / 20.89) | 421 | 6/6 | 22.79 |
| P3 | 18.57 (10.83 / 18.57 / 19.15) | 1285 | 21.14 (12.28 / 21.14 / 21.26) | 400 | 6/6 | 23.72 |

nsys (one warm token): expert GEMVs 17.15 -> 15.23 ms (233 / 154 us a layer from device against 256 / 180 from pinned; the isolated bench's 216 / 146 is the 100 GiB random reach), GPU kernel time 46.4 -> 44.2 ms, wall 44.15 -> 42.42 ms, launches / syncs / D2H unchanged at 1,865 / 186 / 49. In the serve `MemAvailable` is 11.0 GiB after the arena and 9.3 GiB live.

### Where the step is bound after P3 (the trace, token 35, one layer of 895 us)

GPU busy 42.8 ms of a 42.4 ms wall (the shared expert overlaps on its side stream), idle gaps 1.6 ms a token. One layer, nearly gap-free: router 41 us (under the shared expert), the routing read-back and the host's selection hidden under the shared expert's 45 us down projection, the routed experts 228 + 153 = 380 us (43% of the layer), HC 22, attention projections 243 (`wq_b` 79 on 8,192 blocks, `wo_a` groups 67, `wo_b` 72, `wq_a` 16, `wkv` 9), sparse attention 69 on 64 blocks. The host waits (186 a token) are no longer on the critical path at 1.6 ms of idle; the step is the bytes the kernels read: experts at 205 GB/s now, attention projections at 130-170. What passes this on one Spark is fewer bytes (a smaller expert quant, which changes the numbers) or a second Spark (every expert resident); the B200 plan (#1140-#1142) carries the same kernels with 8 TB/s under them.


## Phase 5 (PR #1160, branch `ds41-decode-retune-5`): the last of one Spark

Start: 38c4c2097 (PR #1156's head: the sm_100a gate, device routing kept, segment graphs default off). The standard gains `ATLAS_DS41_DEVICE_ROUTE=1` (kept in #1156, byte-identical). Baseline `p5base` (38c4c2097 rebuilt and re-measured the same evening): MinHeap 18.17 (10.58 / 18.17 / 18.77), Volvo 20.60 (11.92 / 20.60 / 20.70), 6/6 byte-identical; the S2 binary re-run under the same env at the same hour gave 18.18 / 21.56, so the box ran about a token a second under the afternoon's 19.19 / 21.87 and the phase's deltas are against `p5base`.

### L1, the miss path (a94bedaa3): KEPT

Measured first. One real expert of layer 3 (gate 3,870,720 B + up 3,870,720 B Q2_K, down 5,068,800 B Q3_K; per-expert strides 0 mod 512, tensor bases 416 mod 512) read by 16 threads from the shard: buffered `pread` + `fadvise(DONTNEED)` 2.27-2.69 ms at 12-16 parts; `O_DIRECT` in 512 B aligned windows straight into the destination 1.37-1.42 ms, flat from 4 to 24 parts. Eviction on the phase 2 route trace at 8,381 slots (misses a step, MinHeap r2 / r3 / Volvo r2): LRU 5.53 / 5.53 / 0.00; random victim among the oldest 5% (kept in phase 2) 1.14 / 1.03 / 0.00; 7% 0.84 / 0.80 / 0.01; 8% 0.78 / 0.80 / 0.01; 10% 0.68 / 0.75 / 0.21; 15% 0.57 / 0.56 / 0.99; CLOCK 4.79 / 4.79 / 0.15; S3FIFO, TWOQ, SLRU, LRU2 worse on Volvo; BELADY 0.12 / 0.12 / 0.00.

Built: every shard also opened `O_DIRECT` (`ATLAS_DS41_DIRECT_READS=0` restores the buffered path); the staging ring slot padded per segment so each segment lands at the residue of its file offset mod 512; windows cut 512-aligned in file space (disjoint; the head rounds down into the slack, the tail rounds up, a tail past the end of the file tolerated up to the tensor's bytes); three stream-ordered copies into the unchanged device slot (`expert_reads.rs`, `expert_direct.rs`). The victim window at 8% (`ATLAS_DS41_EVICT_RANDOM_PCT`). Not built: launching the resident experts before the misses land (~0.3 ms a miss layer once the read is 1.4 ms; byte-identical since the per-expert rows are independent until `sum_rows`, but a two-phase fetch API for well under 1%).

### L2, the engram `wkv` projection off its raw Q2_K blocks (36752d456): KEPT

Layers 1 and 14 project the 24 looked-up rows through `wkv` `[25600 x 6144]`, shipped Q2_K (49.2 MiB a layer) and expanded to bf16 (300 MiB a layer) for the prefill GEMM; the single-token step read the expansion, 600 MiB and 3.0 ms a token. `engram_v41_wkv_q2k_gemv` reads the raw blocks and is `dense_gemv_bf16` over the expansion bit for bit by construction (same 64 threads an output, same uint4 groups in the same per-thread order, each weight dequantised as `dequant_q2_k_to_bf16` does it and rounded to bf16 before the product, same shuffle tree and two-warp sum; both modules `--fmad=false`), proved by a GPU test against the bf16 GEMV over the device expansion of random blocks (390 x 6144, bitwise) before the suite. The raw blocks sit on the device beside the expansion (`ShardFiles::locate_q2k`, `deepseek_v41/engram_q2k.rs`; `ATLAS_DS41_ENGRAM_Q2K=0` keeps the bf16 path).

| | MinHeap | best | TTFT | Volvo | best | TTFT | oracle | nsys kernel ms | launches / syncs / D2H |
|---|---|---|---|---|---|---|---|---|---|
| p5base 38c4c2097 | 18.17 (10.58 / 18.17 / 18.77) | 18.77 | 1296 | 20.60 (11.92 / 20.60 / 20.70) | 20.70 | 391 | 6/6 | 44.5 (S2) | 1905 / 186 / 49 |
| L1 a94bedaa3 | 19.31 (12.80 / 19.31 / 19.51) | 19.51 | 1234 | 20.89 (14.29 / 21.28 / 20.89) | 21.28 | 521 | 6/6 | 44.7 | 1905 / 186 / 49 |
| L2 36752d456 | 20.50 (13.35 / 20.50 / 20.98) | 20.98 | 1156 | 22.99 (15.00 / 22.99 / 23.00) | 23.00 | 487 | 6/6 | 42.3 | 1905 / 186 / 49 |

After L2 (kernel time a warm token, `ds41_nsys_decode_L2`): routed experts 14.8 ms (`q2_k_experts_w8` 8.99 at 229 us a layer, `q3_k_experts_w8` 5.78 at 147 us), attention projections 9.9 (`q2_k_w`, 276 launches) + 2.75 (`groups_w`) + 2.27 (`q3_k_w`), head 2.77, sparse attention 2.10, router 1.71, HC 1.4, quantisation 0.8, engram 0.6.

### L3, the expert GEMV latency chain: REVERTED TWICE, the phase stops here

Bench first (`bench_kq3.cu`, the real per-layer shapes, one layer's six experts from device memory, an 8-layer ring, every variant bitwise identical to the shipped `_w8` entries on the layer-0 outputs): shipped 217-222 us gate + up (Q2_K, 210 GB/s) / 186-191 us down (Q3_K, 163 GB/s) = 404-413 us a layer; two rows a warp with each row's own serial chain (`kq_mmvq_warp_m1x2`, scalar accumulators) 219-220 / 151-152 = 371 us (-8.6%, all of it in down); three or four rows a warp, 4 or 16 warps a block, and the two-row form over `tmp[KQ_MAX_M]` stack arrays were all slower. ncu: Q3_K 202.5 -> 157.8 us (40 -> 48 registers, occupancy 100 / 96.6% -> 83.3 / 72.7%, 27.2 -> 23.6 warp cycles an issued instruction); Q2_K 239 -> 222 us at the same occupancy loss. (One harness process launching two kernel variants deadlocked in a driver spin lock inside `cuLaunchKernel`; the bench runs one variant a process.)

On the standard the isolated gain did not survive the serve, where the experts come from anywhere in a 100 GiB arena: two rows a warp on both projections MinHeap 20.09 (13.14 / 20.09 / 20.62), Volvo 22.48 (14.75 / 22.48 / 22.51); on the down projection only 19.90 (13.09 / 19.90 / 20.40), 22.27 (14.66 / 22.27 / 22.32); both 6/6 byte-identical and both under L2, whose binary re-run between them gave 20.32 (13.29 / 20.32 / 20.83), 22.83 (14.94 / 22.83 / 22.88). Two arms not kept: L3 reverted, L4 not attempted. The kernel source is kept in the phase record for the B200, where the same chain runs under 8 TB/s.

### Phase 5 closing line

38c4c2097 -> 36752d456: MinHeap 18.17 -> 20.50 (best 20.98), Volvo 20.60 -> 22.99 (best 23.00), every text byte-identical to the phase 1 oracle; GPU kernel time a warm token 44.5 -> 42.3 ms; the miss read 2.3-2.7 -> 1.4 ms and 0.3 fewer misses a step on MinHeap. Against the 09-17 standard of 10.6 / 10.9 the one-Spark line now stands at +93% / +111%.
