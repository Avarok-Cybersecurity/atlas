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
