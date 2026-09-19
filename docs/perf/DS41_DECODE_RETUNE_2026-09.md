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
