# Where the attention Q/K/V decode GEMMs go (#927, H100)

**Headline: a 5 MB weight read cannot amortise a launch.** At `n = 16` each of
the 16 full-attention layers issues THREE cuBLASLt W8A8 GEMMs at `K = 5120`; ONE
at `N = 14336` deletes two launches per layer and pays the partial wave once.
Lever `[defaults] attn_qkv_fused` (hopper `true`, gb10/b200 `false`);
`ATLAS_ATTN_QKV_FUSED=0` kills.

## Anchors (given; nsys `--cuda-graph-trace=node`, 1xH100 80GB HBM3, Qwen3.8-27B-FP8, round 13 cell V @ `3717cb05e`, `h100-r13-attribution.md` §§C.2–C.4)

Median `n = 16` step: **19.887 ms** busy, 1 619 nodes, ONE graph launch. Byte
model at `M = 16`: `K·N + (K/128)·(N/128)·4`.

| arm | K | N | nodes | µs (µs/node) | GB/s | % HBM |
|---|---:|---:|---:|---:|---:|---:|
| attn `q_proj` (gated `[Q\|gate]`) | 5120 | 12288 | 16 | 460.0 (28.75) | 2 189 | 65.3 % |
| **attn `k_proj` + `v_proj`** | 5120 | 1024 | **32** | **510.9 (15.97)** | **328** | **9.8 %** |
| FFN `down` (control, same step) | 17408 | 5120 | 64 | 2 384.0 (37.25) | 2 393 | 71.4 % |

## The mechanism and the saving

One k/v node moves **5.24 MB**; `down` moves 17× that in twice the time. At a
128-wide N tile a k/v node is `8 × ceil(16/128) = 8` tiles on **132 SMs** — 124
idle for the whole launch, twice per layer. `q_proj` is 96 tiles, under one wave;
concatenated, 112 tiles is one wave and the two short launches vanish into it.
The bytes are read once either way, so this is not a traffic saving.
`now 970.9 µs/step`; `at 60 % HBM: 10.72 GB / (0.60 × 3.35 TB/s) = 543 µs`;
**saving 428 µs/step = 2.2 % of 19.887 ms**. Rank **5** of the round-13 decode
table, behind gate+up (1 476 µs, shipped), paged-decode split-K (986 µs), GDN
state decode (945 µs) and `o_proj` (400 µs).

**Round-17 prediction:** the `n = 16` decode step **19.887 → ≈ 19.46 ms** busy,
and `1024x256` C=16 TPOT **25.38 → ≈ 24.95 ms**. TPOT carries the step's
1.955 ms host gap unchanged — a kernel lever, not a scheduler one — so 428 µs
lands on the 22.186 ms span as-is; with the merged gate+up row, 1 904 µs (9.6 %).

## Numerics: a bit claim, not a tolerance

The fused weight is the three `[N_i, 5120]` E4M3 blocks appended along N; its
`[112, 40]` FP32 scale grid is the `[96, 40]` + two `[8, 40]` grids appended the
same way (both seams are block boundaries). Splitting N gives
**independent output columns over the same K with the same scales**: fused
element `(m, j)` is the same dot product, in the same order, as `q (m, j)` below
12288, `k (m, j−12288)` next and `v` above. `native_fp8_attn_qkv_fused_microtest`
asserts **byte equality** of the three slices over all 16 padded rows, of the
`deinterleave_qg` output and of the KV bytes on BOTH KV dtypes, with three
KNOWN_BAD controls.

## Layout — and why this fusion adds no kernel at all

`qkv_output` is ALREADY `[n, per_seq_qkv]` with Q at column 0, K at 12288 and V
at 13312, and `per_seq_qkv/2 == q_proj_dim + 2·kv_dim == 14336`
(`multi_seq/ctx.rs`). So the fused GEMM's `[m, 14336]` output at
`ldc = per_seq_qkv/2` **is that layout byte for byte**, and `ldc == N`.
`deinterleave_qg` already takes the row stride and runs over `qkv_buf` (no
strided variant needed, unlike gate+up's `silu_mul_strided`); the KV write
(`reshape_and_cache_flash*`, and the FP8 KV path with the #919 calibration
window) already reads K and V at those column offsets with row stride 14336, its
`key_stride`/`value_stride` unchanged; the paged decode kernel already reads Q at
column 0. N-concatenation for that reason, and because each projection stays
addressable as an un-fused `Fp8Weight` VIEW, so the GEMV tiers, the prefill
transposes and n=1 keep reading the same bytes — the NVFP4 wide-verify path has
fused on this identity since #915. **No new `.cu`, no `adds`, no new symlink.**

## Residency and scope

**Residency is net zero.** A second copy is 73.4 MB × 16 = **1.17 GB**. The
loader builds the fused buffer plus one appended scale grid, re-points
`q/k/v_weight` at views inside them, frees the three per-projection widened grids
the concat copied, and `prune_after_load` releases the three source store
tensors; `predicted_residency` prices it as a difference, so the preflight ring
fit is unmoved to the byte.

**Decode only, `n ∈ 5..=16`** — the W8A8 cuBLASLt arm's own band. n=1 keeps its
W8A16 GEMVs, which §C.5 measures at **2 567 GB/s = 76.6 % of HBM** across all
24.327 GB of weights: no launch headroom to buy. The PREFILL W8A8 arm keeps its
three GEMMs — it writes Q into `qkv_output` contiguous while K and V go to
`ssm_qkvz` (`prefill_qkv_w8a8.rs`), so a fused N needs a scatter, and at M=4576
they are compute-bound (§A.4). Widening either is a measurement, not an
argument.
