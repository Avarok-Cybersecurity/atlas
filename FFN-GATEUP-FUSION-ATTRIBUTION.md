# Where the dense-FFN gate+up decode GEMMs go (#927, H100)

**Headline: the weight bytes are read once either way — the gap is the second
launch.** At `n = 16` the dense FFN issues TWO cuBLASLt W8A8 GEMMs per layer at
`K = 5120`, `N = 17408`; ONE at `N = 34816` halves the per-launch fixed cost and
doubles the tile count per wave. Lever `[defaults] ffn_gateup_fused` (hopper
`true`, gb10/b200 `false`); `ATLAS_FFN_GATEUP_FUSED=0` kills.

## Anchors (given; nsys `--cuda-graph-trace=node`, 1xH100 80GB HBM3, Qwen3.8-27B-FP8, round 13 cell V @ `3717cb05e`, `h100-r13-attribution.md` §§C.2–C.4)

Median `n = 16` step: **19.887 ms** busy, 1 619 graph nodes, ONE graph launch,
in-graph gaps 0.744 ms. Byte model at `M = 16`: `K·N + (K/128)·(N/128)·4`.

| arm | K | N | nodes | µs (µs/node) | GB/s | % HBM |
|---|---:|---:|---:|---:|---:|---:|
| **FFN gate + up** | 5120 | 17408 | **128** | **5 730.5 (44.77)** | **1 991** | **59.4 %** |
| FFN `down` | 17408 | 5120 | 64 | 2 384.0 (37.25) | 2 393 | 71.4 % |
| SSM `in_proj_qkvz` | 5120 | 16384 | 48 | 1 641.9 (34.21) | 2 453 | 73.2 % |

One gate/up node moves **89.1 MB** — what `down` moves in one launch, 7.52 µs
faster. The LM head hits **80.2 %** of HBM in this step: that is the target.

## The mechanism

128 nodes = 64 layers × 2, reading *different* weights: fusing saves no traffic.
It removes one launch per layer and one partial wave. At `N = 17408`, tile
`128×128`, a node is `136 × ceil(16/128) = 136` tiles over 132 SMs — **one wave
plus a 4-tile remainder**, so a second wave's tail runs at ~3 % occupancy, twice
per layer. `N = 34816` is 272 tiles: two full waves, that remainder paid ONCE.

## The saving

`now 5 730.5 µs/step`; `at 80 % HBM: 11.41 GB / (0.80 × 3.35 TB/s) = 4 254 µs`;
**saving 1 476 µs/step = 7.4 % of 19.887 ms**.

Rank **1** of the round-13 decode table, ahead of paged-decode split-K (986 µs)
and GDN state decode (945 µs); that table's kernel-side total is 4 971 µs.

### Round-16 prediction

| cell | now | predicted |
|---|---:|---:|
| `n = 16` decode step, busy | **19.887 ms** | **≈ 18.41 ms** |
| `1024x256` C=16 TPOT | **25.38 ms** | **≈ 23.9 ms** |

TPOT carries the step's 1.955 ms host gap unchanged — a kernel lever, not a
scheduler one — so 1 476 µs lands on the 22.186 ms span as-is.

## Numerics: a bit claim, not a tolerance

The fused weight is the two `[17408, 5120]` E4M3 blocks appended along N; its
`[272, 40]` FP32 scale grid is the two `[136, 40]` grids appended the same way
(`17408 % 128 == 0`, so the seam is a block boundary). Splitting N gives
**independent output columns over the same K with the same scales**: fused
element `(m, j)` is the same dot product, in the same order, as gate `(m, j)`
for `j < 17408` and up `(m, j − 17408)` above. Gate:
`examples/native_fp8_ffn_gateup_fused_microtest.rs` asserts **byte equality** of
both halves and of the SiLU consumer at `M ∈ {5, 8, 16}` over all 16 padded
rows, with four KNOWN_BAD controls (one byte, one row, wrong half, nonfinite).

## Layout and residency

Rows are `[gate | up]` at stride `2·inter`; `ops::silu_mul_strided`
(`kernels/hopper/common/silu_mul_strided.cu`, a Hopper-owned ADDITION — one
dispatcher, one target arming it) reads them. N-concatenation and not a tile
interleave because each half stays an un-fused `Fp8Weight` VIEW — every other
rung of `w8_gemm!` is unchanged — and a row half is 34 816 contiguous bytes,
so coalescing is untouched.

**Residency is net zero.** A second copy would be 178.3 MB × 64 = **11.4 GB**,
straight out of the bs32 KV budget. The loader builds the fused buffer,
re-points `gate_proj`/`up_proj` at views inside it, and `prune_after_load`
releases the two source store tensors; `predicted_residency` prices the pair
as a difference, so the preflight ring fit is unmoved.

**Scope: decode only, `m ∈ 5..=16`.** Below 5 the batch4 GEMV already makes one
weight pass. Above 16 the saving stops — at `M = 4576` these same GEMMs run at
**68.6 % of FP8 PEAK** (§A.4), compute-bound, where a launch buys nothing. The
prefill arm is untouched; widening it is a measurement, not an argument.
