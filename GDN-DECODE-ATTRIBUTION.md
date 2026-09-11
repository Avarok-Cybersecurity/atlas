# Where the GDN decode recurrence goes (#927/#928, H100)

Phase-1 attribution. **Headline: the two GDN decode kernels are bound by
different things.** At C=1 `gated_delta_rule_decode_f32` moves 6.29 MB in
17.8 us — 353 GB/s on a 3350 GB/s part — because its grid is 48 CTAs on a
132-SM device. At n=16 `..._strided` is already at 53% of the
compulsory-traffic roofline; its headroom is memory-level parallelism.

## Anchors (given; nsys, 1xH100, Qwen3.8-27B-FP8, 48 SSM layers, round 10, 2026-09-11)

| step | kernel | launches | total | per layer | share |
|---|---|---:|---:|---:|---:|
| n=16 (21.800 ms) | `gated_delta_rule_decode_f32_strided` | 48 | 2.740 ms | 57.1 us | 12.6% |
| C=1 (18.500 ms) | `gated_delta_rule_decode_f32` | 48 | 0.850 ms | 17.8 us | 4.6% |

Rest of the mixer, same trace: n=16 conv+l2norm 0.280 ms, ba-gates 0.230, norm
0.120; C=1 ba-gates 0.210, conv+l2norm 0.150. **Shape, derived:** 3.16 MB of
FP32 state per layer per sequence at `hd = 128` gives `3.16 MB/(128*128*4 B) =
48` value heads, so below uses `nv = 48, k_dim = v_dim = 128`.

## Bytes, FLOPs, achieved rate, bound

Both parents launch `grid = (num_v_heads, batch_size)`, `block = (128,1,1)`:
one CTA per (row, head), one thread per state COLUMN walking its `k_dim` rows.
Per launch, per row:

* state: 3.146 MB read + 3.146 MB written = **6.29 MB compulsory**; 9.44 MB

  ISSUED, because `h[j][i]` is read once for `hk_dot` and again for the update.
  That second read is CTA-local and microseconds later, so L2 serves most of it.

* q, k, v, the output row, gate and beta are 98 KB — 1.6% of the state.
* FP ops per column: 256 (`hk_dot`) + 384 (update) + 256 (`q_dot`), +256 for the

  strided kernel's Frobenius accumulation.

| | C=1 | n=16 |
|---|---:|---:|
| compulsory bytes/launch (issued) | 6.29 MB (9.44) | 100.7 MB (151.0) |
| FLOPs/launch | 5.50 M | 113 M |
| measured us/launch | 17.8 | 57.1 |
| **compulsory GB/s** (issued) | **353** = 10.5% of 3350 (530) | **1763** = 53% (2645) |
| TFLOP/s | 0.31 | 1.98 (3% of FP32 peak) |
| CTAs / warps | 48 / 192 | 768 / 3072 |
| warps per SM (132 SMs) | **1.45 of 64**, 84 SMs idle | 23.3 of 64 |
| **bound** | **launch geometry** | memory-level parallelism |

At C=1 the arithmetic is 0.5% of FP32 peak and the bandwidth 10% of it while 84
SMs hold no CTA: the kernel is not bandwidth-bound, it is *absent* from two
thirds of the GPU. At n=16 the grid is large enough and the lever is in-flight
loads per thread.
## What this change does, and what it refuses to do

`kernels/hopper/common/gdn_decode_hopper.cu` — bit-exact twins, same entry
arguments, selected by kernel PRESENCE (the file exists only under
`kernels/hopper`, so handles are 0 elsewhere). A/B kill switch
`ATLAS_NO_GDN_HOPPER=1`.

1. **C=1 grid** `(ceil(v_dim/cols), num_v_heads, batch_size)`; `cols` narrows to
   32 only when `num_v_heads * rows < sm_count`, turning 48 CTAs into 192.
   Columns are independent: it re-partitions work, it does not re-associate.
2. **Unroll 4 -> 16** on both `j += 4` loops: 64 independent 4-byte loads in
   flight per thread instead of 16, and unrolling replicates the body rather
   than reordering the accumulation.
3. **Refused: state retention.** The tree already measured both ways to remove
   the re-read: register retention -11.6% e2e (ptxas sm_90a: 255 registers,
   88 B spill), SMEM staging +0.5%/-0.5% on dgx2 at C=128.
4. **Refused: tiling the strided kernel's columns.** Its parent compiles with
   `SSM_STATE_NORM_ENABLED` — the define sits BETWEEN the two parents in
   `kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu`, which is why only one
   carries it — and clamps on a Frobenius norm reduced across the WHOLE head.
   Tiling needs a grid-wide barrier; dropping the clamp is a numerics change.
5. **Not attempted: fusing conv + l2norm + ba-gates in.** Worth 0.63 ms of the
   21.8 ms step; `gated_delta_rule_decode_f32_conv_norm` is the shape for it.

## Status of the claims

**No H100 measurement of these twins exists.** The only A/B is dgx2 (GB10, 48
SMs), production flags (`--fmad=false -DTQ_PLUS_SIGNS`), state re-uploaded per
rep, 3 scored passes of 30, single harness — engagement evidence and the unroll
sweep, not a Hopper result:

| n | 1 | 2 | 4 | 8 | 16 | unroll sweep, strided, n=16 |
|---|---:|---:|---:|---:|---:|---|
| strided twin | 2.67x | 1.26x | 0.99x | 1.27x | 1.56x | 4: 1.16x · 8: 1.19x |
| contiguous twin | 1.92x | 1.27x | 1.08x | 1.09x | 1.34x | **16: 1.56x** · 32: 1.09x |

n=4 is a flat spot on both (1.02x/1.01x on a second run). Every leg is
BIT-IDENTICAL — state and output, n in {1,4,16}, tile 32/64/128, on a state
scaled to trip SSM_STATE_MAX_NORM and one that is not. Two cautions. (a) An earlier sweep that omitted `--fmad=false` read 1.8x-3.5x;
the flag is in `kernels/gb10/common/KERNEL.toml` and the model's own, and every
number above carries it. (b) GB10 has exactly 48 SMs — one per value head — so
the C=1 underfill this change targets does not exist there. GB10 is the
CONTRAPOSITIVE, not a weak positive: where the grid already fills the device,
narrowing reads 1.42x where the parent's width reads 1.92x — why the chooser
narrows only on underfill.
Owed on H100, in order: the `ATLAS_NO_GDN_HOPPER` kill-switch A/B at C=1 and
n=16 under the fingerprint rules; nsys to confirm the new us/layer; then the
retention re-probe a 50 MB L2 against a 50 MB n=16 working set makes worth
re-asking.