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
arguments. Selection was by kernel PRESENCE (the file exists only under
`kernels/hopper`, so handles are 0 elsewhere) until round 12 measured them;
it is now the declared lever `[defaults] gdn_decode_hopper`, **false on every
target**, with `ATLAS_GDN_DECODE_HOPPER=1` as the positive and the original
`ATLAS_NO_GDN_HOPPER=1` kill switch still outranking both. See "The H100
answer" below.

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
re-asking. **The first two were run — round 12, below.**

## The H100 answer: bit-identical, and a small net LOSS

1xH100 80GB HBM3, `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a`, Atlas `cc5a21e46`,
driver 580.173.02 / CUDA 13.0.88, 2026-09-11 (`h100-round12-report.md`). Three
independent measurements, taken in this order, agreeing to within a hair:

**1. `native_gdn_decode_hopper_microtest`** — `nv=48 kd=128 vd=128
sm_count=132`, both state scales, n in {1,4,16}, contiguous and strided. The
numerics gate is a hard `unequal` count, not a tolerance:

```
ALL LEGS BIT-IDENTICAL          state_diff 0  out_diff 0  max_abs 0.000e0  (12/12)
```

| leg | parent | twin | ratio |
|---|---|---|---|
| **contiguous n=1** | 11.30 us / 557 GB/s | 13.62 us / 462 GB/s | **0.83x** |
| strided n=1 | 30.06 us | 29.82 us | 1.01x |
| contiguous n=4 / n=16 | 12.53 / 80.23 us | 12.61 / 81.93 us | 0.99x / 0.98x |
| strided n=4 / n=16 | 30.37 / 168.72 us | 30.08 / 170.67 us | 1.01x / 0.99x |

(The `hs=20` half of the matrix reads the same to within 0.01x.)

**2. nsys, launch count for launch count**, against round 10's trace of the
parents on the same box and the same step shape:

| step | parent | twin | delta |
|---|---|---|---|
| C=1, 48 launches | `…decode_f32` 854.2 us | `…decode_f32_hopper` **912.7 us** | **+6.8%** |
| n=16, 48 launches | `…_strided` 2744.8 us | `…_strided_hopper` **2749.9 us** | +0.19% (null) |

On a 14.913 ms C=1 step that +58.5 us is +0.39%.

**3. The serve A/B**, cell E (twins on) against cell F
(`ATLAS_NO_GDN_HOPPER=1`), 1193-in/256-out, 3 reps, temp 0 / seed 42, 0 errors:

| | E (on) | F (off) | F vs E |
|---|---|---|---|
| tok/s agg C=1 | 66.08 | **66.35** | **+0.41%** |
| TPOT C=1 | 14.14 ms | **14.08 ms** | **-0.43%** |
| tok/s agg C=16 | 429.04 | 428.18 | -0.20% |
| TPOT C=16 | 28.50 ms | 28.50 ms | 0.00% |

The C=1 delta is larger than either cell's rep spread (0.02% and 0.04%) and
points the same way on both metrics, so it is a sign and not noise — and the
nsys +0.39% predicted it to within a hair. It is also small.

**Why.** The column-tiled grid has nothing to fill on a 132-SM H100: the parent
already saturates the device at n>=4, and at n=1 the extra CTAs cost more in
launch and reduction than they recover. This contradicts nothing above — the
dgx2 table is 48 SMs, where the underfill this change targets does not exist,
and the H100 underfill argument in "Where the time goes" is about *CTAs per SM*,
which the twin does fix while losing more elsewhere.

**What changed, and what did not.** The default moved and nothing else:
`kernels/hopper/HARDWARE.toml` declares `gdn_decode_hopper = false`, so a
hopper serve with an empty environment runs the gb10 parents. The kernel stays
in that target's `[kernels] overrides` and keeps being compiled — the receipt
is per-target and a smaller-SM Hopper part may well take the other side.
`ATLAS_GDN_DECODE_HOPPER=1` is how the next one gets measured;
`ATLAS_NO_GDN_HOPPER=1`, the spelling cell F was run with, still forces off and
outranks the positive. Numerics were never the question: 12/12 bit-identical
means the choice is free in both directions.