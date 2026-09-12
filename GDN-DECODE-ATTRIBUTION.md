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
## Round 17 — the n>=4 strided twin: read the state once, not twice

The round-12 answer above closed one question and opened another. The
column-tiled twin lost because **there was nothing left to fill**: at n=16 the
parent's grid is already 768 CTAs on 132 SMs. That verdict says nothing about
the OTHER axis, and round 13's decode trace priced it.

### The cost

nsys `--cuda-graph-trace=node`, 1xH100 80GB HBM3, `Qwen/Qwen3.8-27B-FP8` @
`3717cb05e`, round 13 cell V, median n=16 decode step **19.887 ms busy**, 1 619
graph nodes, one graph launch per step (`h100-r13-attribution.md` SS C.1–C.4):

| | value |
|---|---|
| `gated_delta_rule_decode_f32_strided*` | **48 nodes, 2 748.9 µs = 13.82 % of the step** |
| per launch | **57.27 µs** |
| live f32 state per launch, `n·nv·kd·vd·4 B` | 50.33 MB |
| COMPULSORY traffic, `2·n·nv·vd·kd·4 B` | 100.66 MB → **1 758 GB/s = 52.5 % of HBM** |
| ISSUED traffic (the parent reads the state TWICE) | 150.99 MB → **2 637 GB/s = 78.7 % of HBM** |

It is the third largest item in the step, behind only the two cuBLASLt FFN
arms. **The 52.5 % is not slack — it is a second read.** The parent walks the
state once to form `hk_dot = (Hᵀk)` and again to apply
`H ← g·H + k ⊗ v_new` while forming `q_dot = (H_newᵀq)`, because `v_new`
depends on the whole of the first pass.

### What the kernel does

`kernels/hopper/common/gdn_decode_strided_hopper.cu`, entry
`gated_delta_rule_decode_f32_strided_hopper_smem`. Each (sequence, head) tile
is 128 × 128 f32 = 64 KB, and its rows are split three ways:

| rows | where they live between the passes | global reads |
|---|---|---|
| `[0, 72)` | staged in SHARED MEMORY on pass 1, `float4` (LDG.128, 512 B/warp) | 1 |
| `[72, 96)` | RETAINED IN REGISTERS across both passes | 1 |
| `[96, 128)` | re-read from global on pass 2, as the parent does for all 128 | 2 |

**2.25 reads-equivalents + 1 write against the parent's 2 + 1 — 25 % less state
traffic.** The 32 re-read rows are 16 KB/tile = 12.6 MB across the launch,
re-read within microseconds and comfortably inside 50 MB of L2, so they are the
ones L2 plausibly serves; the 96 the kernel keeps are the ones it plausibly
does not, because the whole live state is 50.33 MB against an H100's 50 MB of
L2 and all 768 CTAs are resident at once.

### The ordering argument — why this is the ONLY shape that can be bit-identical

The parent's per-element reduction over `kd` is a **serial f32 chain owned by
one thread**. For state column `i`:

```
acc = 0;
for (j = 0; j < 128; j += 4)
    acc = acc + (((h[j]*k[j] + h[j+1]*k[j+1]) + h[j+2]*k[j+2]) + h[j+3]*k[j+3]);
```

f32 addition is not associative. Any re-partition of `j` across threads — a
warp-shuffle butterfly, a split into partial sums, a reduction tree of any
shape — re-brackets that sum and changes the answer. So the only bit-identical
partition is the parent's own: **one thread per state column, walking every
`j` itself**. That is why this kernel keeps `grid = (nv, n)`, `block = (128,1,1)`
and `tid` = column; why it does *not* tile columns the way
`gdn_decode_hopper.cu` does; and why there is no shuffle anywhere in the two
dot products. The only cross-thread reduction present is the
`SSM_STATE_MAX_NORM` clamp, which the parent also does across the whole head
and which is reproduced shuffle for shuffle.

Given that mapping, bit-identity is a **storage** argument and nothing else:

* every float pass 2 consumes is the exact float pass 1 loaded from `H` at the
  same index — an f32 round trip through shared memory or a register is the
  identity, and each thread touches only its own `tid` column of `smem_h`;
* `j` is visited in ascending order in both passes, in groups of four, with the
  expression text copied character for character from the parent — including
  the group shape and the sequential `hk_dot +=` / `q_dot +=` / `norm_acc +=`
  chains. The three row segments are three spellings of one loop body;
  splitting a loop at a constant bound does not re-bracket the accumulator,
  because the accumulator is carried across the split;
* the write set and the write ADDRESSES are the parent's, element for element;
* the file compiles under `--fmad=false` (`kernels/gb10/common/KERNEL.toml`,
  and the model's own KERNEL.toml sets it too), so no FMA-contraction freedom
  is left for a schedule change to exercise.

`ops::gdn_strided_smem_row_home` and `ops::gdn_strided_smem_elem` mirror the
row map and the address map on the host, and
`ssm_gdn_strided_hopper_tests.rs` asserts the three segments partition
`[0,128)` exactly once, in ascending order, on multiples of four.

### The ptxas receipt — sm_90a, CUDA 13.0.88, `--fmad=false`, `--Werror all-warnings`

```
ptxas info : Function properties for gated_delta_rule_decode_f32_strided_hopper_smem
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 80 registers, used 1 barriers, 37904 bytes smem
```

On a GH100 SM (65 536 registers, 233 472 B of shared memory) that is
`min(65536/(80·128), 233472/37904)` = `min(6, 6)` = **6 resident CTAs/SM = 24
warps/SM**, and 6 × 132 = 792 slots for the n=16 grid's 768 CTAs — **one wave**,
which is what the parent gets too (40 registers, 1 040 B of smem; its own
ceiling is 12 CTAs/SM but the grid caps it at 5.82). The split is the shipped
point of this sweep, all rows 0-spill except the last:

| smem rows / reg rows | registers | smem B | CTAs/SM | traffic units |
|---|---|---|---|---|
| *(parent)* | 40 | 1 040 | 5.82 *(grid-limited)* | 3.000 |
| 72 / 16 | 66 | 37 904 | 6 | 2.313 |
| **72 / 24 — SHIPPED** | **80** | **37 904** | **6** | **2.250** |
| 72 / 32 | 96 | 37 904 | 5 | 2.188 |
| 80 / 24 | 72 | 42 000 | 5 | 2.188 |
| 80 / 48 *(no re-read)* | 128 | 42 000 | 4 | 2.000 |
| 64 / 64 *(no re-read)* | **255, 68 B spill** | 33 808 | 2 | 2.000 |

The last row reproduces, from the other direction, the loss already on file:
full-width register retention spills (`gated_delta_rule_decode_f32_strided_norm_half`,
−11.6 % e2e), turning the retained columns into LOCAL memory — the traffic the
lever was removing. All-shared is the other end: 64 KB/CTA is past the 48 KB
static limit and caps residency at 3 CTAs/SM. **A twin that halves the traffic
and also halves the residency has made a trade, not a fix — that is exactly how
the n=1 twin became a 0.83×.** 72/24/32 is the lowest-traffic point that keeps
the parent's wave structure.

### The lever and the width guard

`kernels/hopper/HARDWARE.toml` `[defaults] gdn_decode_strided_hopper = true`;
`gb10` and `b200` declare it `false` (and do not carry the source — it is not
symlinked into b200). A **new row** rather than a third value on
`gdn_decode_hopper`, because the two rows are opposite claims about different
kernels — column re-partition for the n=1 underfill (OFF, measured loss) versus
state traffic for the n≥4 batched arm (ON) — and one row governing both is what
`gdn_prefill_tc`'s family lever had to grow an `ATLAS_NO_*_REMNANTS` escape
hatch for. `ATLAS_GDN_DECODE_STRIDED_HOPPER=0` is the one-variable A/B;
`ATLAS_NO_GDN_HOPPER=1` still outranks it, deliberately shared with the other
row so an operator disarming "the Hopper GDN decode twins" disarms all of them.

The boot line carries it (`target defaults (hopper): … gdn_decode_hopper=off
gdn_decode_strided_hopper=on(target) …`) and the dispatch prints one route line
per process naming the entry it launched:

```
GDN state decode: gated_delta_rule_decode_f32_strided_hopper_smem ([defaults]
gdn_decode_strided_hopper; f32 state read once for 96 of 128 rows, 72 staged in
smem + 24 retained in registers) grid=[48,16] block=128 smem=37904B ctas=768
sm_count=132 ctas_per_sm<=6
```

**The width guard** (`ops::gdn_decode_strided_smem_accept`) takes the twin only
when `n ≥ 4` **and** `nv·n ≥ sm_count`. One CTA per (sequence, head) means the
grid IS `nv·n`, so the row count is the occupancy; at this model's `nv = 48`,
n=4 is 192 CTAs on 132 SMs — the first width at which every SM gets one. At
n=1 the grid is 48 CTAs and the problem is underfill, not traffic, which is the
other twin's job and which round 12 says it does not manage either. So **the
n=1 path stays on the parent**, and the C=1 ladder cells are untouched by this
change.

### The gate

`native_gdn_decode_hopper_microtest` runs the new kernel as a **third arm** at
strided × `n ∈ {1,4,16}` × `hs ∈ {0.05, 20}` — 6 legs, all compared against the
gb10 parent with `state_diff == 0` and `out_diff == 0`, a hard count and not a
tolerance, with guard bytes around every written buffer. The kernel is exercised
at n=1 as well, where the LAUNCHER declines it, because a gate that only ran it
where the launcher runs it could not tell "declined" from "broken". A
**KNOWN_BAD control** runs first: the parent against itself with one f32 of the
second state perturbed by one ULP, required to report a non-zero `state_diff` —
without it, a `diff` that returned `(0, 0.0)` unconditionally would make every
assertion vacuous and the suite would print `ALL LEGS BIT-IDENTICAL` just as
loudly.

### ⚠️ The prediction is a BAND, and here is why

Round 13's nsys is the receipt for the COST. The SAVING depends on how much of
the parent's second read L2 already serves, and **no trace in this campaign
measures that**. GB10 has a receipt for the same idea and it is NEGATIVE —
`gated_delta_rule_decode_f32_strided_norm_smem` stages the same way behind
`ATLAS_GDN_SMEM_STAGE` and measured +0.5 %/−0.5 % at C=128 on dgx2, inside the
boot band, "because the re-read is CTA-local and is served by L2". That is a
48-SM part with a different L2 and a different working set; it is why this is a
separate Hopper file and not a change to the shared parent, and it is also why
the row below is stated as a range. All of it is arithmetic on the measured
57.27 µs/launch and the 25 % traffic cut:

Let `η` be the fraction of the parent's second read that L2 already serves.
The parent's HBM traffic is then `3 − η` units of the 50.33 MB state (read,
re-read, write); this kernel's is `2.25 − 0.25η` (it still re-reads its own 32
rows). Holding the achieved GB/s equal — both kernels are one thread per
column walking one tile — the twin's launch is `(2.25 − 0.25η)/(3 − η)` of the
measured 57.27 µs:

| `η` — parent's second read served by L2 | parent HBM | twin/launch | saving/launch | saving/step (48 layers) |
|---|---|---|---|---|
| 0 — entirely missing | 150.99 MB | 42.95 µs | **−14.3 µs** | **−687 µs (−3.5 %)** |
| 0.5 — half served | 125.83 MB | 48.68 µs | −8.6 µs | −412 µs (−2.1 %) |
| 1 — entirely served | 100.66 MB | 57.27 µs | 0 | instruction-side only |

The round-13 lever table's **“945 µs/step at an 80 %-of-HBM target”** is the
CEILING of that band (it prices the kernel against a roofline computed on
compulsory bytes), not a prediction. Round 17's nsys picks the point.

### Round-17 cells to fill

Predicted, at the top of the band (second read entirely missing) and with the
n=1 path unchanged by construction:

| cell | now (round 13/16) | predicted | mechanism |
|---|---|---|---|
| n=16 decode step, GPU busy | 19.887 ms | **≈ 19.20 ms** | −687 µs on the GDN decode launch |
| `gated_delta_rule_decode_f32_strided*` | 2 748.9 µs (13.82 %) | **≈ 2 062 µs (≈ 10.7 %)** | 3.00 → 2.25 traffic units |
| `1024x256` C=16 TPOT | 25.9 ms | **≈ 25.2 ms** | one GDN decode launch per SSM layer per step |
| `4096x512` C=16 TPOT | 31.6 ms | **≈ 30.9 ms** | same, and context-independent — the state is fixed-size |
| `1024x256` / `4096x512` C=1 | unchanged | **unchanged** | the width guard keeps n=1 on the parent |
| microtest, 6 new legs | — | `state_diff=0 out_diff=0` | bit-identity is a contract, not a measurement |

The brief this work was scoped from predicted **−0.9 ms/step** and TPOT
25.9 → 25.0 / 31.6 → 30.7. That is the 945 µs ceiling of the round-13 lever
table — the whole distance from 52.5 % to an 80 %-of-HBM target — and it is
reachable only if the parent's second read misses entirely AND the twin's own
re-read is free. The −687 µs above is the same `η = 0` case priced on the
traffic this kernel actually issues, which is the number to hold it to.

The A/B is one variable: `ATLAS_GDN_DECODE_STRIDED_HOPPER=0` against the
default, same binary, same seed. If the null arrives instead, the row goes to
`false` with the GB10 note beside it and the kernel stays compiled, exactly as
`gdn_decode_hopper` did.
