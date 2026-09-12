# FP8 activation quantizer — 1×H100, Qwen3.8-27B-FP8 (#928, #927)

`per_token_group_quant_fp8` turns every W8A8 projection's BF16 activation into
FP8 E4M3 bytes plus one FP32 scale per (token, 128-K-group) — pure bandwidth
work, running at a fifth of this part's bandwidth.

## Measurement — round 13, cells T1N/V at `3c0379030`, `--cuda-graph-trace=node`

| window | launches | total µs | share |
|---|---|---|---|
| 4593-tok prefill (busy union 459.812 ms) | 544 | **47 659.5** | **10.36 %** |
| 1168-tok forward (busy 220.588 ms) | 256 | **12 270.2** | **5.56 %** |
| n=16 decode step (busy 19.887 ms) | 256 nodes | **470.1** | 2.36 % |

Per launch at `M = 4576`, bytes `M·K·(2 rd + 1 wr) + M·(K/128)·4` (compulsory):

| K | launches | µs/launch | GB/s | % of 3350 GB/s |
|---|---|---|---|---|
| 5120 | 128 | 112.18 | **633** | **18.9 %** |
| 17408 | 64 | 376.72 | **641** | **19.1 %** |
| 6144 | 64 | 134.10 | **636** | **19.0 %** |

`rms_norm_residual` in the same trace: 2 644 GB/s (78.9 %), 3 298 GB/s (98.4 %
at `M=1168`). 80 % is demonstrated on this hardware, not aspirational — but it
is not what this twin reached. **Measured: 63.7–68.4 %** (§ Round-16
measurement), and every prediction below is restated from that figure.

## Root cause, and the twin

`kernels/gb10/common/per_token_group_quant_fp8.cu:39`, launched
`.grid([m, k/128, 1]).block([128,1,1])`: **one 128-thread CTA per 128-element
group, one bf16 element per thread** — grids `(4576,136)`, `(4576,40)`,
`(4576,48)` = `M × K/128` are the receipt. The loads coalesce; too few are in
flight. A CTA issues ONE 256-byte load, then stalls its full latency; at 19
registers the limit is 16 CTAs/SM, so an SM holds **4 KB** of reads —
132 × 4 KB / ~600 ns ≈ 0.9 TB/s, bracketing the measured 0.63. MLP-bound, not
bandwidth-bound. It also reads A twice and round-trips the scale through smem
behind two `__syncthreads`.

`kernels/hopper/common/fp8_act_quant_hopper.cu`: 16 threads per group, one
`uint4` (8 bf16) each, **8 groups per 128-thread CTA** → 2 KB per CTA load;
values stay in registers (A read once); group max via a 16-lane
`__shfl_xor_sync` butterfly, so no smem and no barrier. Grid
`(M, ceil(K/128 / 8), 1)`, and the kernel derives its span from `gridDim.y`, so
any Y in `1..=K/128` is correct (`fp8_act_quant_tests.rs` proves the partition).
At 40 registers: 12 CTAs/SM → **24 KB** in flight, 6× the parent.

### ptxas — CUDA 13.0 `compiler.36424714_0`, `-arch=sm_90a --fmad=false`

```
per_token_group_quant_fp8_hopper  0 stack frame, 0 spill stores, 0 spill loads
                                  Used 40 registers, used 0 barriers
per_token_group_quant_fp8 (gb10)  0 stack frame, 0 spill stores, 0 spill loads
                                  Used 19 registers, used 1 barriers, 20 bytes smem
```

PTX, twin/parent: `ld.global.nc.v4` 1/0, `st.global.v2` 1/0, `shfl.sync.bfly`
4/0 — and UNCHANGED `div.rn.f32` 9/2, `cvt.rn.satfinite.e4m3x2.f32` 8/1: the
same two instructions eight times per thread instead of once. Hopper PTX gate
**197/197** for sm_90a, strict.

## Bit-identity, and its gate

Same `amax / 448.0f`, `1e-12f` floor, per-element `div.rn.f32` by that scale
(not a reciprocal multiply), saturating clamp,
`__nv_cvt_float_to_fp8(…, __NV_SATFINITE, __NV_E4M3)`. Only the reduction TREE
differs; `fmaxf` is exact, associative and commutative, and both kernels seed
with `0.0f`, which makes the NaN case agree too (PTX `max.f32` returns the
non-NaN). `examples/native_fp8_act_quant_hopper_microtest.rs` runs both on one
device buffer at `M ∈ {16,17,25,1168,4576}` × `K ∈ {5120,6144,17408}` and
requires byte equality of FP8 bytes AND FP32 scales, with guard bands (a span
bug writes past the row, not inside it). KNOWN_BAD = a host E4M3 encoder with
round-to-nearest deleted; it must differ.

An ADDITION under `[kernels] overrides`, not an override of the gb10 stem: both
kernels must be in the Hopper image at once, because that gate runs on device.
gb10/b200/strix are byte-for-byte unaffected. Selected through
`ops::Fp8ActQuant`, which returns entry point and grid together so one kernel's
handle cannot reach the other's grid, by kernel presence AND `[defaults]
fp8_act_quant_hopper` AND the CTA floor. Presence ALONE is what round 16
shipped, and § Round-16 measurement is what that cost. The lever is a SPEED
kill switch rather than a numeric A/B — the two kernels are bit-identical — and
the control for the GB/s claim is still a build without the file. B200 is not
linked: the code is arch-neutral, but there is no B200 receipt, and its floor
would be `2 × 148` CTAs — a threshold nobody has measured.

## Round-16 measurement — `native_fp8_act_quant_hopper_microtest`, 1×H100

Round 14's prediction below was computed from an 80 %-of-HBM target. Round 16
ran the kernel. **All 15 arms bit-identical** (FP8 bytes AND FP32 scales), the
KNOWN_BAD control refused at every one, and the twin reached **63.7–68.4 %**,
not 80 %:

| M | K | parent | twin | speed-up |
|---:|---:|---|---|---:|
| 4576 | 17408 | 377.75 µs / 639.2 GB/s (19.1 %) | **105.33 µs / 2 292.5 (68.4 %)** | **3.59×** |
| 4576 | 6144 | 135.57 µs / 628.6 (18.8 %) | **39.38 µs / 2 164.3 (64.6 %)** | **3.44×** |
| 4576 | 5120 | 113.47 µs / 625.9 (18.7 %) | **33.29 µs / 2 133.4 (63.7 %)** | **3.41×** |
| 1168 | 5120 | 30.76 µs / 589.2 (17.6 %) | **9.12 µs / 1 986.9 (59.3 %)** | **3.37×** |
| 1168 | 6144 | 36.43 µs / 597.2 | **10.31 µs / 2 109.3 (63.0 %)** | **3.53×** |
| 1168 | 17408 | 98.97 µs / 622.7 | **29.96 µs / 2 057.0 (61.4 %)** | **3.30×** |

The parent's standalone 625.9 / 628.6 / 639.2 GB/s reproduces the round-13
in-engine nsys figure (633 / 636 / 641) to **1.1 %**, so the anchor above is
confirmed by two independent methods on two binaries.

### The six regression arms, and the floor they bought

At decode widths the twin is SLOWER, at six of the fifteen arms:

| M | K | parent | twin | speed-up |
|---:|---:|---:|---:|---:|
| 16 | 5120 | 3.22 µs | 3.84 µs | **0.84×** |
| 17 | 5120 | 3.69 µs | 3.89 µs | **0.95×** |
| 25 | 5120 | 3.20 µs | 4.21 µs | **0.76×** |
| 16 | 6144 | 3.18 µs | 3.88 µs | **0.82×** |
| 17 | 6144 | 3.09 µs | 3.87 µs | **0.80×** |
| 25 | 6144 | 3.31 µs | 3.83 µs | **0.87×** |

K=17408 is the exception — 1.02×, 1.02×, 1.08× at M = 16, 17, 25 — and that is
the mechanism stated plainly. 8 groups per CTA is 8× fewer CTAs; at these M the
parent's `M × K/128` grid is already under one wave on 132 SMs, so dividing it
by 8 removes parallelism that was doing work. K=17408 has 136 groups, so even
after the division the twin's grid is 17 wide and M=16 still fills the machine.

Round 16 shipped the twin selected by kernel PRESENCE, with no lever and no
floor, so **every decode-width W8A8 call took the slower arm**. It now takes a
launch only when its own grid clears `2 × sm_count` CTAs — 264 on an H100 —
which is `M ≥ 53` at K=5120, `M ≥ 44` at K=6144 and `M ≥ 16` at K=17408, and
reproduces the sign of all fifteen arms. Rule:
`layers/ops/fp8_act_quant_floor.rs`; lever `[defaults] fp8_act_quant_hopper`
(hopper `true`, gb10/b200 `false` and inert), kill switch
`ATLAS_FP8_ACT_QUANT_HOPPER=0`.

## Round-14 prediction, restated from the measurement

The original table was `measured × (1 − 19/80)`. At the measured 63.7–68.4 %
the saving on the prefill launches is the SUBSTITUTION, not a ratio — the
microtest priced every arm, so no target figure is needed:

| cell | today | round-14 (80 % target) | **restated (measured)** |
|---|---|---|---|
| prefill 4593 tok, busy union | 459.8 ms | ≈ 423.5 ms (−36.3) | **≈ 426.3 ms (−33.5)** |
| forward at M=1168, busy | 220.6 ms | ≈ 211.2 ms (−9.4) | **≈ 212.1 ms (−8.5)** |
| `4096x512` C=1 TTFT p50 | 491.5 ms | ≈ 455 ms | **≈ 458.0 ms** |
| `4096x512` C=16 TTFT | 4 145.5 ms | ≈ 3 835 ms | **≈ 3 859 ms** |
| `1024x256` C=1 TTFT p50 | 162.4 ms | ≈ 153 ms | **≈ 153.9 ms** |
| decode step, n=16 | 470.1 µs | ≈ 220 µs (−250) | **470.1 µs (0)** |

The two busy rows are arithmetic on the round-13 trace, not measurements. The
prefill trace prices the 256 `M=4576` launches at
`128×112.18 + 64×376.72 + 64×134.10` = **47 051 µs** of the kernel's 47 656 µs,
and substituting the twin's measured per-launch times gives
`128×33.29 + 64×105.33 + 64×39.38` = **13 522 µs** → −33.5 ms. The forward's
256 launches at `M=1168` cost 12 270.2 µs traced (the microtest's own parent
sums to 12 602.9 µs on the same split, agreeing to 2.7 %), against
`128×9.12 + 64×10.31 + 64×29.96` = **3 744.6 µs** → −8.5 ms. The three TTFT
rows carry the matching busy-row saving through at round 14's own ratios. A
serve-level A16-vs-A15 TTFT test would confirm or refute all of it and **has
not run** — the round-16 instance stopped before stage 3.

**The decode row is now 0, not −250 µs.** Every projection in an n=16 step runs
at `M = 16..=25` and `K ∈ {5120, 6144, 17408}`; the floor puts the two narrow
K's on the parent, whose per-launch time is unchanged by definition, and leaves
K=17408 on the twin at 1.02–1.08×. So the honest prediction for the decode step
is **no change**, with a small upside at the wide-K launches. Round 14's −250 µs
was derived from the 80 % figure on the assumption that the twin is never
slower; the microtest says it is, at exactly the widths decode uses.

`fp8_act_scale_to_kmajor` (320 nodes, 406 µs/step) is deliberately NOT folded
in: two consumers read two layouts — in-tree `fp8_gemm_t_blockscaled` wants
row-major `[M, K/128]`, cuBLASLt wants `[K/128, ceil16(M)]` with pad rows zeroed
— and `M_pad` is not an argument the quantizer has. Separate change, own
selector, not a byte-identical rewrite of this kernel.
