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
at `M=1168`). 80 % is demonstrated on this hardware, not aspirational.

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
gb10/b200/strix are byte-for-byte unaffected. Selected by kernel PRESENCE
through `ops::Fp8ActQuant`, which returns entry point and grid together so one
kernel's handle cannot reach the other's grid. No `[defaults]` row — there is no
numeric A/B to arm, and the control for the GB/s claim is a build without the
file. B200 is not linked: arch-neutral code, no B200 receipt
(`gdn_decode_hopper.cu` precedent).

## Round-14 prediction

`measured × (1 − 19/80)`: the 80 %-of-HBM target removes 76 % of the kernel.

| cell | today | predicted |
|---|---|---|
| prefill 4593 tok, busy union | 459.8 ms | **≈ 423.5 ms** (−36.3) |
| `4096x512` C=1 TTFT p50 | 491.5 ms | **≈ 455 ms** |
| `4096x512` C=16 TTFT | 4 145.5 ms | **≈ 3 835 ms** |
| forward at M=1168, busy | 220.6 ms | **≈ 211.2 ms** (−9.4) |
| `1024x256` C=1 TTFT p50 | 162.4 ms | **≈ 153 ms** |
| decode step, n=16 | 470.1 µs | **≈ 220 µs** (−250) |

The decode row is a CEILING: at `M=16` these are 256 graph nodes of ~1.8 µs
against a per-node floor of about the same, and the twin launches 8× FEWER CTAs
at a shape already too small to fill the machine. Read −250 µs/step as the
bandwidth bound; anything past −100 µs is the result.

`fp8_act_scale_to_kmajor` (320 nodes, 406 µs/step) is deliberately NOT folded
in: two consumers read two layouts — in-tree `fp8_gemm_t_blockscaled` wants
row-major `[M, K/128]`, cuBLASLt wants `[K/128, ceil16(M)]` with pad rows zeroed
— and `M_pad` is not an argument the quantizer has. Separate change, own
selector, not a byte-identical rewrite of this kernel.
