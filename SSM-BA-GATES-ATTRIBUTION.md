# Where `dense_gemm_ba_gates_prefill` goes (#928, H100)

**Headline: it is instruction-issue bound, not bandwidth bound**, so round 13's
"2.6% of HBM → 60% of HBM" target does not apply. The Hopper twin
`dense_gemm_ba_gates_prefill_hopper` (`kernels/hopper/common/ssm_ba_gates_hopper.cu`,
`[kernels] overrides`, an `adds`) is **bit-identical** and removes ~1.8x of the
launch's instructions. That ratio, not a roofline, is its worth.

## Anchors (given; nsys, 1xH100 80GB HBM3, Qwen3.8-27B-FP8, round 13 cell T1 @ `3c0379030`, `h100-r13-attribution.md` §A.3/§A.4/§B.1/§C.2)

| shape | launches | total | per launch | share |
|---|---:|---:|---:|---:|
| 4593-token prefill (M=4576 + M=17) | 96 | **26 881.8 µs** | — | **5.85%** of 459.812 ms |
| …its M=4576 chunk / its M=17 chunk | 48 / 48 | 26 641.9 / 239.9 | **555.04** / 5.00 | |
| 1168-token forward (cell V) | 48 | **6 896.3** | 143.7 | **3.13%** of 220.588 ms |
| decode step, n=16 (cell V) | 48 | 227.5 | 4.74 | 1.14% of 19.887 ms |

## The mechanism, corrected

The parent's grid is `(ceil(N/4), M, 1)`, block `(256,1,1)`; its 256 threads are
**four 64-lane groups, one per BA output, and every group sweeps all of K**. At
`N = 2·nv = 96`, `K = 5120` the activation row is therefore read **96 times per
token — once per BA output**, not 24. (§A.4 counts CTAs and states 24x; its
issued-traffic row is 4x low. The ranking's ORDER is unaffected.)

Still not a bandwidth story: compulsory traffic is 88 GB/s = **2.6% of HBM** and
the re-reads are CTA-local, served from L1/L2. What the 96x buys is 96x the
bf16→f32 conversion of A, and with `--fmad=false` every MAC is a separate
`mul.f32` + `add.f32`, so the floor is ISSUE. SASS + ptxas, sm_90a, CUDA
13.0.88, `-O3 --fmad=false` (loops found by backward branch):

| | inner loop | inst/MAC | LDG.128/MAC | regs | spills | smem |
|---|---:|---:|---:|---:|---:|---:|
| parent (unrolled ×4) | 175 / 32 MAC | **5.47** | 0.250 | 32 | 0 | 32 B |
| `…_hopper` (BAH_GROUPS=8) | 268 / 64 MAC | **4.19** | 0.141 | **64** | **0** | 256 B |

At M=4576 the parent's loop bodies alone are `4576 · 6144 thr · 458 inst / 32 =
402.3 M` warp-instructions; an H100 issues `132 · 4 · 1.755 GHz = 926.6 G`/s, so
they are **434 µs of a 555.04 µs launch (78%)**. No bandwidth headroom exists.

## The twin

One CTA per token; a thread keeps its lane and sweeps the output groups in a
register tile of 8, so one fetch+convert of the row serves 8 outputs: **12 row
reads per token instead of 96**. Everything else is the parent's, in order — the
lane-strided `kv` sweep, the 16/8/4/2/1 butterfly, the `warp_even + warp_odd`
cross-warp sum, the transforms; `__bfloat162float` is exact, so hoisting A's
conversion cannot move a bit. `BAH_GROUPS` is measured: 4 → 47 regs / 4.69
inst/MAC; **8 → 64 / 4.19**; 12 → spills; 16 → 80 regs / 4.05 but 3 CTAs/SM, not
4. Occupancy halves against the parent's 8 CTAs/SM; the eight independent
accumulator chains per thread pay for that.

## Round-15 measurement — MEASURED, 1xH100 80GB HBM3, tip `8a6f50b61`

The prediction below landed. Cell B0 (`ATLAS_SSM_BA_GATES_HOPPER=0`) against
cell A15 (`auto`), same binary, one variable:

| rung | B0 (twin off) | **A15 (twin on)** | Δ | predicted |
|---|---:|---:|---:|---|
| `4096x512` C=1 TTFT | 498.2 ms | **489.7 ms** | **−8.5 ms** | **−7.9** (−6.0…−9.7) |
| `1024x256` C=1 TTFT | 162.7 ms | **160.5 ms** | **−2.2 ms** | **−2.1** |
| `4096x512` C=1 TPOT | 13.66 ms | 13.66 ms | **0.00%** | 0 (guard) |
| `1024x256` C=1 TPOT | 13.40 ms | 13.40 ms | **0.00%** | 0 (guard) |
| C=1 tok/s, both shapes | 68.47 / 71.49 | 68.54 / 71.53 | +0.10 / +0.06% | 0 (guard) |

**Both TTFT predictions land inside their stated range and TPOT is identical to
the hundredth of a millisecond on both shapes** — the token-count guard keeping
decode on the parent, exactly as designed.

**The kernel, by two independent methods, agreeing to 0.35%.** nsys (cell A15N,
prefill forward) prices the twin at **372.86 µs** per `M=4576` launch, grid
`(4576,1,1)`, 48 launches = 3.85% of prefill busy; the microtest, on the same
binary, at **371.56 µs** against the parent's 545.20 µs there and **555 µs** in
round 13's live trace — **0.67×**. Over 48 launches the M=4576 chunk saves
**8.74 ms**, against the B0-vs-A15 serve TTFT delta of **8.5 ms**: 3% apart.
The predicted 355–430 µs (midpoint 390) landed at the fast end.

**The guard is confirmed on the live engine.** nsys shows the M=17 tail chunk on
the parent at `grid=(24,17,1)`, 4.93 µs, and decode on `dense_gemv_ba_gates` at
`(24,1,1)`, 4.61 µs, 48×/step — round-13 costs, unchanged. The microtest prices
the twin BELOW the floor at a 3× loss (M=17: 6.17 → 20.04 µs), which is what
`twin_floor = 264 tokens` exists for.

**Byte-identical, as claimed.** `out_diff=0` and `max_abs=0.000e0` at all four M
in the microtest, the one-ulp KNOWN_BAD control refused at every one, A-row
reads 96 → 12 exactly; and at serve level A15 and B0 are md5-identical on all
seven coherency exchanges.

**One observability defect, now closed.** The route line's once-flag was shared
between its two branches, so on five of six round-15 serve cells the log read
`the Hopper twin is NOT running at M=27` for the life of the process — the
smoke test's 27-token request tripping the flag — while nsys showed the twin
running 48× per prefill. `ba_gates_log` now keeps one flag per branch, so a
serve log carries both the refusal and the positive pick, each naming the M it
was reached at.

## Round-14 prediction — PREDICTION, arithmetic only, not a measurement

Twin issue floor at M=4576: `4576 · 256 thr · 8865 inst / 32 / 926.6 G = 351 µs`
against the parent's **measured** 555.04. Allowing for the halved occupancy,
predict **355–430 µs, midpoint 390 (0.70x)**. M=17 and n=16 stay on the parent
(token-count guard), so only the M=4576 launches move.

| cell | now | predicted | Δ |
|---|---:|---:|---:|
| `4096x512` C=1 prefill busy | 459.8 ms | ≈ 451.9 | **−7.9 ms** (−6.0…−9.7) |
| `4096x512` C=1 TTFT | 491.5 ms | **≈ 483.6** | −7.9 |
| `1024x256` C=1 TTFT | 162.4 ms | **≈ 160.3** | −2.1 |
| decode n=16 step | 19.887 ms | 19.887 | 0 (guard keeps the parent) |

⚠️ **SUPERSEDES `h100-r13-attribution.md` §E lever 3's −25.7 / −6.6 ms.** Those
are `measured × (2.6% / 60% of HBM)` — a bandwidth target this kernel is not
bound by (that doc's caveat 6 labels such figures roofline headroom, not a
promise). Mechanism: the compulsory-byte roofline is ~23x away, the instruction
roofline ~1.6x away, and the second binds. Still worth taking — it cannot change
a bit — but it is worth ~8 ms, not ~26. Option (a) there (fold N=96 into the
`in_proj_qkvz` cuBLASLt GEMM, N 16384→16480) is the only way to remove the
remaining B-side convert+mul+add — it moves the work to tensor cores — but it
quantises BA to block-scaled FP8, a model-quality change to the GDN gates. Not
prototyped here; it must not ship as a default without its own accuracy receipt
against this bf16 path.

## Gates

`native_ssm_ba_gates_hopper_microtest` (`cuda,gpu-examples`): parent vs twin at
`M ∈ {17, 25, 1168, 4576}` — **byte equality** of `gate`/`beta`, guard bands, a
one-ulp KNOWN_BAD control that must trip, µs + GB/s + row-reads per arm. Host
`ssm_ba_gates_hopper_tests` grades the lane/`kv`/warp-slot/output mapping and
every guard string; system gate is coherency 4/4 + determinism 8/8.
