// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle for the TENSOR-CORE 5..=32-row native-FP8 dense-FFN decode tier
//! (`ATLAS_FFN_M16_TC`, #927) — `w8a16_gemm_m16`.
//!
//! Runs the REAL Qwen/Qwen3.8-27B-FP8 FFN shapes — gate/up `[17408, 5120]` and
//! down `[5120, 17408]` — at M in {1, 5, 8, 13, 16, 32} and compares against
//! the scalar `w8a16_gemv` each row's M=1 decode runs.
//!
//! 🔴 THE PASS CONDITION IS A TOLERANCE, NOT BIT-EQUALITY, and that is the
//! point of this file. `w8a16_gemv_batch16` (#927's tier) reduces each output
//! in ONE FP32 accumulator in strict K order and is bit-identical to the
//! scalar; an m16n8k16 MMA reduces 16 K-products in the tensor core's own order
//! first, so this kernel is not. The contract is
//!
//!     max_ulp <= 2   (BF16 ULP, excluding sign flips on |scalar| < 0.05)
//!     rel_rms <= 1e-3
//!
//! The excluded band is where a BF16 ULP is meaningless: a value whose true
//! magnitude sits near zero can land either side of it under a different
//! summation order, and one ULP across zero is a full sign change. Those
//! elements are COUNTED and printed, never silently dropped.
//!
//! It also pins the guards the wrapper promises — no write outside `[M, N]`
//! (rows past M and columns past N keep their sentinel), and the strided
//! sibling leaves the row-pitch gaps intact — and times the tier against the
//! two arms it competes with: `w8a16_gemv_batch16` (the bit-exact tier it would
//! displace) and `w8a16_gemm_n128_m128` (the transposed tile GEMM the FFN used
//! at these widths BEFORE #927, built here by transposing the weight on the
//! host the way the loader's `_t` copy does).
//!
//! Effective GB/s counts the FP8 WEIGHT bytes once per kernel pass: that is the
//! whole budget at decode widths and the number the tier exists to move. The
//! 17..=32 rung makes two passes by design, so its GB/s is against two.
//!
//! TIMING METHOD: `synchronize` + host `Instant` over N reps, the house pattern
//! (`examples/native_fp8_ffn_batch16_microtest.rs`).
//!
//! Run on the H100:
//!     cargo run --release --example native_fp8_ffn_m16_tc_microtest \
//!       --features cuda,gpu-examples
use anyhow::{Result, ensure};
use half::bf16;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

const H: usize = 5120;
const INTER: usize = 17408;
const MAX_M: usize = 32;
const GUARD: usize = 64;
/// Extra columns/elements between rows for the strided-sibling leg.
const A_PAD: usize = 8;
const C_PAD: usize = 64;
const REPS: u32 = 20;
const WARMUP: u32 = 3;
/// BF16 ULP budget, and the magnitude below which a sign flip is not a defect.
const MAX_ULP: i32 = 2;
const SIGN_FLIP_BAND: f64 = 0.05;
const REL_RMS_GATE: f64 = 1e-3;

struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
}

const SHAPES: [Shape; 2] = [
    Shape {
        name: "gate/up",
        n: INTER,
        k: H,
    },
    Shape {
        name: "down",
        n: H,
        k: INTER,
    },
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// BF16 bits -> a monotone integer, so |ord(a) - ord(b)| is the ULP distance
/// and +0/-0 are the same point.
fn ord(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7FFF) as i32)
    } else {
        bits as i32
    }
}

fn f(bits: u16) -> f64 {
    bf16::from_bits(bits).to_f32() as f64
}

#[derive(Default)]
struct Diff {
    max_ulp: i32,
    over_budget: usize,
    sign_flips: usize,
    max_abs: f64,
    rel_rms: f64,
}

/// Compare `m * n` BF16 elements of `actual` against `expected`.
fn compare(actual: &[u8], expected: &[u8]) -> Diff {
    let mut d = Diff::default();
    let (mut se, mut sr) = (0.0_f64, 0.0_f64);
    for (a, b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
        let (ab, bb) = (
            u16::from_le_bytes([a[0], a[1]]),
            u16::from_le_bytes([b[0], b[1]]),
        );
        let (av, bv) = (f(ab), f(bb));
        se += (av - bv) * (av - bv);
        sr += bv * bv;
        d.max_abs = d.max_abs.max((av - bv).abs());
        let ulp = (ord(ab) - ord(bb)).abs();
        if av.signum() != bv.signum() && bv.abs() < SIGN_FLIP_BAND {
            d.sign_flips += 1;
            continue;
        }
        d.max_ulp = d.max_ulp.max(ulp);
        if ulp > MAX_ULP {
            d.over_budget += 1;
        }
    }
    d.rel_rms = if sr > 0.0 {
        (se / sr).sqrt()
    } else {
        se.sqrt()
    };
    d
}

/// The TC route exactly as `DenseFfnLayer::w8a16_m16_tc_proj` runs it: one
/// launch at m<=16, two on contiguous row halves at 17..=32.
#[allow(clippy::too_many_arguments)]
fn tc_route(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    scale: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<usize> {
    let launch = |rows: usize, first: usize| {
        ops::w8a16_gemm_m16(
            gpu,
            kernel,
            input.offset(first * k * 2),
            weight,
            scale,
            out.offset(first * n * 2),
            rows as u32,
            n as u32,
            k as u32,
            0,
        )
    };
    if m <= 16 {
        launch(m, 0)?;
        Ok(1)
    } else {
        let first = m.div_ceil(2);
        launch(first, 0)?;
        launch(m - first, first)?;
        Ok(2)
    }
}

fn batch16_route(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    scale: DevicePtr,
    out: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let launch = |rows: usize, first: usize| {
        ops::w8a16_gemv_batch16(
            gpu,
            kernel,
            input.offset(first * k * 2),
            weight,
            scale,
            out.offset(first * n * 2),
            rows as u32,
            n as u32,
            k as u32,
            0,
        )
    };
    if m <= 16 {
        launch(m, 0)
    } else {
        let first = m.div_ceil(2);
        launch(first, 0)?;
        launch(m - first, first)
    }
}

fn time_ms(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / f64::from(REPS))
}

/// `[N, K]` FP8 -> `[K, N]`, and `[N/128, K/128]` FP32 -> `[K/128, N/128]` —
/// the loader's `_t` copy, built on the host so the pre-#927 arm can be timed.
fn transpose(weights: &[u8], scales: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let mut wt = vec![0_u8; n * k];
    for row in 0..n {
        for col in 0..k {
            wt[col * n + row] = weights[row * k + col];
        }
    }
    let (nb, kb) = (n / 128, k / 128);
    let mut st = vec![0_u8; nb * kb * 4];
    for bn in 0..nb {
        for bk in 0..kb {
            let src = (bn * kb + bk) * 4;
            let dst = (bk * nb + bn) * 4;
            st[dst..dst + 4].copy_from_slice(&scales[src..src + 4]);
        }
    }
    (wt, st)
}

fn main() -> Result<()> {
    let gpu = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let scalar = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let batch16 = gpu.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?;
    let tc = gpu.kernel("w8a16_gemm_m16", "w8a16_gemm_m16")?;
    let tc_strided = gpu.kernel("w8a16_gemm_m16", "w8a16_gemm_m16_strided")?;
    let tile_t = gpu.kernel("w8a16_gemm_t_m128", "w8a16_gemm_t_m128")?;
    let mut rng = Rng(0x927_16_7C_2026);
    let mut failures = 0_usize;

    for shape in &SHAPES {
        let (n, k) = (shape.n, shape.k);
        // E4M3 byte draws skip 0x7F/0xFF (NaN), as the batch4 oracle does —
        // and as `w8a16_gemm_m16.cu`'s DEQUANT note requires: those are the two
        // bytes on which the hardware `cvt` and the `E4M3_LUT` disagree, and a
        // block-scaled checkpoint cannot contain them.
        let weights: Vec<u8> = (0..n * k)
            .map(|_| {
                let x = rng.next();
                ((x % 127) as u8) | (((x >> 7) & 1) as u8 * 128)
            })
            .collect();
        let acts: Vec<u8> = (0..MAX_M * k)
            .flat_map(|_| {
                bf16::from_f32(((rng.next() % 2049) as f32 - 1024.0) / 1024.0)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        let scales: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| (((rng.next() % 16 + 1) as f32) / 1024.0).to_le_bytes())
            .collect();
        // Strided A: the same rows, re-laid at a `k + A_PAD` pitch.
        let a_pitch = k + A_PAD;
        let mut acts_strided = vec![0x3c_u8; MAX_M * a_pitch * 2];
        for row in 0..MAX_M {
            let src = row * k * 2;
            let dst = row * a_pitch * 2;
            acts_strided[dst..dst + k * 2].copy_from_slice(&acts[src..src + k * 2]);
        }
        let (wt, st) = transpose(&weights, &scales, n, k);

        let weight = upload(&gpu, &weights)?;
        let scale = upload(&gpu, &scales)?;
        let input = upload(&gpu, &acts)?;
        let input_s = upload(&gpu, &acts_strided)?;
        let weight_t = upload(&gpu, &wt)?;
        let scale_t = upload(&gpu, &st)?;
        let out_bytes = MAX_M * n * 2;
        let sentinel = vec![0x5a_u8; out_bytes + 2 * GUARD];
        let c_pitch = n + C_PAD;
        let sentinel_s = vec![0x5a_u8; MAX_M * c_pitch * 2 + 2 * GUARD];
        let scalar_base = upload(&gpu, &sentinel)?;
        let tc_base = upload(&gpu, &sentinel)?;
        let b16_base = upload(&gpu, &sentinel)?;
        let tile_base = upload(&gpu, &sentinel)?;
        let strided_base = upload(&gpu, &sentinel_s)?;
        let (scalar_out, tc_out, b16_out, tile_out, strided_out) = (
            scalar_base.offset(GUARD),
            tc_base.offset(GUARD),
            b16_base.offset(GUARD),
            tile_base.offset(GUARD),
            strided_base.offset(GUARD),
        );
        let weight_gb = (n * k) as f64 / 1e9;

        for m in [1_usize, 5, 8, 13, 16, 32] {
            gpu.copy_h2d(&sentinel, scalar_base)?;
            gpu.copy_h2d(&sentinel, tc_base)?;
            gpu.copy_h2d(&sentinel_s, strided_base)?;
            for row in 0..m {
                ops::w8a16_gemv(
                    &gpu,
                    scalar,
                    input.offset(row * k * 2),
                    weight,
                    scale,
                    scalar_out.offset(row * n * 2),
                    n as u32,
                    k as u32,
                    0,
                )?;
            }
            let passes = tc_route(&gpu, tc, input, weight, scale, tc_out, m, n, k)?;
            // Strided leg: one launch at m<=16 (its band), same rows, padded
            // pitches on BOTH sides.
            let strided_m = m.min(16);
            ops::w8a16_gemm_m16_strided(
                &gpu,
                tc_strided,
                input_s,
                weight,
                scale,
                strided_out,
                strided_m as u32,
                n as u32,
                k as u32,
                a_pitch as u32,
                c_pitch as u32,
                0,
            )?;
            gpu.synchronize(0)?;

            let mut baseline = vec![0_u8; sentinel.len()];
            let mut observed = vec![0_u8; sentinel.len()];
            let mut strided = vec![0_u8; sentinel_s.len()];
            gpu.copy_d2h(scalar_base, &mut baseline)?;
            gpu.copy_d2h(tc_base, &mut observed)?;
            gpu.copy_d2h(strided_base, &mut strided)?;
            let bytes = m * n * 2;
            let d = compare(
                &observed[GUARD..GUARD + bytes],
                &baseline[GUARD..GUARD + bytes],
            );

            // GUARDS. Nothing outside [M, N]: the leading/trailing sentinel and
            // every row past M must be untouched.
            let guards_intact = observed[..GUARD] == sentinel[..GUARD]
                && observed[GUARD + bytes..] == sentinel[GUARD + bytes..];
            // Strided: each row's [n, c_pitch) gap and every row past
            // `strided_m` must still be sentinel, and the used extent must
            // match the scalar within the same budget.
            let mut gaps_intact = strided[..GUARD] == sentinel_s[..GUARD];
            let mut strided_ulp = 0_i32;
            for row in 0..MAX_M {
                let base = GUARD + row * c_pitch * 2;
                let gap = &strided[base + n * 2..base + c_pitch * 2];
                gaps_intact &= gap.iter().all(|b| *b == 0x5a);
                if row < strided_m {
                    let sd = compare(
                        &strided[base..base + n * 2],
                        &baseline[GUARD + row * n * 2..GUARD + (row + 1) * n * 2],
                    );
                    strided_ulp = strided_ulp.max(sd.max_ulp);
                    if sd.over_budget != 0 {
                        gaps_intact = false;
                    }
                } else {
                    gaps_intact &= strided[base..base + n * 2].iter().all(|b| *b == 0x5a);
                }
            }

            let tc_ms = time_ms(&gpu, || {
                tc_route(&gpu, tc, input, weight, scale, tc_out, m, n, k)?;
                Ok(())
            })?;
            let b16_ms = time_ms(&gpu, || {
                batch16_route(&gpu, batch16, input, weight, scale, b16_out, m, n, k)
            })?;
            let tile_ms = time_ms(&gpu, || {
                ops::w8a16_gemm_n128_m128(
                    &gpu, tile_t, input, weight_t, scale_t, tile_out, m as u32, n as u32, k as u32,
                    0,
                )
            })?;
            let gbs = |ms: f64| weight_gb * passes as f64 / (ms / 1e3);
            let ok =
                d.over_budget == 0 && d.rel_rms <= REL_RMS_GATE && guards_intact && gaps_intact;
            println!(
                "{name:<8} M={m:<3} N={n} K={k} passes={passes} \
                 max_ulp={ulp} over_budget={ob} sign_flips={sf} max_abs={ma:.9} \
                 rel_rms={rr:.3e} strided_max_ulp={su} guards={g} gaps={gp} | \
                 m16_tc {tc_ms:.3}ms ({tcg:.1} GB/s) \
                 vs batch16 {b16_ms:.3}ms ({b16g:.1} GB/s) = {sp1:.2}x \
                 vs t_m128 {tile_ms:.3}ms ({tileg:.1} GB/s) = {sp2:.2}x  {verdict}",
                name = shape.name,
                ulp = d.max_ulp,
                ob = d.over_budget,
                sf = d.sign_flips,
                ma = d.max_abs,
                rr = d.rel_rms,
                su = strided_ulp,
                g = if guards_intact { "ok" } else { "CLOBBERED" },
                gp = if gaps_intact { "ok" } else { "CLOBBERED" },
                tcg = gbs(tc_ms),
                b16g = gbs(b16_ms),
                tileg = weight_gb / (tile_ms / 1e3),
                sp1 = b16_ms / tc_ms,
                sp2 = tile_ms / tc_ms,
                verdict = if ok { "PASS" } else { "FAIL" },
            );
            if !ok {
                failures += 1;
            }
        }

        // Oracle self-check, once: a two-ULP mutation of the baseline MUST be
        // refused by the same comparison the loop runs, so a green report
        // cannot mean "the comparison was vacuous".
        let mut good = vec![0_u8; sentinel.len()];
        gpu.copy_d2h(scalar_base, &mut good)?;
        let mut bad = good.clone();
        // Find an element well outside the sign-flip band and push it 3 ULP.
        let idx = (GUARD..GUARD + n * 2)
            .step_by(2)
            .find(|i| f(u16::from_le_bytes([good[*i], good[*i + 1]])).abs() > 1.0)
            .expect("baseline has a value above 1.0");
        let bits = u16::from_le_bytes([good[idx], good[idx + 1]]);
        bad[idx..idx + 2].copy_from_slice(&(bits.wrapping_add(3)).to_le_bytes());
        let caught =
            compare(&bad[GUARD..GUARD + n * 2], &good[GUARD..GUARD + n * 2]).over_budget > 0;
        println!(
            "KNOWN_BAD {} three-ULP mutation: refused={caught}",
            shape.name
        );
        ensure!(caught, "comparison oracle admitted a three-ULP mutation");
    }

    ensure!(
        failures == 0,
        "{failures} shape/M cases exceeded the {MAX_ULP}-ULP / {REL_RMS_GATE} rel_rms budget \
         or broke a guard"
    );
    println!(
        "ALL PASS: real Qwen3.8-27B FFN shapes, w8a16_gemm_m16 at M 1/5/8/13/16/32 within \
         {MAX_ULP} BF16 ULP and {REL_RMS_GATE} rel_rms of the scalar w8a16_gemv, \
         strided gaps and [M,N] bounds intact"
    );
    Ok(())
}
