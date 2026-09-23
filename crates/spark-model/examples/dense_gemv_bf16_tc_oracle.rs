// SPDX-License-Identifier: AGPL-3.0-only
//! ORACLE (and microbench) for the MTP drafter's tensor-core BF16 GEMV
//! (`dense_gemv_bf16_tc.cu`, routed by `ops::dense_gemv_tc`).
//!
//! The drafter tensor-core path sends every BF16 projection at M = 2..=32 to the
//! tensor-core entries. That is sound only if, on every real 27B drafter
//! shape and every row count, the routed result is as accurate as the
//! CUDA-core kernel it replaces. This decides it:
//!
//!   1. ARMED: `try_dense_gemv_tc` (the PRODUCTION launcher) launches for every
//!      (M, N, K); otherwise this would silently measure nothing. It must
//!      DECLINE M=1 (`MIN_M`): the C=1 propose keeps `dense_gemv_bf16`.
//!   2. vs CPU f64 (64 sampled output columns + first/last, every row): the
//!      routed max error must not exceed max(1.25 x the CUDA-core kernel's own
//!      error, one BF16 half-ulp of the output range).
//!   3. vs the CUDA-core kernel (`dense_gemv_bf16_batchm`, bit-identical to
//!      per-row `dense_gemv_bf16`), EVERY element: |tc - ref| <= 1 BF16 ulp of
//!      the larger magnitude + 2^-10 x output absmax (FP32 summation order).
//!   4. Nothing outside the M x N output is written: `out_stride = N + 8` and
//!      the 8 pad columns of every row plus the whole row M are sentinels.
//!   5. DETERMINISM: a second routed launch is bit-identical to the first.
//!   6. LEVER MOVED: some element differs bitwise from the reference.
//!
//! Also reported (not gated): BATCH INVARIANCE — whether token row t is
//! bit-identical at every M (the MMA reduction per column should not depend on
//! how many other columns are live).
//!
//! KNOWN-BAD CONTROLS (the gate must FAIL each, or it proves nothing):
//!   A. weight row 0 scaled x4 (BF16 exponent +2): check 3 must trip.
//!   B. routed launch with M-1 rows into an M-row buffer: check 3/4 must trip.
//!
//! `--bench`: per M, one full draft position (the 8 projections, 849 MB of
//! cold BF16 weights) looped for ~4 s on (a) the CUDA-core dispatch the
//! drafter runs without this path and (b) the routed tensor-core entry.
//! Prints `PHASE` lines with unix-ms edges so an nvidia-smi power trace can
//! be joined to each window.
//!
//! Exit: 0 pass, 1 any leg or control misbehaved, 2 not armed / kernels absent.
//!
//! Run (GB10):
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example dense_gemv_bf16_tc_oracle [-- --bench]

use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use spark_model::layers::ops;
use spark_model::layers::ops::dense_gemv_tc;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// One 27B drafter draft position, forward order: (label, N, K).
const SHAPES: [(&str, u32, u32); 8] = [
    ("fc      ", 5120, 10240),
    ("q+gate  ", 12288, 5120),
    ("k       ", 1024, 5120),
    ("v       ", 1024, 5120),
    ("o       ", 5120, 6144),
    ("ffn gate", 17408, 5120),
    ("ffn up  ", 17408, 5120),
    ("ffn down", 5120, 17408),
];
/// Odd N: a partial last weight tile (rows past N load zeros, never stored).
const ODD: (&str, u32, u32) = ("odd N   ", 1001, 5120);
const MS: [u32; 11] = [2, 3, 4, 5, 8, 9, 12, 16, 17, 24, 32];
const PAD: u32 = 8;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// BF16 bits of a uniform value in [-amp, amp).
    fn bf(&mut self, amp: f32) -> u16 {
        let v = ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp;
        (v.to_bits() >> 16) as u16
    }
}

fn bf(b: u16) -> f64 {
    f32::from_bits((b as u32) << 16) as f64
}
fn bf_ulp(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else {
        2f64.powi(v.abs().log2().floor() as i32 - 7)
    }
}
fn to_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn to_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}
fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

struct Case<'a> {
    g: &'a dyn GpuBackend,
    a: DevicePtr,
    w: DenseWeight,
    c: DevicePtr,
    n: u32,
    k: u32,
}

impl Case<'_> {
    fn stride(&self) -> u32 {
        self.n + PAD
    }
    /// Output buffer of `m + 1` rows x `stride`, pre-filled with 0xFFFF.
    fn run(&self, m: u32, f: impl Fn(DevicePtr) -> Result<()>) -> Result<Vec<u16>> {
        let bytes = (m as usize + 1) * self.stride() as usize * 2;
        self.g.copy_h2d(&vec![0xFFu8; bytes], self.c)?;
        f(self.c)?;
        self.g.synchronize(0)?;
        let mut out = vec![0u8; bytes];
        self.g.copy_d2h(self.c, &mut out)?;
        Ok(to_u16(&out))
    }
    /// The CUDA-core reference, 16 rows per launch (per-row bit-identical to
    /// `dense_gemv_bf16`).
    fn reference(&self, bm: KernelHandle, m: u32) -> Result<Vec<u16>> {
        let s = self.stride();
        self.run(m, |c| {
            let mut r0 = 0u32;
            while r0 < m {
                let rows = (m - r0).min(16);
                let off = r0 as usize;
                ops::dense_gemv_batchm(
                    self.g,
                    bm,
                    self.a.offset(off * self.k as usize * 2),
                    &self.w,
                    c.offset(off * s as usize * 2),
                    rows,
                    self.n,
                    self.k,
                    s,
                    0,
                )?;
                r0 += rows;
            }
            Ok(())
        })
    }
    /// The PRODUCTION launcher.
    fn routed(&self, m_launch: u32, m_buf: u32) -> Result<Vec<u16>> {
        self.run(m_buf, |c| {
            let s = self.stride();
            let ran = dense_gemv_tc::try_dense_gemv_tc(
                self.g, self.a, &self.w, c, m_launch, self.n, self.k, s, 0,
            )?;
            anyhow::ensure!(ran, "not routed at m={m_launch} n={} k={}", self.n, self.k);
            Ok(())
        })
    }
}

/// Checks 3 + 4 over an (m+1) x stride buffer. Returns violating elements.
fn compare(tc: &[u16], rf: &[u16], m: usize, n: usize, stride: usize) -> usize {
    let at = |r: usize, j: usize| r * stride + j;
    let absmax = (0..m)
        .flat_map(|r| (0..n).map(move |j| (r, j)))
        .map(|(r, j)| bf(rf[at(r, j)]).abs())
        .fold(0.0, f64::max);
    let mut bad = 0;
    for r in 0..m {
        for j in 0..n {
            let (x, y) = (bf(tc[at(r, j)]), bf(rf[at(r, j)]));
            let tol = bf_ulp(x.abs().max(y.abs())) + absmax / 1024.0;
            if !x.is_finite() || (x - y).abs() > tol {
                bad += 1;
            }
        }
        bad += (n..stride).filter(|&j| tc[at(r, j)] != 0xFFFF).count();
    }
    bad + tc[m * stride..].iter().filter(|&&b| b != 0xFFFF).count()
}

fn main() -> Result<()> {
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let Ok(bm) = g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm") else {
        eprintln!("dense_gemv_bf16_batchm not in this target's module set");
        std::process::exit(2);
    };
    if !dense_gemv_tc::mtp_tc_enabled() {
        eprintln!("NOT ARMED: unset AVAROK_NO_MTP_TC");
        std::process::exit(2);
    }
    if std::env::args().any(|a| a == "--bench") {
        return bench(g, bm);
    }

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let kmax = 17408usize;
    // Activations: RMS-normed scale (~1) with a sparse set of outlier channels.
    let a_host: Vec<u16> = (0..32 * kmax)
        .map(|i| rng.bf(if i % 211 == 0 { 24.0 } else { 1.7 }))
        .collect();
    let a = g.alloc(a_host.len() * 2)?;
    g.copy_h2d(&to_bytes(&a_host), a)?;
    let c = g.alloc(33 * (17408 + PAD as usize) * 2)?;

    let (mut failures, mut controls_ok, mut lever_moved) = (0usize, true, false);
    let (mut worst_abs, mut worst_rel) = (0f64, 0f64);
    let mut batch_invariant = true;
    for (label, n, k) in SHAPES.into_iter().chain([ODD]) {
        let (nu, ku) = (n as usize, k as usize);
        let s = (n + PAD) as usize;
        let mut wb: Vec<u16> = (0..nu * ku).map(|_| rng.bf(0.05)).collect();
        let wp = g.alloc(wb.len() * 2)?;
        g.copy_h2d(&to_bytes(&wb), wp)?;
        let case = Case {
            g,
            a,
            w: DenseWeight { weight: wp },
            c,
            n,
            k,
        };
        let cols: Vec<usize> = (0..64)
            .map(|r| (r * 2654435761usize) % nu)
            .chain([0, nu - 1])
            .collect();
        let widest = case.routed(32, 32)?;
        if dense_gemv_tc::try_dense_gemv_tc(g, a, &case.w, c, 1, n, k, n + PAD, 0)? {
            println!("{label} M=1 was ROUTED: MIN_M is not enforced  FAIL");
            failures += 1;
        }
        for m in MS {
            let mu = m as usize;
            let rf = case.reference(bm, m)?;
            let tc = case.routed(m, m)?;
            let tc2 = case.routed(m, m)?;
            let (mut e_ref, mut e_tc, mut range) = (0f64, 0f64, 0f64);
            for r in 0..mu {
                for &j in &cols {
                    let acc: f64 = (0..ku)
                        .map(|kk| bf(a_host[r * ku + kk]) * bf(wb[j * ku + kk]))
                        .sum();
                    range = range.max(acc.abs());
                    e_ref = e_ref.max((bf(rf[r * s + j]) - acc).abs());
                    e_tc = e_tc.max((bf(tc[r * s + j]) - acc).abs());
                }
            }
            let cpu_ok = e_tc <= (1.25 * e_ref).max(bf_ulp(range) / 2.0);
            let bad = compare(&tc, &rf, mu, nu, s);
            let deterministic = tc == tc2;
            let (mut differ, mut max_abs) = (0usize, 0f64);
            for r in 0..mu {
                for j in 0..nu {
                    let (x, y) = (tc[r * s + j], rf[r * s + j]);
                    differ += usize::from(x != y);
                    max_abs = max_abs.max((bf(x) - bf(y)).abs());
                }
            }
            let inv = (0..mu).all(|r| tc[r * s..r * s + nu] == widest[r * s..r * s + nu]);
            batch_invariant &= inv;
            lever_moved |= differ > 0;
            worst_abs = worst_abs.max(max_abs);
            worst_rel = worst_rel.max(max_abs / range.max(f64::MIN_POSITIVE));
            let ok = cpu_ok && bad == 0 && deterministic;
            failures += usize::from(!ok);
            println!(
                "{label} N={n:5} K={k:5} M={m:2}  cpu_err ref={e_ref:.3e} tc={e_tc:.3e} (range {range:.2})  \
                 max|tc-ref|={max_abs:.3e} out_of_tol={bad} bits_differ={differ} det={deterministic} \
                 batch_inv={inv}  {}",
                if ok { "PASS" } else { "FAIL" }
            );
        }

        // Control A: weight row 0 x4 (exponent +2 on every non-zero element).
        let m = 4u32;
        let rf = case.reference(bm, m)?;
        let orig: Vec<u16> = wb[..ku].to_vec();
        for b in &mut wb[..ku] {
            if *b & 0x7FFF != 0 {
                *b += 0x0100;
            }
        }
        g.copy_h2d(&to_bytes(&wb[..ku]), wp)?;
        let tc_bad = case.routed(m, m)?;
        wb[..ku].copy_from_slice(&orig);
        g.copy_h2d(&to_bytes(&wb[..ku]), wp)?;
        let ctl_a = compare(&tc_bad, &rf, m as usize, nu, s);
        // Control B: M-1 rows launched into an M-row buffer.
        let tc_short = case.routed(m - 1, m)?;
        let ctl_b = compare(&tc_short, &rf, m as usize, nu, s);
        let this_ok = ctl_a > 0 && ctl_b > 0;
        controls_ok &= this_ok;
        println!(
            "{label} CONTROLS  weight-row-x4 trips={ctl_a}  short-M trips={ctl_b}  {}",
            if this_ok {
                "CONTROLS-FIRE"
            } else {
                "CONTROL-SILENT (gate is blind)"
            }
        );
        g.free(wp)?;
    }
    let pass = failures == 0 && controls_ok && lever_moved;
    println!(
        "\nworst max|tc-ref| = {worst_abs:.3e} ({:.3e} of the output range); batch-invariant: {batch_invariant}",
        worst_rel
    );
    println!(
        "ORACLE {} (legs failed: {failures}, controls fire: {controls_ok}, lever moved: {lever_moved})",
        if pass { "PASS" } else { "FAIL" }
    );
    std::process::exit(if pass { 0 } else { 1 });
}

/// One draft position's 8 projections per iteration, per variant, ~4 s each.
fn bench(g: &dyn GpuBackend, bm: KernelHandle) -> Result<()> {
    let pipe = g.kernel("gemm", "dense_gemm_bf16_pipelined")?;
    let mut rng = Rng(7);
    let mut ws = Vec::new();
    for (_, n, k) in SHAPES {
        let host: Vec<u16> = (0..(n * k) as usize).map(|_| rng.bf(0.05)).collect();
        let p = g.alloc(host.len() * 2)?;
        g.copy_h2d(&to_bytes(&host), p)?;
        ws.push((DenseWeight { weight: p }, n, k));
    }
    let a_host: Vec<u16> = (0..32 * 17408).map(|_| rng.bf(1.7)).collect();
    let a = g.alloc(a_host.len() * 2)?;
    g.copy_h2d(&to_bytes(&a_host), a)?;
    let c = g.alloc(32 * 17408 * 2)?;

    // The CUDA-core dispatch the drafter runs today (row_dispatch.rs):
    // 2..=8 batchm, above 8 the pipelined mma.sync GEMM (M=1 never routes).
    let current = |m: u32| -> Result<()> {
        for (w, n, k) in &ws {
            match m {
                ..=8 => ops::dense_gemv_batchm(g, bm, a, w, c, m, *n, *k, *n, 0)?,
                _ => ops::dense_gemm_bf16_pipelined(g, pipe, a, w, c, m, *n, *k, 0)?,
            }
        }
        Ok(())
    };
    let routed = |m: u32| -> Result<()> {
        for (w, n, k) in &ws {
            anyhow::ensure!(dense_gemv_tc::try_dense_gemv_tc(
                g, a, w, c, m, *n, *k, *n, 0
            )?);
        }
        Ok(())
    };
    for m in [2u32, 4, 8, 12, 16, 24, 32] {
        let variants: [(&str, &dyn Fn(u32) -> Result<()>); 2] =
            [("current", &current), ("tc", &routed)];
        for (name, f) in variants {
            for _ in 0..3 {
                f(m)?;
            }
            g.synchronize(0)?;
            let (t0, wall) = (unix_ms(), Instant::now());
            let mut iters = 0u32;
            while wall.elapsed().as_secs_f64() < 4.0 {
                for _ in 0..10 {
                    f(m)?;
                }
                g.synchronize(0)?;
                iters += 10;
            }
            let us = wall.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "PHASE {name} M={m} t0={t0} t1={} us_per_position={us:.1} GB/s={:.1}",
                unix_ms(),
                849.35e3 / us
            );
        }
    }
    Ok(())
}
