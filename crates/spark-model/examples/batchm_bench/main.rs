// SPDX-License-Identifier: AGPL-3.0-only

//! W4A16 batched-GEMV M-sweep bench + correctness gate for the chain-verify
//! widening (K=5..8 → M=5..8 rows) — decides whether `w4a16_gemv_batch8`
//! removes the M>4 projection cliff (the documented 19.3 ms lm_head tile-GEMM
//! fallback) before any dispatch gate is widened.
//!
//! Compares, at M ∈ {1,3,4,5,6,8} on the real Qwen3.6-27B shapes
//! (qkv/o 5120×5120, ffn gate/up 17408×5120, ffn down 5120×17408,
//! lm_head 248320×5120):
//!   w4a16_gemv_batch4   (M<=4, shipping verify kernel)
//!   w4a16_gemv_batch8   (M<=8, NEW — acc[8]/smem[8][8])
//!   w4a16_gemv_batch16  (M<=16, exists but unused at M=5..8)
//!   w4a16_gemm          (M64-tile GEMM — the current lm_head M>4 fallback)
//!   w4a16_gemm_t        (N128/M64 small-M tile GEMM — what dense
//!                        forward_prefill dispatches at M<=64, i.e. the FFN
//!                        arm the missing n==4 branch falls onto; timed with
//!                        the same buffers — content is timing-irrelevant)
//!
//! COLD WEIGHTS: iterations cycle across enough weight copies to exceed L2,
//! so every launch streams weights from DRAM like a real verify step.
//!
//! CORRECTNESS GATE (references stated):
//!   1. batch8 vs batch4 at M ∈ {1,3,4}: BIT-EXACT (batch4 is the shipping,
//!      production-proven kernel; same template `w4a16_gemv_batchm_impl`).
//!   2. batch8 vs batch16 at M ∈ {5,6,8}: BIT-EXACT (independent
//!      instantiation of the same template — per-row FP32 order identical).
//!   3. batch8 at M=8 vs CPU f64 dequant reference (first 256 N rows):
//!      tolerance gate (catches a template-wide bug that 1/2 would miss;
//!      the tile GEMM epilogue has a different FP order, so bit-exactness
//!      vs the GEMM is not expected and not used).
//!
//!   cargo run -p spark-model --release --example batchm_bench --features cuda,gpu-examples
//!
//! Env: ATLAS_PEAK_GBPS (default 273 — GB10 LPDDR5x) for the %-of-peak column.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const WARMUP: usize = 10;
const ITERS: usize = 50;
/// Weight bytes to cycle through so back-to-back launches miss L2/SLC.
const COLD_CYCLE_BYTES: usize = 256 << 20;
const M_SWEEP: &[u32] = &[1, 3, 4, 5, 6, 8];
const M_WIDE: &[u32] = &[1, 8, 9, 12, 18, 24, 36]; // GEMM-only: batched verify = n seqs × 9 rows
const M_MAX: usize = 36;
/// N rows checked against the CPU reference (full N for GPU-GPU bit checks).
const CPU_CHECK_ROWS: usize = 256;

/// Qwen3.6-27B shapes: (label, N, K). C[M,N] = A[M,K] · W[N,K].
const SHAPES: &[(&str, u32, u32)] = &[
    ("qkv/o    N=5120   K=5120 ", 5120, 5120),
    ("ffn_up   N=17408  K=5120 ", 17408, 5120),
    ("ffn_down N=5120   K=17408", 5120, 17408),
    // Qwen3.8-27B GDN in_proj qkvz — 48 launches/step in the DFlash2 M=8
    // verify (nsys node-trace 2026-08-19; ran ~148 GB/s on shipping batch8).
    ("gdn_qkvz N=16384  K=5120 ", 16384, 5120),
    ("lm_head  N=248320 K=5120 ", 248320, 5120),
];

mod refmath;
use refmath::*;

/// Launch a batchm GEMV (batch4/8/16): grid ceil(N/4), block 256, args
/// (A, B_packed, B_scale, scale2, C, M, N, K) — mirrors ops::w4a16_gemv_batchm.
#[allow(clippy::too_many_arguments)]
fn launch_batchm(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bs)
        .arg_f32(1.0)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

/// Launch a tile GEMM: base w4a16_gemm grid (N/64, M/64) or transposed
/// w4a16_gemm_t grid (N/128, M/64), block 128 — mirrors ops::w4a16_gemm /
/// ops::w4a16_gemm_n128.
#[allow(clippy::too_many_arguments)]
fn launch_gemm(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    n_tile: u32,
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    // `Some(ldb)` for `w4a16_gemm_t` — its compiled signature grew a 9th
    // `ldb` param (lm_head-716 fix). Launching it with 8 args makes
    // cuLaunchKernel read one-past-the-end of the param array: the
    // documented host-SIGSEGV class (see gemm_dense.rs on the bf16 _v2
    // sibling). This was THE bench segfault of 2026-08-19.
    ldb: Option<u32>,
) -> Result<()> {
    let mut l = KernelLaunch::new(g, kh)
        .grid([div_ceil(n, n_tile), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bs)
        .arg_f32(1.0)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k);
    if let Some(ldb) = ldb {
        l = l.arg_u32(ldb);
    }
    l.launch(0)
}

#[derive(Clone, Copy)]
enum Kind {
    Batchm {
        max_m: u32,
    },
    /// Base w4a16_gemm: grid (N/64, M/64).
    Gemm,
    /// Transposed small-M w4a16_gemm_t: grid (N/128, M/64). Timing only —
    /// takes the same (non-transposed) buffers; GEMM time is not
    /// data-dependent and buffer SIZES are exact.
    GemmT,
}

fn read_c(g: &dyn GpuBackend, c: DevicePtr, elems: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; elems * 2];
    g.copy_d2h(c, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn correctness_gate(
    g: &dyn GpuBackend,
    kernels: &[(&str, KernelHandle, Kind)],
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
    a_host: &[u16],
    b_host: &[u8],
    bs_host: &[u8],
) -> Result<()> {
    let handle = |name: &str| -> Result<KernelHandle> {
        kernels
            .iter()
            .find(|(kn, _, _)| *kn == name)
            .map(|&(_, h, _)| h)
            .ok_or_else(|| anyhow::anyhow!("kernel {name} missing"))
    };
    let (b4, b8, b16) = (handle("batch4")?, handle("batch8")?, handle("batch16")?);
    let (n_us, k_us) = (n as usize, k as usize);
    let zero_c = |g: &dyn GpuBackend| g.memset(c, 0, M_MAX * n_us * 2);

    // Gate 1+2: batch8 bit-exact vs batch4 (M<=4) / batch16 (M=5..8).
    for &m in M_SWEEP {
        let (ref_kh, ref_name) = if m <= 4 {
            (b4, "batch4")
        } else {
            (b16, "batch16")
        };
        zero_c(g)?;
        launch_batchm(g, ref_kh, a, b, bs, c, m, n, k)?;
        g.synchronize(0)?;
        let reference = read_c(g, c, m as usize * n_us)?;
        zero_c(g)?;
        launch_batchm(g, b8, a, b, bs, c, m, n, k)?;
        g.synchronize(0)?;
        let got = read_c(g, c, m as usize * n_us)?;
        if reference != got {
            let bad = reference.iter().zip(&got).filter(|(x, y)| x != y).count();
            bail!(
                "GATE FAIL: batch8 != {ref_name} at M={m} ({bad}/{} elems differ)",
                got.len()
            );
        }
    }
    eprintln!("  gate 1/2 PASS: batch8 BIT-EXACT vs batch4 @M<=4 and batch16 @M=5..8");

    // Gate 3: batch8 @M=8 vs CPU f64 dequant reference (first CPU_CHECK_ROWS).
    zero_c(g)?;
    launch_batchm(g, b8, a, b, bs, c, 8, n, k)?;
    g.synchronize(0)?;
    let got = read_c(g, c, 8 * n_us)?;
    let rows = CPU_CHECK_ROWS.min(n_us);
    let groups = k_us / 16;
    let mut worst = 0.0f64;
    for row in 0..rows {
        let mut wf = vec![0.0f64; k_us];
        for kk in 0..k_us {
            let byte = b_host[row * k_us / 2 + kk / 2];
            let nib = if kk % 2 == 0 { byte & 0xF } else { byte >> 4 };
            let scale = e4m3_to_f32(bs_host[row * groups + kk / 16]) as f64;
            wf[kk] = E2M1_LUT[nib as usize] as f64 * scale;
        }
        for t in 0..8usize {
            let mut acc = 0.0f64;
            for kk in 0..k_us {
                acc += bf16_bits_to_f32(a_host[t * k_us + kk]) as f64 * wf[kk];
            }
            let out = bf16_bits_to_f32(got[t * n_us + row]) as f64;
            let tol = (0.02 * acc.abs()).max(0.25);
            let diff = (out - acc).abs();
            worst = worst.max(diff / tol);
            if diff > tol {
                bail!("GATE FAIL: batch8 vs CPU ref at row={row} t={t}: got {out} want {acc}");
            }
        }
    }
    eprintln!(
        "  gate 3   PASS: batch8 @M=8 vs CPU f64 ref, {rows} rows (worst {:.2}x tol)",
        worst
    );

    // Gate 4: pipelined batch8 variants must be BIT-EXACT vs shipping batch8
    // at every M — same virtual-lane order by construction; this proves it.
    for &(kn, kh, _) in kernels
        .iter()
        .filter(|(kn, _, _)| kn.starts_with("batch8_pf") || kn.starts_with("batch8_rt"))
    {
        for &m in M_SWEEP {
            zero_c(g)?;
            launch_batchm(g, b8, a, b, bs, c, m, n, k)?;
            g.synchronize(0)?;
            let reference = read_c(g, c, m as usize * n_us)?;
            zero_c(g)?;
            launch_batchm(g, kh, a, b, bs, c, m, n, k)?;
            g.synchronize(0)?;
            let got = read_c(g, c, m as usize * n_us)?;
            if reference != got {
                let bad = reference.iter().zip(&got).filter(|(x, y)| x != y).count();
                bail!(
                    "GATE FAIL: {kn} != batch8 at M={m} ({bad}/{} elems differ)",
                    got.len()
                );
            }
        }
        eprintln!("  gate 4   PASS: {kn} BIT-EXACT vs batch8 at all M");
    }
    Ok(())
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let peak_gbps: f64 = std::env::var("ATLAS_PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(273.0);

    let kernels: Vec<(&str, KernelHandle, Kind)> = [
        (
            "batch4",
            "w4a16_gemv",
            "w4a16_gemv_batch4",
            Kind::Batchm { max_m: 4 },
        ),
        (
            "batch5",
            "w4a16_gemv",
            "w4a16_gemv_batch5",
            Kind::Batchm { max_m: 5 },
        ),
        (
            "batch6",
            "w4a16_gemv",
            "w4a16_gemv_batch6",
            Kind::Batchm { max_m: 6 },
        ),
        (
            "batch7",
            "w4a16_gemv",
            "w4a16_gemv_batch7",
            Kind::Batchm { max_m: 7 },
        ),
        (
            "batch8",
            "w4a16_gemv",
            "w4a16_gemv_batch8",
            Kind::Batchm { max_m: 8 },
        ),
        (
            "batch16",
            "w4a16_gemv",
            "w4a16_gemv_batch16",
            Kind::Batchm { max_m: 16 },
        ),
        // Lever-1 candidates (2026-08-19): software-pipelined batch8 —
        // weight prefetch, bit-identical FMA order (gate 4 proves it).
        // provenance-id: 526f6e616c6420522e205374657369616b
        (
            "batch8_pf",
            "w4a16_gemv",
            "w4a16_gemv_batch8_pf",
            Kind::Batchm { max_m: 8 },
        ),
        (
            "batch8_pfree",
            "w4a16_gemv",
            "w4a16_gemv_batch8_pf_free",
            Kind::Batchm { max_m: 8 },
        ),
        // v2: activation row-ahead prefetch (the M-scaling fix; weight
        // prefetch benched as a wash 2026-08-19). pf3 = both prefetches.
        (
            "batch8_pf2",
            "w4a16_gemv",
            "w4a16_gemv_batch8_pf2",
            Kind::Batchm { max_m: 8 },
        ),
        (
            "batch8_pf3",
            "w4a16_gemv",
            "w4a16_gemv_batch8_pf3",
            Kind::Batchm { max_m: 8 },
        ),
        // v3: register-tiled, T outputs/thread (activation traffic, load
        // instructions, and FMA-chain ILP all ÷T). Grid stays ceil(N/4):
        // surplus blocks early-exit, coverage and outputs unchanged.
        (
            "batch8_rt2",
            "w4a16_gemv",
            "w4a16_gemv_batch8_rt2",
            Kind::Batchm { max_m: 8 },
        ),
        (
            "batch8_rt4",
            "w4a16_gemv",
            "w4a16_gemv_batch8_rt4",
            Kind::Batchm { max_m: 8 },
        ),
        ("gemm_m64", "w4a16", "w4a16_gemm", Kind::Gemm),
        ("gemm_t", "w4a16", "w4a16_gemm_t", Kind::GemmT),
    ]
    .into_iter()
    .filter_map(|(name, module, func, kind)| match g.kernel(module, func) {
        Ok(h) => Some((name, h, kind)),
        Err(_) => {
            eprintln!("SKIP {name}: {module}::{func} not in this target's module set");
            None
        }
    })
    .collect();

    eprintln!(
        "W4A16 batchm M-sweep (chain-verify widening)  peak {peak_gbps:.0} GB/s, \
         {ITERS} iters, cold-cycled weights\n"
    );

    let mut rng = XorShift(0x9E3779B97F4A7C15);
    for &(label, n, k) in SHAPES {
        let (n_us, k_us) = (n as usize, k as usize);
        let packed_bytes = n_us * k_us / 2;
        let scale_bytes = n_us * k_us / 16;
        let weight_bytes = packed_bytes + scale_bytes;
        let copies = (COLD_CYCLE_BYTES.div_ceil(weight_bytes)).clamp(1, 16);

        // Random activations (bf16 in [-1,1)) + random weights. Scale bytes in
        // [0x18,0x28) → E4M3 2^-4..2^-2 range: keeps |C| ~O(10), no NaN/inf.
        let a_host: Vec<u16> = (0..M_MAX * k_us)
            .map(|_| f32_to_bf16_bits(rng.unit_f32()))
            .collect();
        let b_host: Vec<u8> = (0..packed_bytes).map(|_| rng.byte()).collect();
        let bs_host: Vec<u8> = (0..scale_bytes)
            .map(|_| 0x18 + (rng.byte() & 0x0F))
            .collect();

        let a = g.alloc(M_MAX * k_us * 2)?;
        let c = g.alloc(M_MAX * n_us * 2)?;
        let a_bytes: Vec<u8> = a_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d(&a_bytes, a)?;
        let mut b_copies = Vec::with_capacity(copies);
        for _ in 0..copies {
            let b = g.alloc(packed_bytes)?;
            let bs = g.alloc(scale_bytes)?;
            g.copy_h2d(&b_host, b)?;
            g.copy_h2d(&bs_host, bs)?;
            b_copies.push((b, bs));
        }

        let floor_us = weight_bytes as f64 / (peak_gbps * 1e9) * 1e6;
        eprintln!(
            "── {label}  weights {:.1} MB × {copies} copies, floor {floor_us:.0} us ──",
            weight_bytes as f64 / 1e6
        );

        let (b0, bs0) = b_copies[0];
        correctness_gate(g, &kernels, a, b0, bs0, c, n, k, &a_host, &b_host, &bs_host)?;

        for &(kname, kh, kind) in &kernels {
            for &m in [M_SWEEP, M_WIDE][matches!(kind, Kind::Gemm | Kind::GemmT) as usize] {
                if let Kind::Batchm { max_m } = kind
                    && m > max_m
                {
                    continue;
                }
                let launch = |i: usize| -> Result<()> {
                    let (b, bs) = b_copies[i % copies];
                    match kind {
                        Kind::Batchm { .. } => launch_batchm(g, kh, a, b, bs, c, m, n, k),
                        Kind::Gemm => launch_gemm(g, kh, 64, a, b, bs, c, m, n, k, None),
                        Kind::GemmT => launch_gemm(g, kh, 128, a, b, bs, c, m, n, k, Some(n)),
                    }
                };
                for i in 0..WARMUP {
                    launch(i)?;
                }
                g.synchronize(0)?;
                let t0 = Instant::now();
                for i in 0..ITERS {
                    launch(WARMUP + i)?;
                }
                g.synchronize(0)?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
                let gbps = weight_bytes as f64 / (us * 1e-6) / 1e9;
                eprintln!(
                    "  {kname:<9} M={m}  {us:>9.1} us  {gbps:>6.1} GB/s  {:>5.1}% peak  {:>5.2}x floor",
                    100.0 * gbps / peak_gbps,
                    us / floor_us,
                );
            }
            eprintln!();
        }

        g.free(a)?;
        g.free(c)?;
        for (b, bs) in b_copies {
            g.free(b)?;
            g.free(bs)?;
        }
    }
    Ok(())
}
