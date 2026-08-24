// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 drafter-propose kernel shoot-out at M ∈ {8, 16} on the real
//! block-16-drafter shapes (Apathy-v2 class: hidden 5120, inter 17408,
//! vocab 248320) — picks the propose GEMV for γ=16 drafters without a
//! server boot per candidate.
//!
//! Context (nsys node traces, 2026-08-24): the fp8 tile family
//! (`fp8_gemm_t_row_scaled*`, cp.async scattered-row) is structurally
//! capped at ~90-107 GB/s on GB10 — z-sliced split-K, 4-warp, and
//! full-64B-line revisions all pinned there. `rt16` (K-major lane-striped
//! LDG) reached 119-144 GB/s live; `dense_gemv_fp8w_batchm` measured
//! 217 GB/s on the vocab shape at M=8. This bench compares, cold-weight:
//!
//!   tile_m16  fp8_gemm_t_row_scaled_m16        (w4a16, grid N/128×32t)
//!   rt16      fp8_gemv_rowscale_batch16_rt2    (fp8_gemv_rt, N/8×256t)
//!   bm16      dense_gemv_fp8w_batch16m         (batchm impl<16>, N/4×256t)
//!
//! COLD WEIGHTS: iterations cycle weight copies past L2/SLC so every
//! launch streams from DRAM like a real propose step.
//!
//! Correctness: cross-kernel max-abs-diff of C against bm16 (all three
//! compute the same fp32 math; only summation order differs, so the gate
//! is a loose float tolerance, not bit equality — drafter-side contract).
//!
//!   cargo run -p spark-model --release --example fp8_propose_bench \
//!       --features cuda,gpu-examples
//!
//! Env: ATLAS_PEAK_GBPS (default 273 — GB10 LPDDR5x) for the %-of-peak column.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

#[path = "batchm_bench/dtype.rs"]
#[allow(dead_code)] // shared module; this bench uses a subset
mod dtype;
use dtype::{XorShift, bf16_bits_to_f32, f32_to_bf16_bits};

const WARMUP: usize = 5;
const ITERS: usize = 20;
const COLD_CYCLE_BYTES: usize = 256 << 20;
const M_SWEEP: &[u32] = &[8, 16];
const M_MAX: usize = 16;

/// Drafter shapes: (label, N, K). C[M,N] = A[M,K] · dequant(B[N,K]) · rs[N].
const SHAPES: &[(&str, u32, u32)] = &[
    ("kv      N=1024   K=5120 ", 1024, 5120),
    ("q       N=4096   K=5120 ", 4096, 5120),
    ("o       N=5120   K=5120 ", 5120, 5120),
    ("gate/up N=17408  K=5120 ", 17408, 5120),
    ("down    N=5120   K=17408", 5120, 17408),
    ("head    N=248320 K=5120 ", 248320, 5120),
];

#[derive(Clone, Copy)]
enum Kind {
    /// grid (N/128, 1, 1), block 32 — single-warp tile.
    TileM16,
    /// grid (N/8, 1, 1), block 256 — rt 4-group T=2.
    Rt16,
    /// grid (N/4, 1, 1), block 256 — batchm 4-outputs.
    Bm16,
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    kind: Kind,
    a: DevicePtr,
    b: DevicePtr,
    rs: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    let grid = match kind {
        Kind::TileM16 => [div_ceil(n, 128), 1, 1],
        Kind::Rt16 => [div_ceil(n, 8), 1, 1],
        Kind::Bm16 => [div_ceil(n, 4), 1, 1],
    };
    let block = match kind {
        Kind::TileM16 => [32, 1, 1],
        _ => [256, 1, 1],
    };
    KernelLaunch::new(g, kh)
        .grid(grid)
        .block(block)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(rs)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

fn main() -> Result<()> {
    // The default ptx_modules() is target 0 (not qwen3.8) — pick the
    // qwen3.8-27b/nvfp4 set, whose w4a16 module carries the fp8 propose
    // kernels this bench compares.
    let modules = atlas_kernels::all_ptx_sets()
        .into_iter()
        .find(|s| s.target.model == "qwen3.8-27b" && s.target.quant == "nvfp4")
        .map(|s| s.modules)
        .unwrap_or_else(atlas_kernels::ptx_modules);
    let g0 = AtlasCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &g0;
    let peak_gbps: f64 = std::env::var("ATLAS_PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(273.0);

    let kernels: Vec<(&str, KernelHandle, Kind)> = vec![
        (
            "bm16",
            g0.kernel("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batch16m")?,
            Kind::Bm16,
        ),
        (
            "rt16",
            g0.kernel("fp8_gemv_rt", "fp8_gemv_rowscale_batch16_rt2")?,
            Kind::Rt16,
        ),
        (
            "tile_m16",
            g0.kernel("w4a16", "fp8_gemm_t_row_scaled_m16")?,
            Kind::TileM16,
        ),
    ];

    let mut rng = XorShift(0x5eed_f00d);

    println!(
        "fp8 propose GEMV shoot-out — cold weights ({} MB cycle), {} iters, peak {:.0} GB/s",
        COLD_CYCLE_BYTES >> 20,
        ITERS,
        peak_gbps,
    );

    for &(label, n, k) in SHAPES {
        let n_us = n as usize;
        let k_us = k as usize;
        let w_bytes = n_us * k_us;
        let copies = (COLD_CYCLE_BYTES / w_bytes).clamp(1, 8);

        // Host A: bf16 in [-1, 1). Host B: raw bytes with NaN patterns
        // (0x7F/0xFF) rewritten — dequant of NaN would poison the diff gate.
        let mut a_host = vec![0u8; M_MAX * k_us * 2];
        for i in 0..(M_MAX * k_us) {
            let v = rng.unit_f32();
            let bits = f32_to_bf16_bits(v);
            a_host[i * 2] = (bits & 0xFF) as u8;
            a_host[i * 2 + 1] = (bits >> 8) as u8;
        }
        let mut b_host = vec![0u8; w_bytes];
        for x in b_host.iter_mut() {
            let mut byte = rng.byte();
            if byte & 0x7F == 0x7F {
                byte &= 0xEF;
            }
            *x = byte;
        }
        let mut rs_host = vec![0u8; n_us * 4];
        for i in 0..n_us {
            let v = 1.0 + 0.5 * rng.unit_f32();
            rs_host[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }

        let a = g.alloc(a_host.len())?;
        g.copy_h2d(&a_host, a)?;
        let rs = g.alloc(rs_host.len())?;
        g.copy_h2d(&rs_host, rs)?;
        let c = g.alloc(M_MAX * n_us * 2)?;
        let mut bs = Vec::with_capacity(copies);
        for _ in 0..copies {
            let b = g.alloc(w_bytes)?;
            g.copy_h2d(&b_host, b)?;
            bs.push(b);
        }

        println!("\n{label}  ({:.1} MB weights × {copies} copies)", w_bytes as f64 / 1e6);
        println!(
            "  {:>4} {:>9} {:>9} {:>7} {:>6}  {:>10}",
            "M", "kernel", "us/call", "GB/s", "%peak", "maxdiff"
        );

        for &m in M_SWEEP {
            // Reference C = first kernel in the list (same fp32 math family).
            let mut c_ref: Vec<u16> = Vec::new();
            for &(kname, kh, kind) in &kernels {
                // Correctness pass on copy 0.
                launch(g, kh, kind, a, bs[0], rs, c, m, n, k)?;
                g.synchronize(0)?;
                let mut c_host = vec![0u8; m as usize * n_us * 2];
                g.copy_d2h(c, &mut c_host)?;
                let c_bits: Vec<u16> = c_host
                    .chunks_exact(2)
                    .map(|p| u16::from_le_bytes([p[0], p[1]]))
                    .collect();
                let maxdiff = if c_ref.is_empty() {
                    c_ref = c_bits.clone();
                    0.0f32
                } else {
                    let mut d = 0.0f32;
                    for (x, y) in c_bits.iter().zip(c_ref.iter()) {
                        d = d.max((bf16_bits_to_f32(*x) - bf16_bits_to_f32(*y)).abs());
                    }
                    d
                };
                // Loose float gate: same math, different summation order.
                // K=17408 dot of ~unit terms: bf16 output granularity alone
                // allows ~1.0 at |acc|~100; anything past 4.0 is a real bug.
                if maxdiff > 4.0 {
                    println!("  !! {kname} M={m}: maxdiff {maxdiff} vs reference — BROKEN");
                }

                // Timed cold loop.
                for i in 0..WARMUP {
                    launch(g, kh, kind, a, bs[i % copies], rs, c, m, n, k)?;
                }
                g.synchronize(0)?;
                let t0 = Instant::now();
                for i in 0..ITERS {
                    launch(g, kh, kind, a, bs[i % copies], rs, c, m, n, k)?;
                }
                g.synchronize(0)?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
                let gbps = w_bytes as f64 / (us * 1e-6) / 1e9;
                println!(
                    "  {:>4} {:>9} {:>9.1} {:>7.0} {:>5.0}%  {:>10.3}",
                    m,
                    kname,
                    us,
                    gbps,
                    100.0 * gbps / peak_gbps,
                    maxdiff,
                );
            }
        }

        for b in bs {
            g.free(b)?;
        }
        g.free(a)?;
        g.free(rs)?;
        g.free(c)?;
    }
    Ok(())
}
