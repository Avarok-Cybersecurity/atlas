// SPDX-License-Identifier: AGPL-3.0-only

//! Speed-of-light bench for the W4A16 (NVFP4) GEMM family at DFlash-verify
//! shapes (M=17) — the kernels the `--cuda-graph-trace=node` profile ranked
//! #1 and #2 in the 233ms verify step:
//!
//!   w4a16_gemm_t_m128  ~100ms/step  (dense-FFN projections, 128x128 M-tile
//!                                    built for prefill, dispatched at M=17
//!                                    with no M threshold — 87% tile padding)
//!   w4a16_gemm         ~81ms/step   (batched verify projections)
//!   w4a16_gemm_t       ~14ms/step   (SSM projections, M<=128 arm)
//!
//! At M=17 these GEMMs are weight-bandwidth-bound in theory: the floor is
//! (K*N/2 packed E2M1 + K*N/16 FP8 scales) bytes / peak-BW. The profile puts
//! the family ~4x above that floor. This bench isolates each kernel on the
//! real Qwen3.6-27B shapes and sweeps M to show where the efficiency cliff
//! is — data for choosing between (a) an M-threshold dispatch fix, (b) a
//! small-M tile variant, (c) accepting the floor.
//!
//! Timing is wall-clock around N back-to-back launches on stream 0 with a
//! final sync (same style as the step-level numbers we're explaining; CUDA
//! launch overhead at these sizes is <1% of a 300us kernel).
//!
//! Buffer CONTENT is arbitrary (timing only — GEMM time is not
//! data-dependent); buffer SIZES are exact. No correctness checked here;
//! that's the microtests' job.
//!
//!   cargo run -p spark-model --release --example w4a16_m17_bench \
//!       --features cuda,gpu-examples
//!
//! Env: ATLAS_PEAK_GBPS (default 273 — GB10 LPDDR5x) for the %-of-peak column.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const WARMUP: usize = 20;
const ITERS: usize = 100;

/// Qwen3.6-27B verify shapes (hidden=5120, ffn intermediate=17408).
/// (label, N, K) with C[M,N] = A[M,K] · W.
const SHAPES: &[(&str, u32, u32)] = &[
    ("ffn_gate/up  N=17408 K=5120 ", 17408, 5120),
    ("ffn_down     N=5120  K=17408", 5120, 17408),
];

const M_SWEEP: &[u32] = &[17, 32, 64, 128];

/// Launch geometry per kernel, mirroring the ops wrappers exactly.
#[derive(Clone, Copy)]
enum Geom {
    /// w4a16_gemm: grid (N/64, M/64), block 128
    N64M64,
    /// w4a16_gemm_t via w4a16_gemm_n128: grid (N/128, M/64), block 128
    N128M64,
    /// w4a16_gemm_t_m128 via w4a16_gemm_n128_m128: grid (N/128, M/128), block 128
    N128M128,
}

fn grid_for(g: Geom, m: u32, n: u32) -> [u32; 3] {
    match g {
        Geom::N64M64 => [div_ceil(n, 64), div_ceil(m, 64), 1],
        Geom::N128M64 => [div_ceil(n, 128), div_ceil(m, 64), 1],
        Geom::N128M128 => [div_ceil(n, 128), div_ceil(m, 128), 1],
    }
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    k_h: KernelHandle,
    geom: Geom,
    a: DevicePtr,
    b: DevicePtr,
    b_scale: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, k_h)
        .grid(grid_for(geom, m, n))
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(b_scale)
        .arg_f32(1.0) // weight_scale_2
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    let peak_gbps: f64 = std::env::var("ATLAS_PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(273.0);

    // (display name, module, function, geometry)
    let kernels: Vec<(&str, KernelHandle, Geom)> = [
        ("w4a16_gemm       (N64,M64) ", "w4a16_gemm", Geom::N64M64),
        ("w4a16_gemm_t     (N128,M64)", "w4a16_gemm_t", Geom::N128M64),
        (
            "w4a16_gemm_t_k64 (N128,M64)",
            "w4a16_gemm_t_k64",
            Geom::N128M64,
        ),
        (
            "w4a16_gemm_t_m128(N128,M128)",
            "w4a16_gemm_t_m128",
            Geom::N128M128,
        ),
    ]
    .into_iter()
    .filter_map(|(name, func, geom)| match g.kernel("w4a16", func) {
        Ok(h) => Some((name, h, geom)),
        Err(_) => {
            eprintln!("SKIP {name}: kernel w4a16::{func} not in this target's module set");
            None
        }
    })
    .collect();

    let m_max = *M_SWEEP.iter().max().unwrap() as usize;

    eprintln!(
        "W4A16 M=17 speed-of-light bench  (peak {peak_gbps:.0} GB/s, \
         {ITERS} iters, floor = packed(K*N/2) + scales(K*N/16))\n"
    );

    for &(label, n, k) in SHAPES {
        let (n_us, k_us) = (n as usize, k as usize);
        // Exact production sizes; content arbitrary (0x5A fill).
        let a = g.alloc(m_max * k_us * 2)?; // BF16 activations
        let b = g.alloc(k_us * n_us / 2)?; // packed E2M1
        let b_scale = g.alloc(k_us * n_us / 16)?; // FP8 group-16 scales
        let c = g.alloc(m_max * n_us * 2)?; // BF16 out
        for (p, bytes) in [
            (a, m_max * k_us * 2),
            (b, k_us * n_us / 2),
            (b_scale, k_us * n_us / 16),
        ] {
            g.memset(p, 0x5A, bytes)?;
        }

        let weight_bytes = (k_us * n_us / 2 + k_us * n_us / 16) as f64;
        let floor_us = weight_bytes / (peak_gbps * 1e9) * 1e6;
        eprintln!(
            "── {label}  weights {:.1} MB, floor {floor_us:.0} us @ {peak_gbps:.0} GB/s ──",
            weight_bytes / 1e6
        );

        for &(kname, kh, geom) in &kernels {
            for &m in M_SWEEP {
                for _ in 0..WARMUP {
                    launch(g, kh, geom, a, b, b_scale, c, m, n, k)?;
                }
                g.synchronize(0)?;
                let t0 = Instant::now();
                for _ in 0..ITERS {
                    launch(g, kh, geom, a, b, b_scale, c, m, n, k)?;
                }
                g.synchronize(0)?;
                let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
                let gbps = weight_bytes / (us * 1e-6) / 1e9;
                eprintln!(
                    "  {kname}  M={m:>3}  {us:>8.1} us  {gbps:>6.1} GB/s  \
                     {:>5.1}% of peak  {:>4.2}x floor",
                    100.0 * gbps / peak_gbps,
                    us / floor_us,
                );
            }
            eprintln!();
        }

        // Per-step projection: 64 dense-FFN layers x (gate + up + down).
        // gate/up use this shape only when label matches; print once per shape.
        for p in [a, b, b_scale, c] {
            let _ = g.free(p);
        }
    }

    eprintln!(
        "context: verify has 64 layers x (gate,up @ N=17408) + 64 x (down @ N=5120K17408).\n\
         per-step FFN cost = 128 x t(gate/up shape, M=17) + 64 x t(down shape, M=17).\n\
         profile said ~100ms via w4a16_gemm_t_m128 — compare against that here."
    );
    Ok(())
}
