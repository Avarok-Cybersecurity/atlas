// SPDX-License-Identifier: AGPL-3.0-only
//! Equivalence + speed test for `dense_gemv_bf16_batchm` (M-row BF16 GEMV).
//!
//! THE GATING EXPERIMENT for batched BF16 decode projections. At decode,
//! Laguna's q/k/v/o and shared-expert projections are BF16 and had no batched
//! tier, so each concurrent sequence re-read the whole weight matrix — 54% of
//! the decode step scaled linearly with concurrency.
//!
//! Checks two things at the REAL decode shapes (K=3072; N=9216 q_proj,
//! N=3072 o_proj, N=1024 k/v/shared-expert):
//!   1. CORRECTNESS: batchm output is bit-identical to M separate M=1
//!      `dense_gemv_bf16` calls (same K order, --fmad=false).
//!   2. SPEED: batchm(M) vs M separate M=1 launches. PASS = batchm is
//!      meaningfully faster at M=2 and M=4 (ideally ~M x, since the weight read
//!      is paid once instead of M times).
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=laguna-s-2.1 ATLAS_TARGET_QUANT=nvfp4 \
//!     cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example dense_gemv_bf16_batchm_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// (N, K, label) at the REAL decode shapes. o_proj is N=h=3072, K=nq*hd=9216 —
/// the same weight bytes as q_proj, not the 3072x3072 it looks like.
const SHAPES: &[(usize, usize, &str)] = &[
    (9216, 3072, "q_proj"),
    (3072, 9216, "o_proj"),
    (1024, 3072, "k/v/shared"),
];
const MS: &[usize] = &[1, 2, 4];
const ITERS: usize = 50;
const WARMUP: usize = 10;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self) -> f32 {
        -1.0 + 2.0 * self.f()
    }
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_bits(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn launch_m1(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    n: usize,
    k_dim: usize,
) -> Result<()> {
    KernelLaunch::new(g, kern)
        .grid([div_ceil(n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(n as u32)
        .arg_u32(k_dim as u32)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn launch_batchm(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k_dim: usize,
    out_stride: usize,
) -> Result<()> {
    KernelLaunch::new(g, kern)
        .grid([div_ceil(n as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k_dim as u32)
        .arg_u32(out_stride as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let m1_k = g.kernel("gemv", "dense_gemv_bf16")?;
    let bm_k = g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;

    let mut rng = Lcg(0x5eed_1234);
    let mut fail = 0usize;

    println!(
        "{:<12} {:>5} {:>10} {:>12} {:>12} {:>9}  {}",
        "shape", "M", "N", "Mx M=1 (ms)", "batchm (ms)", "speedup", "bit-identical"
    );

    for &(n, k_dim, label) in SHAPES {
        let w: Vec<bf16> = (0..n * k_dim).map(|_| bf16::from_f32(rng.r())).collect();
        let wd = up_bf16(g, &w)?;

        for &m in MS {
            let a: Vec<bf16> = (0..m * k_dim).map(|_| bf16::from_f32(rng.r())).collect();
            let ad = up_bf16(g, &a)?;
            let c_ref = g.alloc(m * n * 2)?;
            let c_bat = g.alloc(m * n * 2)?;

            // ---- correctness: M separate M=1 calls vs one batchm call ----
            for t in 0..m {
                launch_m1(
                    g,
                    m1_k,
                    ad.offset(t * k_dim * 2),
                    wd,
                    c_ref.offset(t * n * 2),
                    n,
                    k_dim,
                )?;
            }
            launch_batchm(g, bm_k, ad, wd, c_bat, m, n, k_dim, n)?;
            g.synchronize(0)?;

            let r = dn_bits(g, c_ref, m * n)?;
            let b = dn_bits(g, c_bat, m * n)?;
            let identical = r == b;
            let ndiff = r.iter().zip(&b).filter(|(x, y)| x != y).count();
            if !identical {
                fail += 1;
            }

            // ---- speed ----
            for _ in 0..WARMUP {
                for t in 0..m {
                    launch_m1(
                        g,
                        m1_k,
                        ad.offset(t * k_dim * 2),
                        wd,
                        c_ref.offset(t * n * 2),
                        n,
                        k_dim,
                    )?;
                }
                launch_batchm(g, bm_k, ad, wd, c_bat, m, n, k_dim, n)?;
            }
            g.synchronize(0)?;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                for t in 0..m {
                    launch_m1(
                        g,
                        m1_k,
                        ad.offset(t * k_dim * 2),
                        wd,
                        c_ref.offset(t * n * 2),
                        n,
                        k_dim,
                    )?;
                }
            }
            g.synchronize(0)?;
            let t_ref = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

            let t1 = std::time::Instant::now();
            for _ in 0..ITERS {
                launch_batchm(g, bm_k, ad, wd, c_bat, m, n, k_dim, n)?;
            }
            g.synchronize(0)?;
            let t_bat = t1.elapsed().as_secs_f64() * 1e3 / ITERS as f64;

            println!(
                "{:<12} {:>5} {:>10} {:>12.4} {:>12.4} {:>8.2}x  {}",
                label,
                m,
                n,
                t_ref,
                t_bat,
                t_ref / t_bat,
                if identical {
                    "yes".to_string()
                } else {
                    format!("NO ({ndiff} elems differ)")
                }
            );

            g.free(ad)?;
            g.free(c_ref)?;
            g.free(c_bat)?;
        }
        g.free(wd)?;
    }

    if fail > 0 {
        anyhow::bail!("{fail} shape/M combinations were NOT bit-identical");
    }
    println!("\nAll bit-identical. PASS criterion: speedup should approach Mx at M=2 and M=4.");
    Ok(())
}
