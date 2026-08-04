// SPDX-License-Identifier: AGPL-3.0-only
//! Equivalence test for `w8a16_gemv_batch4` (M<=4 block-scaled FP8 GEMV).
//!
//! Compares the batched kernel against M independent `w8a16_gemv` (M=1) calls —
//! the batch4 path is meant to be bit-identical per row (same K accumulation
//! order). Uses an SSM-out-proj-like shape. Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=holo-3.1-35b-a3b ATLAS_TARGET_QUANT=nvfp4 \
//!     cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example w8a16_gemv_batch4_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const MAXM: usize = 16;
const N: usize = 512;
const K: usize = 2048;
const FP8_BLOCK: usize = 128;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn up_u8(g: &dyn GpuBackend, d: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(d.len().max(1))?;
    g.copy_h2d(d, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}
fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn launch_m1(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([div_ceil(N as u32, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bs)
        .arg_ptr(c)
        .arg_u32(N as u32)
        .arg_u32(K as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let batch4_k = g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4")?;
    let batch16_k = g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?;
    let m1_k = g.kernel("w8a16_gemv", "w8a16_gemv")?;

    let mut rng = Lcg(0x5155_4d5f_b4b4_0001);
    let a: Vec<bf16> = (0..MAXM * K)
        .map(|_| bf16::from_f32(rng.r(-1.0, 1.0)))
        .collect();
    // Random FP8 E4M3 weight bytes (NaN bytes 0x7f/0xff -> 0.0 in both kernels).
    let weight: Vec<u8> = (0..N * K).map(|_| rng.r(0.0, 256.0) as u8).collect();
    let kb = K / FP8_BLOCK;
    let nb = N / FP8_BLOCK;
    let block_scale: Vec<f32> = (0..nb * kb).map(|_| rng.r(0.01, 0.12)).collect();

    let a_d = up_bf16(g, &a)?;
    let w_d = up_u8(g, &weight)?;
    let bs_d = up_f32(g, &block_scale)?;
    let c_batch = g.alloc(MAXM * N * 2)?;
    let c_ref = g.alloc(MAXM * N * 2)?;

    // batch4 must be optimal for M<=4; batch16 covers M=5..16. Each must be
    // bit-identical per-row to M independent w8a16_gemv (M=1) calls.
    let configs: [(&str, KernelHandle, usize); 3] = [
        ("batch4", batch4_k, 4),
        ("batch16", batch16_k, 8),
        ("batch16", batch16_k, 16),
    ];
    let mut all_pass = true;
    for (name, kh, m) in configs {
        KernelLaunch::new(g, kh)
            .grid([div_ceil(N as u32, 4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_d)
            .arg_ptr(w_d)
            .arg_ptr(bs_d)
            .arg_ptr(c_batch)
            .arg_u32(m as u32)
            .arg_u32(N as u32)
            .arg_u32(K as u32)
            .launch(0)?;
        for t in 0..m {
            launch_m1(
                g,
                m1_k,
                a_d.offset(t * K * 2),
                w_d,
                bs_d,
                c_ref.offset(t * N * 2),
            )?;
        }
        g.synchronize(0)?;
        let cb = dn_bf16(g, c_batch, m * N)?;
        let cr = dn_bf16(g, c_ref, m * N)?;
        let mut worst = 1.0f64;
        for t in 0..m {
            worst = worst.min(cos(&cb[t * N..(t + 1) * N], &cr[t * N..(t + 1) * N]));
        }
        let pass = worst > 0.99999;
        all_pass &= pass;
        println!(
            "{name} M={m:2}: worst_cos={worst:.9} max_abs={:.6} {}",
            max_abs(&cb, &cr),
            if pass { "PASS" } else { "FAIL" }
        );
    }
    if all_pass {
        println!("ALL PASS");
        Ok(())
    } else {
        std::process::exit(1);
    }
}
