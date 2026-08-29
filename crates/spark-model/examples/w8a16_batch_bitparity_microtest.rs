// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity gate for the block-scaled W8A16 batched GEMVs.
//!
//! `w8a16_gemv_batch4` / `w8a16_gemv_batch16` are the arms the K<=8 MTP
//! verify dispatch runs for the GDN QKVZ and out_proj projections on
//! native-FP8 checkpoints (`trait_decode_batched.rs`), while sequential
//! M=1 decode runs `w8a16_gemv` on the SAME weights (`ssm_forward.rs`).
//! Speculative decode at temperature 0 is only output-invariant if the
//! batched arm is byte-identical to M independent M=1 calls.
//!
//! The existing `w8a16_gemv_batch4_microtest` gates on COSINE > 0.99999 at
//! one small shape (N=512, K=2048) — the gate style the w4a16 microtest's
//! doc explicitly warns about ("a cosine gate is exactly what hid the
//! `w8a16_gemv_batch4` fused-add defect"), and its own output shows
//! max_abs=0.125 PASSing at M=16. This test is the strict version: RAW
//! BF16 BYTES, at the EXACT qwen3.8-27b GDN projection shapes, at every M
//! each arm serves in the verify dispatch.
//!
//! Exit: 0 all legs byte-identical, 1 any leg differs.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example w8a16_batch_bitparity_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const MAXM: usize = 16;
const FP8_BLOCK: usize = 128;

/// (label, N, K). The two qwen3.8-27b rows are the EXACT GDN projection
/// shapes the K<=8 verify dispatch serves (QKVZ in_proj [16384 x 5120],
/// out_proj [5120 x 6144] — see the M>8 arm's nsys note in
/// `trait_decode_batched.rs`); the small row is the legacy microtest shape
/// kept for continuity with its history.
const SHAPES: [(&str, usize, usize); 3] = [
    ("legacy   [  512 x 2048]", 512, 2048),
    ("qwen3.8 qkvz    [16384 x 5120]", 16384, 5120),
    ("qwen3.8 out_proj[ 5120 x 6144]", 5120, 6144),
];

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
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n_elems: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_elems * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let batch4_k = g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4")?;
    let batch16_k = g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16")?;
    let m1_k = g.kernel("w8a16_gemv", "w8a16_gemv")?;

    let mut all_pass = true;
    for (label, n, k) in SHAPES {
        let mut rng = Lcg(0x5155_4d5f_b4b4_0001 ^ (n as u64) << 20 ^ k as u64);
        let a: Vec<bf16> = (0..MAXM * k)
            .map(|_| bf16::from_f32(rng.r(-1.0, 1.0)))
            .collect();
        let weight: Vec<u8> = (0..n * k).map(|_| rng.r(0.0, 256.0) as u8).collect();
        let (nb, kb) = (n / FP8_BLOCK, k / FP8_BLOCK);
        let block_scale: Vec<f32> = (0..nb * kb).map(|_| rng.r(0.01, 0.12)).collect();

        let a_d = up_bf16(g, &a)?;
        let w_d = up_u8(g, &weight)?;
        let bs_d = up_f32(g, &block_scale)?;
        let c_batch = g.alloc(MAXM * n * 2)?;
        let c_ref = g.alloc(MAXM * n * 2)?;

        // Every M each arm serves in the verify dispatch: batch4 at 2..4
        // (K=4 MTP verify is M=4 — the production width), batch16 at 5..16.
        let legs: [(&str, KernelHandle, &[usize]); 2] = [
            ("batch4", batch4_k, &[2, 3, 4]),
            ("batch16", batch16_k, &[5, 8, 16]),
        ];
        for (name, kh, ms) in legs {
            for &m in ms {
                KernelLaunch::new(g, kh)
                    .grid([div_ceil(n as u32, 4), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(a_d)
                    .arg_ptr(w_d)
                    .arg_ptr(bs_d)
                    .arg_ptr(c_batch)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(0)?;
                for t in 0..m {
                    KernelLaunch::new(g, m1_k)
                        .grid([div_ceil(n as u32, 4), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(a_d.offset(t * k * 2))
                        .arg_ptr(w_d)
                        .arg_ptr(bs_d)
                        .arg_ptr(c_ref.offset(t * n * 2))
                        .arg_u32(n as u32)
                        .arg_u32(k as u32)
                        .launch(0)?;
                }
                g.synchronize(0)?;
                let cb = dn_bytes(g, c_batch, m * n)?;
                let cr = dn_bytes(g, c_ref, m * n)?;
                let diff = cb
                    .chunks_exact(2)
                    .zip(cr.chunks_exact(2))
                    .filter(|(x, y)| x != y)
                    .count();
                let max_abs = cb
                    .chunks_exact(2)
                    .zip(cr.chunks_exact(2))
                    .map(|(x, y)| {
                        (bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32()
                            - bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32())
                        .abs()
                    })
                    .fold(0.0f32, f32::max);
                let pass = diff == 0;
                all_pass &= pass;
                println!(
                    "{label}  {name} M={m:2}  byte-identical={pass}  diff_elems={diff:<8} max|delta|={max_abs:.6}",
                );
            }
        }
    }
    if all_pass {
        println!(
            "PASS — w8a16_gemv_batch4/batch16 byte-identical to M x w8a16_gemv at every shape and M."
        );
        Ok(())
    } else {
        println!("FAIL — batched W8A16 verify arm is NOT byte-identical to sequential decode.");
        std::process::exit(1);
    }
}
