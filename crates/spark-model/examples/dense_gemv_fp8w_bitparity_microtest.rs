// SPDX-License-Identifier: AGPL-3.0-only

//! Bit-parity gate for the FP8 row-scaled vocab GEMV: every batched arm must
//! be byte-identical to M independent `dense_gemv_fp8w` calls.
//!
//! WHY THIS SHAPE MATTERS. This kernel family serves the TARGET's LM head.
//! Under strict-argmax accept a drafter cannot change token identity, so a
//! spec-on vs spec-off divergence at temp 0 is target-side arithmetic by
//! construction — and the batched arm running at M>1 while serial decode runs
//! M=1 is exactly such a difference, on the last projection before the argmax.
//! A reduction-order discrepancy here is a different SAMPLED TOKEN, not a
//! rounding curiosity. Same class as the w8a16 defect in #653.
//!
//! Two arms are checked:
//!   * `dense_gemv_fp8w_batch2` — the pre-existing M=2 arm, whose `mac4`
//!     contracts four products before accumulating rather than accumulating
//!     one per statement as the M=1 kernel does. This gate exists to say
//!     whether that is a real byte difference or only a reading of the source.
//!   * `dense_gemv_fp8w_batchm` — the M<=8 arm, written to the M=1 chain
//!     deliberately (same k order, per-row scale applied to each thread's
//!     partial BEFORE the reduction, as the M=1 kernel does).
//!
//! Compares BF16 BYTES at the real LM-head shape plus two smaller ones.
//! Exit: 0 all legs byte-identical, 1 any leg differs.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example dense_gemv_fp8w_bitparity_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const MAXM: usize = 8;

/// (label, N, K). The qwen3.8-27b row is the real LM head: the checkpoint's
/// native FP8 `lm_head.weight` is [248320 x 5120] (vocab 248077 padded).
const SHAPES: [(&str, usize, usize); 3] = [
    ("small       [  512 x 2048]", 512, 2048),
    ("mid         [ 8192 x 5120]", 8192, 5120),
    ("qwen3.8 lm_head [248320 x 5120]", 248320, 5120),
];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        let x = ((self.next() >> 11) as f32) / ((1u64 << 53) as f32);
        lo + x * (hi - lo)
    }
}

fn up_u8(g: &dyn GpuBackend, d: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(d.len())?;
    g.copy_h2d(d, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|v| v.to_bits().to_ne_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn down(g: &dyn GpuBackend, p: DevicePtr, elems: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; elems * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let m1_k = g.kernel("gemv_fp8w", "dense_gemv_fp8w")?;
    let batch2_k = g.kernel("dense_gemv_fp8w_batch2", "dense_gemv_fp8w_batch2")?;
    let batchm_k = g.kernel("dense_gemv_fp8w_batchm", "dense_gemv_fp8w_batchm")?;

    let mut all_pass = true;
    for (label, n, k) in SHAPES {
        let mut rng = Lcg(0x4650_3857_0000_0001 ^ (n as u64) << 20 ^ k as u64);
        let a: Vec<bf16> = (0..MAXM * k)
            .map(|_| bf16::from_f32(rng.r(-1.0, 1.0)))
            .collect();
        // FP8 E4M3 bytes: avoid 0xFF/0x7F (NaN/Inf encodings) so the compare
        // measures arithmetic order, not NaN propagation.
        let weight: Vec<u8> = (0..n * k)
            .map(|_| (rng.r(0.0, 240.0) as u8) & 0x7E)
            .collect();
        let row_scale: Vec<f32> = (0..n).map(|_| rng.r(0.01, 0.12)).collect();

        let a_d = up_bf16(g, &a)?;
        let w_d = up_u8(g, &weight)?;
        let rs_d = up_f32(g, &row_scale)?;
        let c_batch = g.alloc(MAXM * n * 2)?;
        let c_ref = g.alloc(MAXM * n * 2)?;

        // batch2 serves exactly M=2; batchm serves 2..=8.
        let legs: [(&str, KernelHandle, &[usize]); 2] = [
            ("batch2", batch2_k, &[2]),
            ("batchm", batchm_k, &[2, 3, 4, 8]),
        ];

        for (name, kh, ms) in legs {
            for &m in ms {
                let mut launch = KernelLaunch::new(g, kh)
                    .grid([div_ceil(n as u32, 4), 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(a_d)
                    .arg_ptr(w_d)
                    .arg_ptr(rs_d)
                    .arg_ptr(c_batch);
                // batch2's signature is fixed at M=2 and takes (N, K) only.
                if name != "batch2" {
                    launch = launch.arg_u32(m as u32);
                }
                launch.arg_u32(n as u32).arg_u32(k as u32).launch(0)?;

                for t in 0..m {
                    KernelLaunch::new(g, m1_k)
                        .grid([div_ceil(n as u32, 4), 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(a_d.offset(t * k * 2))
                        .arg_ptr(w_d)
                        .arg_ptr(rs_d)
                        .arg_ptr(c_ref.offset(t * n * 2))
                        .arg_u32(n as u32)
                        .arg_u32(k as u32)
                        .launch(0)?;
                }
                g.synchronize(0)?;

                let cb = down(g, c_batch, m * n)?;
                let cr = down(g, c_ref, m * n)?;
                let diff = cb.iter().zip(&cr).filter(|(x, y)| x != y).count();
                let max_abs = cb
                    .iter()
                    .zip(&cr)
                    .map(|(x, y)| {
                        let fx = f32::from(bf16::from_bits(*x));
                        let fy = f32::from(bf16::from_bits(*y));
                        (fx - fy).abs()
                    })
                    .fold(0.0f32, f32::max);
                let pass = diff == 0;
                all_pass &= pass;
                println!(
                    "{label}  {name} M={m:2}  byte-identical={pass}  diff_elems={diff:<8} \
                     max|delta|={max_abs:.6}"
                );
            }
        }
    }

    if all_pass {
        println!(
            "PASS — every batched FP8 vocab GEMV arm is byte-identical to M x dense_gemv_fp8w."
        );
        Ok(())
    } else {
        println!(
            "FAIL — a batched FP8 vocab arm is NOT byte-identical to sequential decode. \
             At the LM head that is a different SAMPLED TOKEN, not a rounding curiosity."
        );
        std::process::exit(1);
    }
}
