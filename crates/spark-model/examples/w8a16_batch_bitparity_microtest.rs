// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity gate for the batched w8a16 GEMV tiers.
//!
//! `kernels/gb10/common/w8a16_gemv_batch4.cu` documents itself as
//! bit-identical to running `w8a16_gemv` M times, and every batched-decode
//! ladder that prefers it over M separate `w8a16_gemv` launches — it saves
//! weight DRAM traffic, nothing else — depends on that claim holding.
//!
//! The claim cannot be taken from the header comment: until 2026-08-12 the
//! batched inner loop accumulated `acc += lo*w0 + hi*w1` where `w8a16_gemv`
//! accumulates `acc += lo*w0; acc += hi*w1;` — a different association in
//! exact FP32, which under `--fmad=false` is a different result. The old gate
//! was a `cos >= 0.9999` microtest at a 512-column shape, which cannot resolve
//! a handful of flipped elements.
//!
//! So this test compares RAW BF16 BYTES, not a cosine, at production-scale
//! projection shapes, over three seeds and every M both batched tiers serve.
//!
//! `w8a16_gemv_batch16` shares the same templated body and is held to the same
//! byte standard here.
//!
//! FAILS WITHOUT the association fix in `w8a16_gemv_batch4.cu`: with the fused
//! `acc += x + y` restored, every one of the 36 legs below reports
//! `byte-identical=false` (1-9 differing BF16 elements, max|delta| up to 4.0).
//!
//! Exit: 0 all legs byte-identical, 1 any leg differs,
//! 2 kernels absent from this target's module set.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=nemotron-3-nano-30b-a3b \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example w8a16_batch_bitparity_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const FP8_BLOCK: usize = 128;

/// Lightning-30B-A3B: hidden 2688, d_inner 4096, in_proj_size 10304.
/// Both projections, at the exact N/K the SSM layer dispatches.
const SHAPES: [(&str, usize, usize); 2] = [
    ("in_proj  [10304 x 2688]", 10304, 2688),
    ("out_proj [ 2688 x 4096]", 2688, 4096),
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

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn worst_delta(a: &[u8], b: &[u8]) -> (usize, f32) {
    let mut n_diff = 0usize;
    let mut worst = 0f32;
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        if x != y {
            n_diff += 1;
            let fx = bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32();
            let fy = bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
            worst = worst.max((fx - fy).abs());
        }
    }
    (n_diff, worst)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    m: Option<u32>,
    n: u32,
    k: u32,
) -> Result<()> {
    let mut l = KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(bs)
        .arg_ptr(c);
    if let Some(m) = m {
        l = l.arg_u32(m);
    }
    l.arg_u32(n).arg_u32(k).launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let (batch4_k, batch16_k, m1_k) = match (
        g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch4"),
        g.kernel("w8a16_gemv_batch4", "w8a16_gemv_batch16"),
        g.kernel("w8a16_gemv", "w8a16_gemv"),
    ) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => {
            println!("w8a16 GEMV kernels absent from this target set — SKIP");
            std::process::exit(2);
        }
    };

    let mut batch4_clean = true;
    for seed in [1u64, 99, 12345] {
        for (label, n, k) in SHAPES {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB4B4);
            let max_m = 16usize;
            // Activations: [max_m, k] BF16.
            let a_bytes: Vec<u8> = (0..max_m * k)
                .flat_map(|_| bf16::from_f32(rng.r(-1.5, 1.5)).to_bits().to_le_bytes())
                .collect();
            // Weights: [n, k] raw FP8 E4M3 bytes.
            let w_bytes: Vec<u8> = (0..n * k).map(|_| rng.r(0.0, 256.0) as u8).collect();
            let kb = k.div_ceil(FP8_BLOCK);
            let nb = n.div_ceil(FP8_BLOCK);
            let bs_bytes: Vec<u8> = (0..nb * kb)
                .flat_map(|_| rng.r(0.005, 0.15).to_le_bytes())
                .collect();

            let a_d = up(g, &a_bytes)?;
            let w_d = up(g, &w_bytes)?;
            let bs_d = up(g, &bs_bytes)?;
            let c_batch = g.alloc(max_m * n * 2)?;
            let c_ref = g.alloc(max_m * n * 2)?;

            for (tier, kh, ms) in [
                ("batch4 ", batch4_k, vec![2usize, 3, 4]),
                ("batch16", batch16_k, vec![8usize, 12, 16]),
            ] {
                for m in ms {
                    g.memset(c_batch, 0, max_m * n * 2)?;
                    g.memset(c_ref, 0, max_m * n * 2)?;
                    launch(
                        g,
                        kh,
                        a_d,
                        w_d,
                        bs_d,
                        c_batch,
                        Some(m as u32),
                        n as u32,
                        k as u32,
                    )?;
                    for t in 0..m {
                        launch(
                            g,
                            m1_k,
                            a_d.offset(t * k * 2),
                            w_d,
                            bs_d,
                            c_ref.offset(t * n * 2),
                            None,
                            n as u32,
                            k as u32,
                        )?;
                    }
                    g.synchronize(0)?;
                    let cb = down(g, c_batch, m * n * 2)?;
                    let cr = down(g, c_ref, m * n * 2)?;
                    let identical = cb == cr;
                    let (n_diff, worst) = worst_delta(&cb, &cr);
                    batch4_clean &= identical;
                    let _ = tier;
                    println!(
                        "seed {seed:>5}  {label}  {tier} M={m:<3} byte-identical={identical:<5} \
                         diff_elems={n_diff:<7} max|delta|={worst:.6}"
                    );
                }
            }
            for p in [a_d, w_d, bs_d, c_batch, c_ref] {
                g.free(p).ok();
            }
        }
    }

    if batch4_clean {
        println!(
            "PASS — the w8a16 batched GEMV tiers are byte-identical to M x w8a16_gemv at \
             every Lightning projection shape and every M they serve."
        );
        Ok(())
    } else {
        println!(
            "FAIL — a batched tier is NOT byte-identical to M x w8a16_gemv; no decode ladder \
             may substitute it for the M=1 kernel while this holds."
        );
        std::process::exit(1);
    }
}
