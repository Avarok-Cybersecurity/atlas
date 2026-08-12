// SPDX-License-Identifier: AGPL-3.0-only
//! Numeric + bandwidth microbench for the native keep-packed ternary Q2_0
//! decode GEMV and its optimization candidates (A=smem, B=vec, C=fused dual).
//!
//! Builds synthetic packed Q2_0 weights `[N, K]` (random codes 0..3, random
//! fp16 per-group scales) and random BF16 activations `[M, K]`, runs each
//! on-device kernel, and compares against an independent CPU oracle that
//! dequantizes each block (`value = (code-1)*d`) and does a dense f32 dot with
//! the bf16 activation.
//!
//! Shapes are FFN-realistic (Ternary-Bonsai hidden=5120, inter=17408, g128):
//!   gate/up : N=17408 K=5120   down : N=5120 K=17408
//! Reports per (variant × shape × M): rel-err, us/call, weight GB/s, speedup.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' \
//!     cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example q2_0_gemv_microtest

use anyhow::Result;
use half::{bf16, f16};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const GROUP: usize = 128;
const ITERS: usize = 300;

struct Lcg(u64);
impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn f(&mut self) -> f32 {
        (((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn code(&mut self) -> u8 {
        (self.next_u64() >> 40) as u8 & 3
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

/// Pack a weight `[n, k]` (codes row-major, scales row-major) into contiguous
/// `block_q2_0` bytes: `[fp16 d][GROUP/4 bytes, 4 codes/byte, low-bits-first]`.
fn pack_weight(codes: &[u8], scales: &[f32], n: usize, k: usize) -> Vec<u8> {
    let block_bytes = 2 + GROUP / 4;
    let gpr = k / GROUP;
    let mut out = Vec::with_capacity(n * gpr * block_bytes);
    for row in 0..n {
        for b in 0..gpr {
            out.extend_from_slice(&f16::from_f32(scales[row * gpr + b]).to_le_bytes());
            let blk = &codes[row * k + b * GROUP..row * k + (b + 1) * GROUP];
            for chunk in blk.chunks(4) {
                let mut byte = 0u8;
                for (t, &c) in chunk.iter().enumerate() {
                    byte |= (c & 3) << (2 * t as u8);
                }
                out.push(byte);
            }
        }
    }
    out
}

/// Independent CPU oracle: `out[m,n] = sum_k a[m,k]*(code-1)*d`.
fn oracle(codes: &[u8], scales: &[f32], a: &[bf16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let gpr = k / GROUP;
    let mut out = vec![0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0f32;
            for kk in 0..k {
                let code = codes[col * k + kk] as i32;
                let d = scales[col * gpr + kk / GROUP];
                acc += a[row * k + kk].to_f32() * (code - 1) as f32 * d;
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let mut denom = 0f32;
    for w in want {
        denom = denom.max(w.abs());
    }
    let denom = denom.max(1e-6);
    let mut max = 0f32;
    for (g, w) in got.iter().zip(want) {
        max = max.max((g - w).abs() / denom);
    }
    max
}

/// One packed weight on device + its host codes/scales for the oracle.
struct Weight {
    d: DevicePtr,
    codes: Vec<u8>,
    scales: Vec<f32>,
    bytes: usize,
    n: usize,
    k: usize,
}
fn make_weight(g: &dyn GpuBackend, rng: &mut Lcg, n: usize, k: usize) -> Result<Weight> {
    let codes: Vec<u8> = (0..n * k).map(|_| rng.code()).collect();
    let scales: Vec<f32> = (0..n * (k / GROUP)).map(|_| rng.r(-0.06, 0.06)).collect();
    let packed = pack_weight(&codes, &scales, n, k);
    let bytes = packed.len();
    let d = up_u8(g, &packed)?;
    Ok(Weight {
        d,
        codes,
        scales,
        bytes,
        n,
        k,
    })
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    grid_div: u32,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    n: u32,
    kk: u32,
    m: Option<u32>,
    shmem: u32,
) -> Result<()> {
    let mut l = KernelLaunch::new(g, k)
        .grid([div_ceil(n, grid_div), 1, 1])
        .block([256, 1, 1])
        .shared_mem(shmem)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(n)
        .arg_u32(kk)
        .arg_u32(GROUP as u32);
    if let Some(mm) = m {
        l = l.arg_u32(mm);
    }
    l.launch(0)
}

struct VariantR {
    err: f32,
    us: f64,
    gbps: f64,
}

/// Run one non-fused variant: correctness (vs oracle) + timing.
#[allow(clippy::too_many_arguments)]
fn run_variant(
    g: &dyn GpuBackend,
    kern: KernelHandle,
    grid_div: u32,
    w: &Weight,
    a_d: DevicePtr,
    a_host: &[bf16],
    c_d: DevicePtr,
    m: usize,
    shmem: u32,
) -> Result<VariantR> {
    let mopt = if m == 1 { None } else { Some(m as u32) };
    // If m>1 but kernel is the M=1 variant, caller passes the batchm handle.
    launch(
        g, kern, grid_div, a_d, w.d, c_d, w.n as u32, w.k as u32, mopt, shmem,
    )?;
    g.synchronize(0)?;
    let got = dn_bf16(g, c_d, m * w.n)?;
    let want = oracle(&w.codes, &w.scales, a_host, m, w.n, w.k);
    let err = rel_err(&got, &want);

    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        launch(
            g, kern, grid_div, a_d, w.d, c_d, w.n as u32, w.k as u32, mopt, shmem,
        )?;
    }
    g.synchronize(0)?;
    let us = t0.elapsed().as_secs_f64() / ITERS as f64 * 1e6;
    let gbps = w.bytes as f64 / (us * 1e-6) / 1e9;
    Ok(VariantR { err, us, gbps })
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    // Kernel handles.
    // Baseline (original whole-block-strided) vs the shipped winner: candidate B
    // (`q2_0_gemv_vec` — vectorized code loads + smem A-stage, 1 warp/row).
    let base1 = g.kernel("q2_0_gemv", "q2_0_gemv")?;
    let base_m = g.kernel("q2_0_gemv", "q2_0_gemv_batchm")?;
    let vec1 = g.kernel("q2_0_gemv_vec", "q2_0_gemv_vec")?;
    let vec_m = g.kernel("q2_0_gemv_vec", "q2_0_gemv_vec_batchm")?;

    let mut rng = Lcg(0x715f_2b30_9f01_0001);
    const MAXM: usize = 8;

    // Shapes: (name, N, K)
    let shapes = [
        ("gate/up", 17408usize, 5120usize),
        ("down", 5120usize, 17408usize),
    ];

    println!("== Q2_0 decode-GEMV microbench (GROUP={GROUP}, {ITERS} iters) ==\n");
    println!(
        "{:<9} {:>3} {:<9} {:>9} {:>9} {:>8} {:>8}",
        "shape", "M", "variant", "rel_err", "us/call", "GB/s", "speedup"
    );

    let mut pass = true;
    for (name, n, k) in shapes {
        let w = make_weight(g, &mut rng, n, k)?;
        let a: Vec<bf16> = (0..MAXM * k)
            .map(|_| bf16::from_f32(rng.r(-1.0, 1.0)))
            .collect();
        let a_d = up_bf16(g, &a)?;
        let c_d = g.alloc(MAXM * n * 2)?;

        for m in [1usize, 4] {
            // baseline (M=1 kernel for m==1, batchm otherwise)
            let (bk, bg) = if m == 1 {
                (base1, 4u32)
            } else {
                (base_m, 4u32)
            };
            let base = run_variant(g, bk, bg, &w, a_d, &a, c_d, m, 0)?;
            let variants: [(&str, KernelHandle, u32, u32); 1] = if m == 1 {
                [("B-vec", vec1, 8, 0)]
            } else {
                [("B-vec", vec_m, 8, 0)]
            };
            let bl = |v: &VariantR| base.us / v.us;
            println!(
                "{name:<9} {m:>3} {:<9} {:>9.5} {:>9.1} {:>8.1} {:>8}",
                "baseline", base.err, base.us, base.gbps, "1.00x"
            );
            if base.err >= 1e-2 {
                pass = false;
            }
            for (vn, vk, gd, sh) in variants {
                let r = run_variant(g, vk, gd, &w, a_d, &a, c_d, m, sh)?;
                println!(
                    "{name:<9} {m:>3} {:<9} {:>9.5} {:>9.1} {:>8.1} {:>7.2}x",
                    vn,
                    r.err,
                    r.us,
                    r.gbps,
                    bl(&r)
                );
                if r.err >= 1e-2 {
                    pass = false;
                }
            }
        }
        let _ = g.free(a_d);
        let _ = g.free(c_d);
        let _ = g.free(w.d);
    }

    if pass {
        println!("\nALL PASS");
        Ok(())
    } else {
        println!("\nSOME FAIL");
        std::process::exit(1);
    }
}
