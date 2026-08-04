// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone correctness oracle for `fp8_gemm_t` (module `w4a16`) — the
//! UNSCALED FP8 (E4M3 weight) × BF16 (activation) prefill GEMM the Qwen3.6
//! SSM qkvz / out_proj path launches via `fp8_gemm_n128`.
//!
//!   C[M,N] = A[M,K] (BF16) · decode_e4m3(B[N,K])^T   (FP32 accumulation)
//!
//! No block scale (fp8_gemm_t takes raw E4M3 bytes; the scale is folded into
//! the weights upstream). This isolates the HIP AMD-WMMA port of the FP8 GEMM
//! from the rest of the forward pass — if the model is incoherent and this
//! FAILS, the WMMA fp8 path is the bug; if it PASSES, look elsewhere.
//!
//! Launch geometry mirrors `fp8_gemm_n128` (gemm_dense.rs):
//!   Grid (ceil(N/128), ceil(M/64), 1), Block (128,1,1).
//!
//! Usage: cargo run --release -p spark-model --example fp8gemm_microtest \
//!          --features cuda,gpu-examples -- [M] [N] [K] [seed]
//! Exit 0 = PASS (cosine >= gate), 1 = FAIL — scriptable.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

// BF16 MMA-reorder noise + E4M3 quantization → loose cosine gate (a real GEMM
// bug — transposed index, dropped K-step, wrong fragment layout — collapses
// cosine far below this).
const COSINE_GATE: f64 = 0.99;

// splitmix64 — reproducible inputs, no rand dependency (PCND: explicit seed).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let round = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    (round >> 16) as u16
}

// Standard OCP E4M3 (1-4-3, bias 7) decode — matches scl_fp8/atlas_e4m3_to_f32
// in the kernel.
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        0.0 // NaN -> 0 (matches kernel)
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

// Encode a small f32 to the nearest E4M3 byte (so test weights are realistic
// decoded magnitudes rather than arbitrary bytes including NaN/large).
fn f32_to_e4m3(v: f32) -> u8 {
    // Brute search over the 256 codes for the nearest decoded value — fine for
    // a microtest (runs once per weight, K*N small).
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for b in 0..=255u8 {
        let d = e4m3_to_f32(b);
        if !d.is_finite() {
            continue;
        }
        let e = (d - v).abs();
        if e < best_err {
            best_err = e;
            best = b;
        }
    }
    best
}

fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let m: usize = args.get(1).map_or(64, |s| s.parse().unwrap());
    let n: usize = args.get(2).map_or(2048, |s| s.parse().unwrap());
    let k: usize = args.get(3).map_or(4096, |s| s.parse().unwrap());
    let seed: u64 = args.get(4).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });

    println!("=== fp8_gemm_t microtest: M={m} N={n} K={k} seed=0x{seed:X} ===");

    let mut rng = Rng(seed);
    // A: BF16 activations, small post-norm magnitudes.
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    // B: E4M3 weight bytes, decoded magnitudes ~[-0.5, 0.5] (typical weights).
    let b_fp8: Vec<u8> = (0..n * k)
        .map(|_| f32_to_e4m3(rng.uniform(-0.5, 0.5)))
        .collect();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_ptr = upload_bytes(gpu, &b_fp8)?;
    let c_ptr = gpu.alloc(m * n * 2)?;

    // A/B probe: time fp8_fp8_gemm_t (FP8 activation, no in-loop convert) vs
    // fp8_gemm_t (BF16 activation). Accuracy-neutral routing candidate.
    if std::env::var_os("ATLAS_PROBE_FP8FP8").is_some() {
        let a_fp8: Vec<u8> = (0..m * k)
            .map(|i| f32_to_e4m3(bf16_bits_to_f32(a_bf16[i])))
            .collect();
        let a8_ptr = upload_bytes(gpu, &a_fp8)?;
        // ldmab: FP8 A, FP8 B, grid (N/128, M/128), block 256 — vs scalar-load fp8_gemm_t.
        {
            let h = gpu.kernel("w4a16_fp8_ldmab", "fp8_fp8_gemm_ldmab")?;
            let launch = |s| {
                KernelLaunch::new(gpu, h)
                    .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 128), 1])
                    .block([256, 1, 1])
                    .arg_ptr(a8_ptr)
                    .arg_ptr(b_ptr)
                    .arg_ptr(c_ptr)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(s)
            };
            for _ in 0..8 {
                launch(stream)?;
            }
            gpu.synchronize(stream)?;
            let iters = 60u32;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch(stream)?;
            }
            gpu.synchronize(stream)?;
            let secs = t0.elapsed().as_secs_f64() / iters as f64;
            let flop = 2.0 * m as f64 * n as f64 * k as f64;
            println!(
                "PROBE fp8_fp8_gemm_ldmab: M={m} N={n} K={k} {:.3} ms {:.1} TFLOP/s",
                secs * 1e3,
                flop / secs / 1e12
            );
            // CORRECTNESS: ldmab vs fp8_fp8_gemm_t (both fp8xfp8, same math) — cosine ~1.0 iff layout correct.
            launch(stream)?;
            gpu.synchronize(stream)?;
            let mut craw = vec![0u8; m * n * 2];
            gpu.copy_d2h(c_ptr, &mut craw)?;
            let cldm: Vec<f32> = craw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let hff = gpu.kernel("w4a16", "fp8_fp8_gemm_t")?;
            KernelLaunch::new(gpu, hff)
                .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 64), 1])
                .block([128, 1, 1])
                .arg_ptr(a8_ptr)
                .arg_ptr(b_ptr)
                .arg_ptr(c_ptr)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)?;
            gpu.synchronize(stream)?;
            gpu.copy_d2h(c_ptr, &mut craw)?;
            let cff: Vec<f32> = craw
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect();
            let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
            for i in 0..cldm.len() {
                let (x, y) = (cldm[i] as f64, cff[i] as f64);
                d += x * y;
                na += x * x;
                nb += y * y;
            }
            let cos = d / (na.sqrt() * nb.sqrt() + 1e-30);
            println!(
                "PROBE ldmab_vs_fp8fp8 cosine={cos:.6}  ({}=PASS)",
                if cos >= 0.99 { "OK" } else { "FAIL" }
            );
        }
        for (name, aptr) in [("fp8_fp8_gemm_t", a8_ptr), ("fp8_gemm_t", a_ptr)] {
            let h = gpu.kernel("w4a16", name)?;
            let launch = |s| {
                KernelLaunch::new(gpu, h)
                    .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 64), 1])
                    .block([128, 1, 1])
                    .arg_ptr(aptr)
                    .arg_ptr(b_ptr)
                    .arg_ptr(c_ptr)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(s)
            };
            for _ in 0..8 {
                launch(stream)?;
            }
            gpu.synchronize(stream)?;
            let iters = 60u32;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch(stream)?;
            }
            gpu.synchronize(stream)?;
            let secs = t0.elapsed().as_secs_f64() / iters as f64;
            let flop = 2.0 * m as f64 * n as f64 * k as f64;
            println!(
                "PROBE {name}: M={m} N={n} K={k} {:.3} ms {:.1} TFLOP/s",
                secs * 1e3,
                flop / secs / 1e12
            );
        }
        for p in [a8_ptr] {
            gpu.free(p).ok();
        }
    }

    let handle = gpu.kernel("w4a16", "fp8_gemm_t")?;
    KernelLaunch::new(gpu, handle)
        .grid([div_ceil(n as u32, 128), div_ceil(m as u32, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(b_ptr)
        .arg_ptr(c_ptr)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let mut c_raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_ptr, &mut c_raw)?;
    let c_gpu: Vec<u16> = c_raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    // CPU reference: C[m,n] = sum_k A[m,k] * e4m3(B[n,k]), FP32 accum.
    let mut c_cpu = vec![0u16; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += bf16_bits_to_f32(a_bf16[row * k + kk]) * e4m3_to_f32(b_fp8[col * k + kk]);
            }
            c_cpu[row * n + col] = f32_to_bf16_bits(acc);
        }
    }

    let (mut dot, mut ng, mut nc, mut max_rel) = (0f64, 0f64, 0f64, 0f64);
    let mut nan_count = 0usize;
    for i in 0..m * n {
        let g = bf16_bits_to_f32(c_gpu[i]) as f64;
        let c = bf16_bits_to_f32(c_cpu[i]) as f64;
        if !g.is_finite() {
            nan_count += 1;
        }
        dot += g * c;
        ng += g * g;
        nc += c * c;
        max_rel = max_rel.max((g - c).abs() / c.abs().max(1e-3));
    }
    let cosine = dot / (ng.sqrt() * nc.sqrt());
    println!("cosine={cosine:.6}  max_rel={max_rel:.3e}  nan/inf_outputs={nan_count}");

    for p in [a_ptr, b_ptr, c_ptr] {
        gpu.free(p).ok();
    }

    if cosine >= COSINE_GATE && nan_count == 0 {
        println!("RESULT: PASS (cosine {cosine:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        println!("RESULT: FAIL (cosine {cosine:.6} < {COSINE_GATE} or {nan_count} NaN/inf)");
        std::process::exit(1);
    }
}
