// SPDX-License-Identifier: AGPL-3.0-only

//! Losslessness oracle for `w4a16_gemv_sw` (single-warp-per-output decode GEMV)
//! vs the base `w4a16_gemv` (64-thread / 2-warp + smem cross-warp reduce).
//!
//! Unlike the prefill BF16-TC oracle (which only requires cosine ≈ 1.0 because
//! the K-tile reassociation differs), this variant is engineered to be
//! BYTE-IDENTICAL: it reproduces the exact two-warp FP32 reduction tree with two
//! per-lane accumulators, so the PASS bar here is `bit_id == 100%` on every
//! shape. Anything less means the accumulation order diverged → lossy → STOP.
//!
//! Both kernels consume the SAME non-transposed NVFP4 weight layout
//! (B_packed [N, K/2], B_scale [N, K/16]) — no transpose involved.
//!
//! Usage:
//!   cargo run --release -p spark-model --example w4a16_gemv_sw_microtest -- [seed]
//! Exit 0 = all PASS (100% bit-identical), 1 = any FAIL.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

const GROUP_SIZE: usize = 16;

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
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// E4M3 group-scale byte from a small representable set (exact round-trip).
fn e4m3_scale_byte(sel: u32) -> u8 {
    let e = 5 + (sel % 5);
    ((e as u8) << 3) & 0x7F
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

struct Weight {
    packed: Vec<u8>, // [N, K/2]
    scale: Vec<u8>,  // [N, K/16]
    scale2: f32,
}

fn gen_weight(rng: &mut Rng, n: usize, k: usize) -> Weight {
    assert!(k % GROUP_SIZE == 0);
    let half_k = k / 2;
    let num_groups = k / GROUP_SIZE;
    let mut packed = vec![0u8; n * half_k];
    let mut scale = vec![0u8; n * num_groups];
    for i in 0..n {
        for g in 0..num_groups {
            scale[i * num_groups + g] = e4m3_scale_byte(rng.next_u64() as u32);
        }
        for j in 0..half_k {
            let lo = (rng.next_u64() % 16) as u8;
            let hi = (rng.next_u64() % 16) as u8;
            packed[i * half_k + j] = (hi << 4) | lo;
        }
    }
    Weight {
        packed,
        scale,
        scale2: 0.5,
    }
}

fn cosine(a: &[u16], b: &[u16]) -> (f64, usize) {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut bit_eq = 0usize;
    for i in 0..a.len() {
        if a[i] == b[i] {
            bit_eq += 1;
        }
        let x = f32::from_bits((a[i] as u32) << 16) as f64;
        let y = f32::from_bits((b[i] as u32) << 16) as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    (cos, bit_eq)
}

#[allow(clippy::too_many_arguments)]
fn run_shape(
    gpu: &dyn GpuBackend,
    stream: u64,
    base_h: spark_runtime::gpu::KernelHandle,
    sw_h: spark_runtime::gpu::KernelHandle,
    seed: u64,
    n: usize,
    k: usize,
) -> Result<(f64, f64)> {
    let mut rng = Rng(seed ^ ((n as u64) << 16) ^ (k as u64));
    let a_bf16: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a_ptr = upload(gpu, &u16s_to_le(&a_bf16))?;
    let w = gen_weight(&mut rng, n, k);
    let packed = upload(gpu, &w.packed)?;
    let scale = upload(gpu, &w.scale)?;
    let c_base = gpu.alloc(n * 2)?;
    let c_sw = gpu.alloc(n * 2)?;

    // base w4a16_gemv: grid (ceil(N/4),1,1), block (256,1,1)
    KernelLaunch::new(gpu, base_h)
        .grid([n.div_ceil(4) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(packed)
        .arg_ptr(scale)
        .arg_f32(w.scale2)
        .arg_ptr(c_base)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    // w4a16_gemv_sw: grid (ceil(N/8),1,1), block (256,1,1)
    KernelLaunch::new(gpu, sw_h)
        .grid([n.div_ceil(8) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(packed)
        .arg_ptr(scale)
        .arg_f32(w.scale2)
        .arg_ptr(c_sw)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    gpu.synchronize(stream)?;
    let mut rb = vec![0u8; n * 2];
    let mut rs = vec![0u8; n * 2];
    gpu.copy_d2h(c_base, &mut rb)?;
    gpu.copy_d2h(c_sw, &mut rs)?;
    let out_base: Vec<u16> = rb
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let out_sw: Vec<u16> = rs
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let nz = out_base.iter().filter(|&&x| x != 0).count();
    if nz == 0 {
        bail!("dead output (N={n} K={k})");
    }
    for p in [a_ptr, packed, scale, c_base, c_sw] {
        let _ = gpu.free(p);
    }
    let (cos, bit_eq) = cosine(&out_base, &out_sw);
    Ok((cos, bit_eq as f64 / n as f64))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let base_h = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;
    let sw_h = gpu.kernel("w4a16_gemv", "w4a16_gemv_sw")?;

    // Decode GEMV shapes for Qwen3.6-27B (M=1). hidden=5120, intermediate=17408.
    let shapes: &[(&str, usize, usize)] = &[
        ("ffn gate/up", 17408, 5120),
        ("ffn down   ", 5120, 17408),
        ("gdn in_proj", 12384, 5120),
        ("gdn out_prj", 5120, 6144),
        ("attn qkv   ", 7168, 5120),
        ("attn o_proj", 5120, 6144),
        ("N%8!=0 edge", 5124, 5120),
        ("K-tail edge", 4096, 5104),
    ];

    println!(
        "=== w4a16_gemv_sw losslessness microtest (base vs single-warp) seed=0x{seed:X} ===\n"
    );
    println!(
        "{:<12} {:>7} {:>7} | {:>12} {:>9}  result",
        "shape", "N", "K", "cosine", "bit_id%"
    );
    println!("{}", "-".repeat(60));

    let mut all_pass = true;
    for &(label, n, k) in shapes {
        let (cos, bit_id) = run_shape(gpu, stream, base_h, sw_h, seed, n, k)?;
        // Bit-identical gate: every output byte must match.
        let pass = bit_id >= 1.0 - 1e-12;
        all_pass &= pass;
        println!(
            "{label:<12} {n:>7} {k:>7} | {cos:>12.8} {:>8.3}%  {}",
            bit_id * 100.0,
            if pass { "PASS" } else { "FAIL" },
        );
    }
    println!("{}", "-".repeat(60));
    if all_pass {
        println!("RESULT: PASS — w4a16_gemv_sw is BYTE-IDENTICAL to base on all shapes");
        Ok(())
    } else {
        println!("RESULT: FAIL — at least one shape not 100% bit-identical (LOSSY, do not ship)");
        std::process::exit(1);
    }
}
