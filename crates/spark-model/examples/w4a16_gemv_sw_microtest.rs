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
    assert!(k.is_multiple_of(GROUP_SIZE));
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

struct Cmp {
    cos: f64,
    bit_id: f64,
    mismatches: Vec<(usize, u16, u16)>,
}

fn compare(a: &[u16], b: &[u16]) -> Cmp {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let mut bit_eq = 0usize;
    let mut mismatches = Vec::new();
    for i in 0..a.len() {
        if a[i] == b[i] {
            bit_eq += 1;
        } else if mismatches.len() < 4 {
            mismatches.push((i, a[i], b[i]));
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
    Cmp {
        cos,
        bit_id: bit_eq as f64 / a.len() as f64,
        mismatches,
    }
}

fn dump_fail(label: &str, cmp: &Cmp) {
    for &(i, base, sw) in &cmp.mismatches {
        eprintln!(
            "  {label} mismatch[{i}]: base=0x{base:04x} sw=0x{sw:04x} \
             ({:.6} vs {:.6})",
            f32::from_bits((base as u32) << 16),
            f32::from_bits((sw as u32) << 16),
        );
    }
}

fn download_u16(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
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
) -> Result<Cmp> {
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
    let out_base = download_u16(gpu, c_base, n)?;
    let out_sw = download_u16(gpu, c_sw, n)?;

    let nz = out_base.iter().filter(|&&x| x != 0).count();
    if nz == 0 {
        bail!("dead output (N={n} K={k})");
    }
    for p in [a_ptr, packed, scale, c_base, c_sw] {
        let _ = gpu.free(p);
    }
    Ok(compare(&out_base, &out_sw))
}

#[allow(clippy::too_many_arguments)]
fn run_dual(
    gpu: &dyn GpuBackend,
    stream: u64,
    base_h: spark_runtime::gpu::KernelHandle,
    sw_h: spark_runtime::gpu::KernelHandle,
    seed: u64,
    n: usize,
    k: usize,
) -> Result<(Cmp, Cmp)> {
    let mut rng = Rng(seed ^ 0xD2A1_0000 ^ ((n as u64) << 16) ^ (k as u64));
    let a_bf16: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a_ptr = upload(gpu, &u16s_to_le(&a_bf16))?;
    let w1 = gen_weight(&mut rng, n, k);
    let w2 = gen_weight(&mut rng, n, k);
    let p1 = upload(gpu, &w1.packed)?;
    let s1 = upload(gpu, &w1.scale)?;
    let p2 = upload(gpu, &w2.packed)?;
    let s2 = upload(gpu, &w2.scale)?;
    let c1b = gpu.alloc(n * 2)?;
    let c2b = gpu.alloc(n * 2)?;
    let c1s = gpu.alloc(n * 2)?;
    let c2s = gpu.alloc(n * 2)?;

    KernelLaunch::new(gpu, base_h)
        .grid([n.div_ceil(4) as u32, 1, 2])
        .block([256, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(p1)
        .arg_ptr(s1)
        .arg_f32(w1.scale2)
        .arg_ptr(c1b)
        .arg_ptr(p2)
        .arg_ptr(s2)
        .arg_f32(w2.scale2)
        .arg_ptr(c2b)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, sw_h)
        .grid([n.div_ceil(8) as u32, 1, 2])
        .block([256, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(p1)
        .arg_ptr(s1)
        .arg_f32(w1.scale2)
        .arg_ptr(c1s)
        .arg_ptr(p2)
        .arg_ptr(s2)
        .arg_f32(w2.scale2)
        .arg_ptr(c2s)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    gpu.synchronize(stream)?;
    let b1 = download_u16(gpu, c1b, n)?;
    let s1o = download_u16(gpu, c1s, n)?;
    let b2 = download_u16(gpu, c2b, n)?;
    let s2o = download_u16(gpu, c2s, n)?;
    if b1.iter().all(|&x| x == 0) && b2.iter().all(|&x| x == 0) {
        bail!("dead dual output (N={n} K={k})");
    }
    let g1 = compare(&b1, &s1o);
    let g2 = compare(&b2, &s2o);
    for p in [a_ptr, p1, s1, p2, s2, c1b, c2b, c1s, c2s] {
        let _ = gpu.free(p);
    }
    Ok((g1, g2))
}

#[allow(clippy::too_many_arguments)]
fn run_silu(
    gpu: &dyn GpuBackend,
    stream: u64,
    base_h: spark_runtime::gpu::KernelHandle,
    sw_h: spark_runtime::gpu::KernelHandle,
    seed: u64,
    n: usize,
    k: usize,
) -> Result<Cmp> {
    let mut rng = Rng(seed ^ 0x51_0000 ^ ((n as u64) << 16) ^ (k as u64));
    let gate: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let up: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let g_ptr = upload(gpu, &u16s_to_le(&gate))?;
    let u_ptr = upload(gpu, &u16s_to_le(&up))?;
    let w = gen_weight(&mut rng, n, k);
    let packed = upload(gpu, &w.packed)?;
    let scale = upload(gpu, &w.scale)?;
    let c_base = gpu.alloc(n * 2)?;
    let c_sw = gpu.alloc(n * 2)?;

    KernelLaunch::new(gpu, base_h)
        .grid([n.div_ceil(4) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(g_ptr)
        .arg_ptr(u_ptr)
        .arg_ptr(packed)
        .arg_ptr(scale)
        .arg_f32(w.scale2)
        .arg_ptr(c_base)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, sw_h)
        .grid([n.div_ceil(8) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(g_ptr)
        .arg_ptr(u_ptr)
        .arg_ptr(packed)
        .arg_ptr(scale)
        .arg_f32(w.scale2)
        .arg_ptr(c_sw)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    gpu.synchronize(stream)?;
    let out_base = download_u16(gpu, c_base, n)?;
    let out_sw = download_u16(gpu, c_sw, n)?;
    if out_base.iter().all(|&x| x == 0) {
        bail!("dead silu output (N={n} K={k})");
    }
    for p in [g_ptr, u_ptr, packed, scale, c_base, c_sw] {
        let _ = gpu.free(p);
    }
    Ok(compare(&out_base, &out_sw))
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
    let dual_h = gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?;
    let dual_sw_h = gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual_sw")?;
    let silu_h = gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input")?;
    let silu_sw_h = gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_silu_input_sw")?;

    // Decode GEMV shapes for Qwen3.6-27B (M=1). hidden=5120, intermediate=17408.
    let shapes: &[(&str, usize, usize)] = &[
        ("ffn gate/up", 17408, 5120),
        ("ffn down   ", 5120, 17408),
        ("gdn in_proj", 12384, 5120),
        ("gdn out_prj", 5120, 6144),
        ("attn qkv   ", 7168, 5120),
        ("attn o_proj", 5120, 6144),
        ("N%8!=0 edge", 5124, 5120),
        ("N%4!=0 tail", 5122, 5120),
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
        let cmp = run_shape(gpu, stream, base_h, sw_h, seed, n, k)?;
        let pass = cmp.bit_id >= 1.0 - 1e-12;
        all_pass &= pass;
        println!(
            "{label:<12} {n:>7} {k:>7} | {:>12.8} {:>8.3}%  {}",
            cmp.cos,
            cmp.bit_id * 100.0,
            if pass { "PASS" } else { "FAIL" },
        );
        if !pass {
            dump_fail(label, &cmp);
        }
    }

    println!("\n=== w4a16_gemv_dual_sw (FFN gate+up) ===");
    for &(label, n, k) in shapes {
        let (g1, g2) = run_dual(gpu, stream, dual_h, dual_sw_h, seed, n, k)?;
        let pass = g1.bit_id >= 1.0 - 1e-12 && g2.bit_id >= 1.0 - 1e-12;
        all_pass &= pass;
        println!(
            "{label:<12} {n:>7} {k:>7} | gate {:>7.3}% up {:>7.3}%  {}",
            g1.bit_id * 100.0,
            g2.bit_id * 100.0,
            if pass { "PASS" } else { "FAIL" },
        );
        if !pass {
            dump_fail(&format!("{label} gate"), &g1);
            dump_fail(&format!("{label} up"), &g2);
        }
    }

    println!("\n=== w4a16_gemv_silu_input_sw (FFN down) ===");
    for &(label, n, k) in &[
        ("ffn down   ", 5120usize, 17408usize),
        ("N%8!=0 edge", 5124, 5120),
        ("K-tail edge", 4096, 5104),
    ] {
        let cmp = run_silu(gpu, stream, silu_h, silu_sw_h, seed, n, k)?;
        let pass = cmp.bit_id >= 1.0 - 1e-12;
        all_pass &= pass;
        println!(
            "{label:<12} {n:>7} {k:>7} | {:>12.8} {:>8.3}%  {}",
            cmp.cos,
            cmp.bit_id * 100.0,
            if pass { "PASS" } else { "FAIL" },
        );
        if !pass {
            dump_fail(label, &cmp);
        }
    }

    println!("{}", "-".repeat(60));
    if all_pass {
        println!("RESULT: PASS — SW GEMV is BYTE-IDENTICAL to base on gemv/dual/silu");
        Ok(())
    } else {
        println!("RESULT: FAIL — at least one shape not 100% bit-identical (LOSSY, do not ship)");
        std::process::exit(1);
    }
}
