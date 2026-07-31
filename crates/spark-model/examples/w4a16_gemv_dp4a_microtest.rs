// SPDX-License-Identifier: AGPL-3.0-only

//! Correctness + kernel-only-throughput microtest for the strix-hip W4A8 integer-DP4A
//! decode GEMV (`w4a16_gemv_dp4a` + `quantize_act_int8_g16`) against the EXISTING float
//! E2M1-LUT `w4a16_gemv` it is meant to replace.
//!
//! Oracle = the production float `w4a16_gemv` GPU kernel (same inputs). The DP4A path
//! is NOT bit-identical (it quantizes activations to int8, d=amax/16-group), so the gate
//! is cosine similarity, not byte-equality. A correct port matches the float path to
//! >= ~0.999 cosine; the int8-act quant is the only new error term.
//!
//! Usage:
//!   cargo run --release -p spark-model --example w4a16_gemv_dp4a_microtest \
//!       -- [N] [K] [seed]
//! Defaults: N=2048 K=4096 seed=0x51A7. Exit 0 = PASS (cosine >= gate), 1 = FAIL.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

const GROUP_SIZE: usize = 16;
const COSINE_GATE: f64 = 0.999;

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

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
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
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).map_or(2048, |s| s.parse().unwrap());
    let k: usize = args.get(2).map_or(4096, |s| s.parse().unwrap());
    let seed: u64 = args.get(3).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    if n % 4 != 0 || k % GROUP_SIZE != 0 {
        bail!("N must be %4 and K must be %{GROUP_SIZE}");
    }
    println!("=== w4a16_gemv_dp4a microtest: N={n} K={k} seed=0x{seed:X} ===");

    // ── inputs ──
    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    // 4-bit NVFP4 weights: random nibbles 0..15, packed 2/byte → [N, K/2].
    let b_packed: Vec<u8> = (0..n * k / 2)
        .map(|_| (rng.next_u64() & 0xFF) as u8)
        .collect();
    // per-16 group E4M3 scales, positive, ~0.5..2.0 (exp field 6..8, bias 7).
    let num_groups = k / GROUP_SIZE;
    let b_scale: Vec<u8> = (0..n * num_groups)
        .map(|_| {
            let e = 6 + (rng.next_u64() % 3) as u8; // 6,7,8 -> 2^-1..2^1
            let m = (rng.next_u64() % 8) as u8;
            (e << 3) | m
        })
        .collect();
    let scale2: f32 = 1.0;

    // ── GPU ──
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_ptr = upload_bytes(gpu, &b_packed)?;
    let s_ptr = upload_bytes(gpu, &b_scale)?;
    let c_ref = gpu.alloc(n * 2)?; // bf16 out
    let c_dp4 = gpu.alloc(n * 2)?;
    let aq_ptr = gpu.alloc(k)?; // int8 activations
    let as_ptr = gpu.alloc(num_groups * 4)?; // f32 per-group act scales

    // float oracle: w4a16_gemv(A, B, scale, scale2, C, N, K)
    let kf = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;
    let launch_float = |sync: bool| -> Result<()> {
        KernelLaunch::new(gpu, kf.clone())
            .grid([(n as u32).div_ceil(4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(b_ptr)
            .arg_ptr(s_ptr)
            .arg_f32(scale2)
            .arg_ptr(c_ref)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        if sync {
            gpu.synchronize(stream)?;
        }
        Ok(())
    };
    // dp4a: quantize_act_int8_g16(A, a_q, a_scale, K) then w4a16_gemv_dp4a(...)
    let kq = gpu.kernel("w4a16_gemv_dp4a", "quantize_act_int8_g16")?;
    let kd = gpu.kernel("w4a16_gemv_dp4a", "w4a16_gemv_dp4a")?;
    let launch_dp4a = |sync: bool| -> Result<()> {
        KernelLaunch::new(gpu, kq.clone())
            .grid([num_groups as u32, 1, 1])
            .block([GROUP_SIZE as u32, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(aq_ptr)
            .arg_ptr(as_ptr)
            .arg_u32(k as u32)
            .launch(stream)?;
        KernelLaunch::new(gpu, kd.clone())
            .grid([(n as u32).div_ceil(4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(aq_ptr)
            .arg_ptr(as_ptr)
            .arg_ptr(b_ptr)
            .arg_ptr(s_ptr)
            .arg_f32(scale2)
            .arg_ptr(c_dp4)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        if sync {
            gpu.synchronize(stream)?;
        }
        Ok(())
    };

    launch_float(true)?;
    launch_dp4a(true)?;

    let read = |ptr: DevicePtr| -> Result<Vec<f32>> {
        let mut raw = vec![0u8; n * 2];
        gpu.copy_d2h(ptr, &mut raw)?;
        Ok(raw
            .chunks_exact(2)
            .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    };
    let cref = read(c_ref)?;
    let cdp4 = read(c_dp4)?;

    let (mut dot, mut nr, mut nd, mut max_rel, mut sum_rel) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..n {
        let (r, d) = (cref[i] as f64, cdp4[i] as f64);
        dot += r * d;
        nr += r * r;
        nd += d * d;
        let rel = (r - d).abs() / r.abs().max(1e-3);
        max_rel = max_rel.max(rel);
        sum_rel += rel;
    }
    let cosine = dot / (nr.sqrt() * nd.sqrt());
    println!(
        "cosine={cosine:.6}  mean_rel={:.4}  max_rel={:.4}",
        sum_rel / n as f64,
        max_rel
    );

    // kernel timing (wall, relative A/B)
    let time = |f: &dyn Fn(bool) -> Result<()>| -> Result<f64> {
        for _ in 0..10 {
            f(true)?;
        }
        let t0 = Instant::now();
        for _ in 0..100 {
            f(true)?;
        }
        Ok(t0.elapsed().as_secs_f64() / 100.0 * 1e6)
    };
    // GEMV-only (activations pre-quantized once): the amortized per-GEMV cost the real
    // decode path sees, since quant is hoisted to once-per-layer across ~5 GEMVs.
    let launch_dp4a_gemv = |sync: bool| -> Result<()> {
        KernelLaunch::new(gpu, kd.clone())
            .grid([(n as u32).div_ceil(4), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(aq_ptr)
            .arg_ptr(as_ptr)
            .arg_ptr(b_ptr)
            .arg_ptr(s_ptr)
            .arg_f32(scale2)
            .arg_ptr(c_dp4)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        if sync {
            gpu.synchronize(stream)?;
        }
        Ok(())
    };
    let us_float = time(&launch_float)?;
    let us_dp4a = time(&launch_dp4a)?;
    let us_gemv = time(&launch_dp4a_gemv)?;
    println!(
        "float w4a16_gemv: {us_float:.1} us | dp4a quant+gemv: {us_dp4a:.1} us ({:.2}x) | dp4a GEMV-only: {us_gemv:.1} us ({:.2}x amortized)",
        us_float / us_dp4a,
        us_float / us_gemv
    );

    if cosine >= COSINE_GATE {
        println!("PASS (cosine {cosine:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        bail!("FAIL cosine {cosine:.6} < {COSINE_GATE}");
    }
}
