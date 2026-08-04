// SPDX-License-Identifier: AGPL-3.0-only

//! PHASE-1 MMA PROOF: hand-rolled Sm120 block-scaled FP4 MMA vs the CUTLASS
//! collective.
//!
//! The simplest possible single-tile-looped [M,N,K] GEMM built on a hand-written
//!   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//! (kernel module `fp4_mma_microtest`, kernels `fp4_microtest_pack` +
//! `fp4_microtest_mma`). NO cp.async, NO grouping, NO gate+up fusion.
//!
//! Oracle: spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t — the SAME Sm120
//! block-scaled FP4 math. MUST agree cos >= 0.999 at M=64, N=1024, K=2048.
//!
//! Build (remote GB10):
//!   cargo build --release -p spark-model --example fp4_mma_microproof \
//!     --no-default-features --features "cuda gpu-examples"
//! Run: target/release/examples/fp4_mma_microproof

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

// ───────────────────────── deterministic PRNG ─────────────────────────
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
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn read_bf16(gpu: &dyn GpuBackend, ptr: DevicePtr, m: usize, n: usize) -> Result<Vec<u16>> {
    let mut raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(ptr, &mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn cosine_u16(a: &[u16], b: &[u16]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let x = bf16_bits_to_f32(a[i]) as f64;
        let y = bf16_bits_to_f32(b[i]) as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}

fn main() -> Result<()> {
    let (m, n, k) = (64usize, 1024usize, 2048usize);
    let seed = 0x_5151_A7A7u64;
    println!("=== Phase-1 hand-rolled Sm120 FP4 MMA microproof ===");
    println!("M={m} N={n} K={k}; oracle = nvfp4_gemm_bf16_act_weight_t (same Sm120 FP4 math)");

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let b_bf16: Vec<u16> = (0..n * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-0.5, 0.5)))
        .collect();

    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_ptr = upload_bytes(gpu, &u16s_to_le(&b_bf16))?;

    // ── Oracle: CUTLASS collective. Pack B once into [K/2,N]/[K/16,N]. ──
    let packed_len = (k / 2) * n;
    let scale_len = (k / 16) * n;
    let packed_ptr = gpu.alloc(packed_len)?;
    let scale_ptr = gpu.alloc(scale_len)?;
    spark_runtime::cutlass::pack_bf16_weight_to_nvfp4_t(
        b_ptr.0,
        packed_ptr.0,
        scale_ptr.0,
        n as u32,
        k as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;

    let out_oracle = gpu.alloc(m * n * 2)?;
    spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t(
        a_ptr.0,
        packed_ptr.0,
        scale_ptr.0,
        1.0,
        out_oracle.0,
        m as u32,
        n as u32,
        k as u32,
        stream,
    )?;
    gpu.synchronize(stream)?;
    let c_oracle = read_bf16(gpu, out_oracle, m, n)?;

    // ── Hand-rolled MMA path ──
    // Pre-pass pack A[M][K] and B[N][K] into natural-layout packed+scales.
    let pack_handle = gpu.kernel("fp4_mma_microtest", "fp4_microtest_pack")?;
    let mma_handle = gpu.kernel("fp4_mma_microtest", "fp4_microtest_mma")?;

    let a_packed = gpu.alloc((k / 2) * m)?;
    let a_scales = gpu.alloc((k / 16) * m)?;
    let b_packed = gpu.alloc((k / 2) * n)?;
    let b_scales = gpu.alloc((k / 16) * n)?;

    // pack A: grid (rows, groups/threads), block 128 over groups
    let groups = (k / 16) as u32;
    let pack_a = || -> Result<()> {
        KernelLaunch::new(gpu, pack_handle)
            .grid([m as u32, div_ceil(groups, 128), 1])
            .block([128, 1, 1])
            .arg_ptr(a_ptr)
            .arg_ptr(a_packed)
            .arg_ptr(a_scales)
            .arg_u32(m as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        Ok(())
    };
    let pack_b = || -> Result<()> {
        KernelLaunch::new(gpu, pack_handle)
            .grid([n as u32, div_ceil(groups, 128), 1])
            .block([128, 1, 1])
            .arg_ptr(b_ptr)
            .arg_ptr(b_packed)
            .arg_ptr(b_scales)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        Ok(())
    };
    pack_a()?;
    pack_b()?;
    gpu.synchronize(stream)?;

    let out_mma = gpu.alloc(m * n * 2)?;
    let mma_launch = || -> Result<()> {
        KernelLaunch::new(gpu, mma_handle)
            .grid([div_ceil(n as u32, 8), div_ceil(m as u32, 16), 1])
            .block([32, 1, 1])
            .arg_ptr(a_packed)
            .arg_ptr(a_scales)
            .arg_ptr(b_packed)
            .arg_ptr(b_scales)
            .arg_ptr(out_mma)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        Ok(())
    };
    mma_launch()?;
    gpu.synchronize(stream)?;
    let c_mma = read_bf16(gpu, out_mma, m, n)?;

    let cos = cosine_u16(&c_mma, &c_oracle);
    println!();
    println!("hand-rolled MMA  vs  CUTLASS collective  cosine = {cos:.6}");
    // sample a few values
    println!(
        "sample [0][0..4]: mma={:?} oracle={:?}",
        &c_mma[0..4]
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect::<Vec<_>>(),
        &c_oracle[0..4]
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect::<Vec<_>>()
    );
    println!(
        "sample [1][0..4]: mma={:?} oracle={:?}",
        &c_mma[n..n + 4]
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect::<Vec<_>>(),
        &c_oracle[n..n + 4]
            .iter()
            .map(|&b| bf16_bits_to_f32(b))
            .collect::<Vec<_>>()
    );

    for p in [
        a_ptr, b_ptr, packed_ptr, scale_ptr, out_oracle, a_packed, a_scales, b_packed, b_scales,
        out_mma,
    ] {
        gpu.free(p).ok();
    }

    if cos >= 0.999 {
        println!("\nPASS: cos {cos:.6} >= 0.999");
        Ok(())
    } else {
        bail!("FAIL: cos {cos:.6} < 0.999");
    }
}
