// SPDX-License-Identifier: AGPL-3.0-only

//! Wall-time microbenchmark: `w4a16_gemm_t_m128_bf16` (v1) vs
//! `w4a16_gemm_t_m128_bf16_v2` (pipelined) on the Qwen3.6-27B dense-FFN prefill
//! GEMM shapes. Times many back-to-back launches between two stream syncs so
//! launch overhead is amortized; reports per-launch ms and TFLOP/s.
//!
//! Usage: cargo run --release -p spark-model --example w4a16_bf16_v2_bench

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

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
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

#[allow(clippy::too_many_arguments)]
fn time_kernel(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    // Warmup
    for _ in 0..3 {
        KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a)
            .arg_ptr(packed)
            .arg_ptr(scale)
            .arg_f32(scale2)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
    }
    gpu.synchronize(stream)?;

    let t0 = Instant::now();
    for _ in 0..iters {
        KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a)
            .arg_ptr(packed)
            .arg_ptr(scale)
            .arg_f32(scale2)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

// fp8_gemm_t_m128: BF16 A x FP8 B (E4M3), m16n8k32. Args: (A, B_fp8, C, M, N, K).
fn time_kernel_fp8(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    b_fp8: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let launch = |h: KernelHandle| -> Result<()> {
        KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a)
            .arg_ptr(b_fp8)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch(h)?;
    }
    gpu.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        launch(h)?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

// M_TILE=64 kernels (w4a16_gemm_t / _k64): grid.y = M/64. Same arg list as time_kernel.
fn time_kernel_m64(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    a: DevicePtr,
    packed: DevicePtr,
    scale: DevicePtr,
    scale2: f32,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> Result<f64> {
    let launch = || -> Result<()> {
        KernelLaunch::new(gpu, h)
            .grid([n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a)
            .arg_ptr(packed)
            .arg_ptr(scale)
            .arg_f32(scale2)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch()?;
    }
    gpu.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        launch()?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let v1 = gpu.kernel("w4a16", "w4a16_gemm_t_m128_bf16")?;
    let v2 = gpu.kernel("w4a16", "w4a16_gemm_t_m128_bf16_v2")?;
    let fp8 = gpu.kernel("w4a16", "fp8_gemm_t_m128")?;
    let fp8fp8 = gpu.kernel("w4a16", "fp8_fp8_gemm_t_m128")?;
    // K32 vs K64 lossless NVFP4->BF16, M_TILE=64 tiling (grid.y = M/64): bubble test
    let gt_k32 = gpu.kernel("w4a16", "w4a16_gemm_t_m64_bf16")?; // NEW: bit-identical M64
    let gt_k64 = gpu.kernel("w4a16", "w4a16_gemm_t_k64")?;
    let dbf16 = gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?; // pure bf16, 8-warp, no dequant
    let dtc = gpu.kernel("gemm_tc", "dense_gemm_tc")?; // pure bf16, LDMATRIX fragment loads

    // Qwen3.6-27B dense FFN: H=5120, inter=17408. gate/up: N=17408,K=5120.
    // down: N=5120,K=17408. Prefill M = 1024 and 4096 (chunk sizes).
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("gate/up M=1024", 1024, 17408, 5120),
        ("down    M=1024", 1024, 5120, 17408),
        ("gate/up M=4096", 4096, 17408, 5120),
        ("down    M=4096", 4096, 5120, 17408),
    ];

    println!("=== w4a16 BF16 prefill GEMM bench (v1 vs v2) ===\n");
    println!(
        "{:<16} {:>6} {:>6} {:>6} | {:>10} {:>9} | {:>10} {:>9} | {:>7}",
        "shape", "M", "N", "K", "v1 ms", "v1 TFLOP", "v2 ms", "v2 TFLOP", "speedup"
    );
    println!("{}", "-".repeat(96));

    let mut rng = Rng(0x1234);
    for &(label, m, n, k) in shapes {
        let half_k = k / 2;
        let num_groups = k / GROUP_SIZE;
        // Transposed layout [K/2, N] packed, [K/16, N] scale.
        let mut packed = vec![0u8; half_k * n];
        let mut scale = vec![0u8; num_groups * n];
        for b in packed.iter_mut() {
            *b = rng.next_u64() as u8;
        }
        for s in scale.iter_mut() {
            // benign e4m3 scale byte (exp 5..9, mant 0)
            *s = (((5 + (rng.next_u64() % 5)) as u8) << 3) & 0x7F;
        }
        let a: Vec<u8> = (0..m * k * 2).map(|_| rng.next_u64() as u8).collect();

        let a_ptr = upload(gpu, &a)?;
        let p_ptr = upload(gpu, &packed)?;
        let s_ptr = upload(gpu, &scale)?;
        let c1 = gpu.alloc(m * n * 2)?;
        let c2 = gpu.alloc(m * n * 2)?;

        // FLOPs = 2*M*N*K
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let iters = if m >= 4096 { 30 } else { 60 };

        // FP8 B: N*K E4M3 bytes (random — speed only, not correctness).
        let b_fp8: Vec<u8> = (0..n * k).map(|_| rng.next_u64() as u8).collect();
        let bf8_ptr = upload(gpu, &b_fp8)?;
        let c3 = gpu.alloc(m * n * 2)?;

        let t1 = time_kernel(
            gpu, stream, v1, a_ptr, p_ptr, s_ptr, 0.5, c1, m, n, k, iters,
        )?;
        let t2 = time_kernel(
            gpu, stream, v2, a_ptr, p_ptr, s_ptr, 0.5, c2, m, n, k, iters,
        )?;
        let t3 = time_kernel_fp8(gpu, stream, fp8, a_ptr, bf8_ptr, c3, m, n, k, iters)?;
        // fp8_fp8: both operands FP8 -> true m16n8k32.e4m3 MMA (2x candidate)
        let a_fp8: Vec<u8> = (0..m * k).map(|_| rng.next_u64() as u8).collect();
        let af8_ptr = upload(gpu, &a_fp8)?;
        let c4 = gpu.alloc(m * n * 2)?;
        let t4 = time_kernel_fp8(gpu, stream, fp8fp8, af8_ptr, bf8_ptr, c4, m, n, k, iters)?;

        // K32 vs K64 lossless (M64 tiling) — the bubble-reduction test
        let c5 = gpu.alloc(m * n * 2)?;
        let c6 = gpu.alloc(m * n * 2)?;
        let t5 = time_kernel_m64(
            gpu, stream, gt_k32, a_ptr, p_ptr, s_ptr, 0.5, c5, m, n, k, iters,
        )?;
        let t6 = time_kernel_m64(
            gpu, stream, gt_k64, a_ptr, p_ptr, s_ptr, 0.5, c6, m, n, k, iters,
        )?;

        // dense_gemm_bf16_pipelined: pure bf16 A x bf16 B (predequanted weights), no in-kernel dequant
        let b_bf16: Vec<u8> = (0..n * k * 2).map(|_| rng.next_u64() as u8).collect();
        let bbf_ptr = upload(gpu, &b_bf16)?;
        let c7 = gpu.alloc(m * n * 2)?;
        let launch_dense = |it: usize| -> Result<f64> {
            let go = || -> Result<()> {
                KernelLaunch::new(gpu, dbf16)
                    .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(a_ptr)
                    .arg_ptr(bbf_ptr)
                    .arg_ptr(c7)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(stream)
            };
            for _ in 0..3 {
                go()?;
            }
            gpu.synchronize(stream)?;
            let t0 = Instant::now();
            for _ in 0..it {
                go()?;
            }
            gpu.synchronize(stream)?;
            Ok(t0.elapsed().as_secs_f64() / it as f64)
        };
        let t7 = launch_dense(iters)?;
        // dense_gemm_tc: pure bf16 with LDMATRIX fragment loads — tests the smem-bandwidth lever
        let c8 = gpu.alloc(m * n * 2)?;
        let launch_tc = |it: usize| -> Result<f64> {
            let go = || -> Result<()> {
                KernelLaunch::new(gpu, dtc)
                    .grid([n.div_ceil(64) as u32, m.div_ceil(16) as u32, 1])
                    .block([128, 1, 1])
                    .arg_ptr(a_ptr)
                    .arg_ptr(bbf_ptr)
                    .arg_ptr(c8)
                    .arg_u32(m as u32)
                    .arg_u32(n as u32)
                    .arg_u32(k as u32)
                    .launch(stream)
            };
            for _ in 0..3 {
                go()?;
            }
            gpu.synchronize(stream)?;
            let t0 = Instant::now();
            for _ in 0..it {
                go()?;
            }
            gpu.synchronize(stream)?;
            Ok(t0.elapsed().as_secs_f64() / it as f64)
        };
        let t8 = launch_tc(iters)?;

        let tf2 = flops / t2 / 1e12;
        let tf7 = flops / t7 / 1e12;
        let tf8 = flops / t8 / 1e12;
        println!(
            "{label:<16} {m:>6} {n:>6} {k:>6} | bf16v2(deq) {:>6.2} | denseBF16 {:>6.2} | denseTC(ldmatrix) {:>6.2} | {:>5.3}x v2",
            tf2,
            tf7,
            tf8,
            t2 / t8,
        );
        let _ = (t1, t3, t4, t5, t6);
        for ptr in [c5, c6, bbf_ptr, c7, c8] {
            let _ = gpu.free(ptr);
        }

        for ptr in [a_ptr, p_ptr, s_ptr, c1, c2, bf8_ptr, c3, af8_ptr, c4] {
            let _ = gpu.free(ptr);
        }
    }
    println!("{}", "-".repeat(96));
    Ok(())
}
