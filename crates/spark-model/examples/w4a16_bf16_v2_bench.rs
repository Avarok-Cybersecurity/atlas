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

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let v1 = gpu.kernel("w4a16", "w4a16_gemm_t_m128_bf16")?;
    let v2 = gpu.kernel("w4a16", "w4a16_gemm_t_m128_bf16_v2")?;

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

        let t1 = time_kernel(gpu, stream, v1, a_ptr, p_ptr, s_ptr, 0.5, c1, m, n, k, iters)?;
        let t2 = time_kernel(gpu, stream, v2, a_ptr, p_ptr, s_ptr, 0.5, c2, m, n, k, iters)?;

        let tf1 = flops / t1 / 1e12;
        let tf2 = flops / t2 / 1e12;
        println!(
            "{label:<16} {m:>6} {n:>6} {k:>6} | {:>9.4}m {:>9.2} | {:>9.4}m {:>9.2} | {:>6.3}x",
            t1 * 1e3,
            tf1,
            t2 * 1e3,
            tf2,
            t1 / t2,
        );

        for ptr in [a_ptr, p_ptr, s_ptr, c1, c2] {
            let _ = gpu.free(ptr);
        }
    }
    println!("{}", "-".repeat(96));
    Ok(())
}
