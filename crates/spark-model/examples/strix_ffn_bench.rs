// SPDX-License-Identifier: AGPL-3.0-only

//! Strix (gfx1151) dense-FFN prefill GEMM microbench.
//!
//! Times ONLY the NVFP4 kernels that actually exist in the strix-hip w4a16
//! module (the gb10 bench references *_bf16 / *_v2 / dense_gemm_* kernels that
//! are absent here and would hard-fail at `gpu.kernel()`):
//!   * w4a16_gemm_t_m128 — current prefill default. M_TILE2=128, LDS 34432,
//!     VGPR 256 (spills), occupancy 3 waves/SIMD.
//!   * w4a16_gemm_t      — same NVFP4 math at M_TILE=64. LDS 24192, VGPR 166,
//!     occupancy 9, no spill.
//!   * w4a16_gemm_t_k64  — M_TILE=64 with K_STEP=64 (halves the K-loop
//!     iteration/barrier count). LDS 45248, occupancy 9.
//!
//! Also cross-checks m128 vs M64 outputs byte-for-byte: they must be identical
//! (same per-output-element K accumulation order), which doubles as a check
//! that the launch grids are right.
//!
//! usage: cargo run --release -p spark-model --example strix_ffn_bench

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

/// m_tile selects grid.y granularity: 128 for the m128 kernel, 64 for the
/// M_TILE=64 kernels. All three share the same (A, Bp, Bs, scale2, C, M,N,K)
/// signature and 128-thread block.
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &dyn GpuBackend, stream: u64, h: KernelHandle, m_tile: usize,
    a: DevicePtr, packed: DevicePtr, scale: DevicePtr, scale2: f32, c: DevicePtr,
    m: usize, n: usize, k: usize,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([n.div_ceil(128) as u32, m.div_ceil(m_tile) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(a).arg_ptr(packed).arg_ptr(scale).arg_f32(scale2).arg_ptr(c)
        .arg_u32(m as u32).arg_u32(n as u32).arg_u32(k as u32)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn time_it(
    gpu: &dyn GpuBackend, stream: u64, h: KernelHandle, m_tile: usize,
    a: DevicePtr, packed: DevicePtr, scale: DevicePtr, c: DevicePtr,
    m: usize, n: usize, k: usize, iters: usize,
) -> Result<f64> {
    for _ in 0..3 {
        run(gpu, stream, h, m_tile, a, packed, scale, 0.5, c, m, n, k)?;
    }
    gpu.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(gpu, stream, h, m_tile, a, packed, scale, 0.5, c, m, n, k)?;
    }
    gpu.synchronize(stream)?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let m128 = gpu.kernel("w4a16", "w4a16_gemm_t_m128")?;
    let m64 = gpu.kernel("w4a16", "w4a16_gemm_t")?;
    let k64 = gpu.kernel("w4a16", "w4a16_gemm_t_k64")?;

    // Qwen3.6-27B dense FFN: H=5120, inter=17408.
    // gate/up: N=17408 K=5120 (2 of the 3 GEMMs). down: N=5120 K=17408.
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("down    M=17",   17, 5120, 17408),
        ("down    M=64",   64, 5120, 17408),
        ("gate/up M=17",   17, 17408, 5120),
        ("gate/up M=64",   64, 17408, 5120),
        ("gate/up M=512",  512, 17408, 5120),
        ("down    M=512",  512, 5120, 17408),
        ("gate/up M=1024", 1024, 17408, 5120),
        ("down    M=1024", 1024, 5120, 17408),
        ("gate/up M=2048", 2048, 17408, 5120),
        ("down    M=2048", 2048, 5120, 17408),
    ];

    println!("=== strix gfx1151 dense-FFN prefill GEMM (NVFP4, TFLOP/s) ===\n");
    println!("{:<16} {:>6} {:>6} {:>6} | {:>9} {:>9} {:>9} | {:>8} {:>8}",
             "shape", "M", "N", "K", "m128", "M64", "k64", "M64/m128", "k64/m128");
    println!("{}", "-".repeat(92));

    let mut rng = Rng(0x5721);
    let mut mismatch = 0usize;

    for &(label, m, n, k) in shapes {
        let mut packed = vec![0u8; (k / 2) * n];
        let mut scale = vec![0u8; (k / GROUP_SIZE) * n];
        for b in packed.iter_mut() { *b = rng.next_u64() as u8; }
        for s in scale.iter_mut() { *s = (((5 + (rng.next_u64() % 5)) as u8) << 3) & 0x7F; }
        let a: Vec<u8> = (0..m * k * 2).map(|_| rng.next_u64() as u8).collect();

        let a_ptr = upload(gpu, &a)?;
        let p_ptr = upload(gpu, &packed)?;
        let s_ptr = upload(gpu, &scale)?;
        let c_a = gpu.alloc(m * n * 2)?;
        let c_b = gpu.alloc(m * n * 2)?;
        let c_c = gpu.alloc(m * n * 2)?;

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let iters = if m >= 2048 { 20 } else { 40 };

        let t1 = time_it(gpu, stream, m128, 128, a_ptr, p_ptr, s_ptr, c_a, m, n, k, iters)?;
        let t2 = time_it(gpu, stream, m64, 64, a_ptr, p_ptr, s_ptr, c_b, m, n, k, iters)?;
        // k64 requires K % 64 == 0
        let t3 = if k % 64 == 0 {
            time_it(gpu, stream, k64, 64, a_ptr, p_ptr, s_ptr, c_c, m, n, k, iters)?
        } else { f64::NAN };

        // byte-compare m128 vs M64 outputs
        let mut buf_a = vec![0u8; m * n * 2];
        let mut buf_b = vec![0u8; m * n * 2];
        gpu.copy_d2h(c_a, &mut buf_a)?;
        gpu.copy_d2h(c_b, &mut buf_b)?;
        if buf_a != buf_b { mismatch += 1; }

        println!("{label:<16} {m:>6} {n:>6} {k:>6} | {:>9.2} {:>9.2} {:>9.2} | {:>7.3}x {:>7.3}x",
                 flops / t1 / 1e12, flops / t2 / 1e12, flops / t3 / 1e12, t1 / t2, t1 / t3);

        for p in [a_ptr, p_ptr, s_ptr, c_a, c_b, c_c] { let _ = gpu.free(p); }
    }
    println!("{}", "-".repeat(92));
    println!("m128 vs M64 output byte-compare: {}",
             if mismatch == 0 { "IDENTICAL on all shapes" } else { "MISMATCH — grids or math differ!" });
    Ok(())
}
