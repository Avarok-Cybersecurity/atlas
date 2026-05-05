// SPDX-License-Identifier: AGPL-3.0-only

//! Dense GEMM and W4A16 kernel microbenchmarks.
//!
//! Shapes match Qwen3-Next-80B-A3B production inference.
//! Requires a GPU — skips gracefully if unavailable.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

static GEMM_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static W4A16_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// Dense BF16 GEMM: dense_gemm_bf16(A, B, C, M, N, K)
/// A: [M,K], B: [N,K] row-major, C: [M,N]
/// Grid: (ceil(N/16), ceil(M/16), 1)  Block: (16, 16, 1)
fn bench_dense_gemm(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(reg, &GEMM_FN, "gemm", "dense_gemm_bf16");

    let elem_bytes = 2_usize;

    let shapes: Vec<(u32, u32, u32, &str)> = vec![
        (80, 512, 2048, "MoE_gate_up_80x2048x512"),
        (80, 2048, 512, "MoE_down_80x512x2048"),
        (16, 256, 2048, "Attn_Q_16x2048x256"),
        (256, 256, 256, "Medium_256x256x256"),
    ];

    let mut group = c.benchmark_group("gemm_bf16");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for (m, n, k, label) in &shapes {
        // A: [M,K], B: [N,K] (weight row-major), C: [M,N]
        let a_bytes = *m as usize * *k as usize * elem_bytes;
        let b_bytes = *n as usize * *k as usize * elem_bytes;
        let c_bytes = *m as usize * *n as usize * elem_bytes;

        let a_ptr = gpu::gpu_alloc_zeroed(stream, a_bytes).unwrap();
        let b_ptr = gpu::gpu_alloc_zeroed(stream, b_bytes).unwrap();
        let c_ptr = gpu::gpu_alloc_zeroed(stream, c_bytes).unwrap();
        gpu::gpu_sync(stream).unwrap();

        let grid_x = n.div_ceil(16);
        let grid_y = m.div_ceil(16);

        group.bench_function(*label, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                    let mut params: Vec<*mut c_void> = vec![
                        &a_ptr as *const u64 as *mut c_void,
                        &b_ptr as *const u64 as *mut c_void,
                        &c_ptr as *const u64 as *mut c_void,
                        m as *const u32 as *mut c_void,
                        n as *const u32 as *mut c_void,
                        k as *const u32 as *mut c_void,
                    ];
                    unsafe {
                        gpu::launch(
                            reg,
                            kernel,
                            (grid_x, grid_y, 1),
                            (16, 16, 1),
                            0,
                            stream,
                            &mut params,
                        )
                        .unwrap();
                    }
                });
                Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
            });
        });

        gpu::gpu_free(a_ptr);
        gpu::gpu_free(b_ptr);
        gpu::gpu_free(c_ptr);
    }
    group.finish();
}

/// W4A16 GEMM: w4a16_gemm(A, B_packed, B_scale, scale2, C, M, N, K)
/// Grid: (ceil(N/64), ceil(M/64), 1)  Block: (128, 1, 1)
fn bench_w4a16(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(reg, &W4A16_FN, "w4a16", "w4a16_gemm");

    let m: u32 = 80;
    let n: u32 = 1024;
    let k: u32 = 2048;
    let group_size: u32 = 128;
    let elem_bytes = 2_usize;
    let scale2: f32 = 1.0;

    // A: [M,K] BF16; B: packed FP4 [K*N/2 bytes]; scale: FP8 [(K/gs)*N bytes]
    let a_bytes = m as usize * k as usize * elem_bytes;
    let b_bytes = k as usize * n as usize / 2;
    let scale_bytes = (k as usize / group_size as usize) * n as usize;
    let c_bytes = m as usize * n as usize * elem_bytes;

    let a_ptr = gpu::gpu_alloc_zeroed(stream, a_bytes).unwrap();
    let b_ptr = gpu::gpu_alloc_zeroed(stream, b_bytes).unwrap();
    let scale_ptr = gpu::gpu_alloc_zeroed(stream, scale_bytes).unwrap();
    let c_ptr = gpu::gpu_alloc_zeroed(stream, c_bytes).unwrap();
    gpu::gpu_sync(stream).unwrap();

    let grid_x = n.div_ceil(64);
    let grid_y = m.div_ceil(64);

    let mut group = c.benchmark_group("w4a16");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let label = format!("[{m},{k}]x[{k},{n}]_nvfp4");
    group.bench_function(&label, |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &a_ptr as *const u64 as *mut c_void,
                    &b_ptr as *const u64 as *mut c_void,
                    &scale_ptr as *const u64 as *mut c_void,
                    &scale2 as *const f32 as *mut c_void,
                    &c_ptr as *const u64 as *mut c_void,
                    &m as *const u32 as *mut c_void,
                    &n as *const u32 as *mut c_void,
                    &k as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel,
                        (grid_x, grid_y, 1),
                        (128, 1, 1),
                        0,
                        stream,
                        &mut params,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    gpu::gpu_free(a_ptr);
    gpu::gpu_free(b_ptr);
    gpu::gpu_free(scale_ptr);
    gpu::gpu_free(c_ptr);

    group.finish();
}

criterion_group!(benches, bench_dense_gemm, bench_w4a16);
criterion_main!(benches);
