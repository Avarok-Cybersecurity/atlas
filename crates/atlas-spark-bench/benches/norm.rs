// SPDX-License-Identifier: AGPL-3.0-only

//! RMSNorm and Gated RMSNorm kernel microbenchmarks.
//!
//! Shapes match Qwen3-Next-80B-A3B production inference (hidden=2048).
//! Requires a GPU — skips gracefully if unavailable.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

static RMS_NORM_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static GATED_RMS_NORM_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static RESIDUAL_ADD_RMS_NORM_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// RMS norm kernel: rms_norm(input, weight, output, hidden_size, eps)
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
fn bench_rms_norm(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(reg, &RMS_NORM_FN, "norm", "rms_norm");

    let hidden_size: u32 = 2048;
    let eps: f32 = 1e-6;
    let elem_bytes = 2_usize; // BF16

    let mut group = c.benchmark_group("rms_norm");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &batch in &[1u32, 16, 80] {
        let buf_bytes = batch as usize * hidden_size as usize * elem_bytes;
        let input = gpu::gpu_alloc_zeroed(stream, buf_bytes).unwrap();
        let weight = gpu::gpu_alloc_zeroed(stream, hidden_size as usize * elem_bytes).unwrap();
        let output = gpu::gpu_alloc_zeroed(stream, buf_bytes).unwrap();
        gpu::gpu_sync(stream).unwrap();

        let block_x = hidden_size.min(1024);
        let label = format!("[{batch},2048]");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                    let mut params: Vec<*mut c_void> = vec![
                        &input as *const u64 as *mut c_void,
                        &weight as *const u64 as *mut c_void,
                        &output as *const u64 as *mut c_void,
                        &hidden_size as *const u32 as *mut c_void,
                        &eps as *const f32 as *mut c_void,
                    ];
                    unsafe {
                        gpu::launch(
                            reg,
                            kernel,
                            (batch, 1, 1),
                            (block_x, 1, 1),
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

        gpu::gpu_free(input);
        gpu::gpu_free(weight);
        gpu::gpu_free(output);
    }
    group.finish();
}

/// Gated RMS norm: gated_rms_norm(input, gate, weight, output, hidden_size, eps)
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
fn bench_gated_rms_norm(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(reg, &GATED_RMS_NORM_FN, "norm", "gated_rms_norm");

    let eps: f32 = 1e-6;
    let batch: u32 = 1;
    let elem_bytes = 2_usize;

    let mut group = c.benchmark_group("gated_rms_norm");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &dim in &[2048u32, 8192] {
        let buf_bytes = batch as usize * dim as usize * elem_bytes;
        let input = gpu::gpu_alloc_zeroed(stream, buf_bytes).unwrap();
        let gate = gpu::gpu_alloc_zeroed(stream, buf_bytes).unwrap();
        let weight = gpu::gpu_alloc_zeroed(stream, dim as usize * elem_bytes).unwrap();
        let output = gpu::gpu_alloc_zeroed(stream, buf_bytes).unwrap();
        gpu::gpu_sync(stream).unwrap();

        let block_x = dim.min(1024);
        let label = format!("dim={dim}");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                    let mut params: Vec<*mut c_void> = vec![
                        &input as *const u64 as *mut c_void,
                        &gate as *const u64 as *mut c_void,
                        &weight as *const u64 as *mut c_void,
                        &output as *const u64 as *mut c_void,
                        &dim as *const u32 as *mut c_void,
                        &eps as *const f32 as *mut c_void,
                    ];
                    unsafe {
                        gpu::launch(
                            reg,
                            kernel,
                            (batch, 1, 1),
                            (block_x, 1, 1),
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

        gpu::gpu_free(input);
        gpu::gpu_free(gate);
        gpu::gpu_free(weight);
        gpu::gpu_free(output);
    }
    group.finish();
}

/// Block D #4 baseline: fused `residual_add_rms_norm` at MiniMax M2.7
/// hidden=4096. This is the 2/3-fused TokenWeave kernel already in
/// production (residual + RMSNorm fused; only the AllReduce stays
/// separate). Microbench gives the floor against which a TokenWeave-style
/// in-kernel-AllReduce variant can be compared. Across 62 layers, every
/// 1 µs saved here is ~62 µs cold TTFT.
///
/// Kernel: residual_add_rms_norm(hidden, src, weight, output, residual,
///   num_tokens, hidden_size, eps)
/// Grid: (num_tokens, 1, 1)  Block: (min(hidden_size, 1024), 1, 1)
fn bench_residual_add_rms_norm(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(
        reg,
        &RESIDUAL_ADD_RMS_NORM_FN,
        "norm",
        "residual_add_rms_norm",
    );

    // MiniMax M2.7 hidden=4096 (per-rank under TP=2 = 4096 total since hidden
    // is full-replicated, only attention/MLP weight tiles are sharded).
    let hidden_size: u32 = 4096;
    let eps: f32 = 1e-6;
    let elem_bytes = 2_usize; // BF16

    let mut group = c.benchmark_group("residual_add_rms_norm_minimax");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    // Token counts: 1 (decode), 1024 (mid prefill chunk), 4096 (full prefill).
    for &num_tokens in &[1u32, 1024, 4096] {
        let buf_bytes = num_tokens as usize * hidden_size as usize * elem_bytes;
        let weight_bytes = hidden_size as usize * elem_bytes;

        let hidden = gpu::gpu_alloc_zeroed(stream, buf_bytes).unwrap();
        let src = gpu::gpu_alloc_zeroed(stream, buf_bytes).unwrap();
        let weight = gpu::gpu_alloc_zeroed(stream, weight_bytes).unwrap();
        let output = gpu::gpu_alloc_zeroed(stream, buf_bytes).unwrap();
        let residual = gpu::gpu_alloc_zeroed(stream, buf_bytes).unwrap();
        gpu::gpu_sync(stream).unwrap();

        let block_x = hidden_size.min(1024);
        let label = format!("[{num_tokens},4096]");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                    let mut params: Vec<*mut c_void> = vec![
                        &hidden as *const u64 as *mut c_void,
                        &src as *const u64 as *mut c_void,
                        &weight as *const u64 as *mut c_void,
                        &output as *const u64 as *mut c_void,
                        &residual as *const u64 as *mut c_void,
                        &hidden_size as *const u32 as *mut c_void,
                        &eps as *const f32 as *mut c_void,
                    ];
                    unsafe {
                        gpu::launch(
                            reg,
                            kernel,
                            (num_tokens, 1, 1),
                            (block_x, 1, 1),
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

        gpu::gpu_free(hidden);
        gpu::gpu_free(src);
        gpu::gpu_free(weight);
        gpu::gpu_free(output);
        gpu::gpu_free(residual);
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rms_norm,
    bench_gated_rms_norm,
    bench_residual_add_rms_norm
);
criterion_main!(benches);
