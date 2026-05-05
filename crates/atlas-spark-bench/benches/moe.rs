// SPDX-License-Identifier: AGPL-3.0-only

//! MoE kernel microbenchmarks.
//!
//! Shapes match Qwen3-Next-80B-A3B: W4A16 grouped GEMM.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

static MOE_W4A16_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// moe_w4a16_grouped_gemm(a, b_packed, b_scale, scale2, c, expert_offsets, num_experts, n, k)
fn bench_moe_w4a16_grouped(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(reg, &MOE_W4A16_FN, "moe_w4a16", "moe_w4a16_grouped_gemm");

    let num_experts: u32 = 8; // Benchmark with 8 active experts
    let tokens_per_expert: u32 = 10;
    let n: u32 = 1024;
    let k: u32 = 2048;
    let group_size: u32 = 128; // Used for scale sizing
    let scale2: f32 = 1.0;

    let mut group = c.benchmark_group("moe_w4a16_grouped");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let total_tokens = (num_experts * tokens_per_expert) as usize;

    // A: [total_tokens, K] BF16
    let a_bytes = total_tokens * k as usize * 2;
    // B_packed: per expert [K, N/2] W4 packed
    let b_bytes = num_experts as usize * k as usize * n as usize / 2;
    // B_scale: per expert [K/gs, N] FP8
    let scale_bytes = num_experts as usize * (k as usize / group_size as usize) * n as usize;
    // C: [total_tokens, N] BF16
    let c_bytes = total_tokens * n as usize * 2;
    // expert_offsets: [num_experts] i32
    let offsets_bytes = num_experts as usize * 4;

    let a_ptr = gpu::gpu_alloc_zeroed(stream, a_bytes).unwrap();
    let b_packed_ptr = gpu::gpu_alloc_zeroed(stream, b_bytes).unwrap();
    let b_scale_ptr = gpu::gpu_alloc_zeroed(stream, scale_bytes).unwrap();
    let c_ptr = gpu::gpu_alloc_zeroed(stream, c_bytes).unwrap();
    let offsets_ptr = gpu::gpu_alloc_zeroed(stream, offsets_bytes).unwrap();
    gpu::gpu_sync(stream).unwrap();

    let label = format!("experts={num_experts} tp_exp={tokens_per_expert}");
    group.bench_function(&label, |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &a_ptr as *const u64 as *mut c_void,
                    &b_packed_ptr as *const u64 as *mut c_void,
                    &b_scale_ptr as *const u64 as *mut c_void,
                    &scale2 as *const f32 as *mut c_void,
                    &c_ptr as *const u64 as *mut c_void,
                    &offsets_ptr as *const u64 as *mut c_void,
                    &num_experts as *const u32 as *mut c_void,
                    &n as *const u32 as *mut c_void,
                    &k as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel,
                        (num_experts, 1, 1),
                        (256, 1, 1),
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
    gpu::gpu_free(b_packed_ptr);
    gpu::gpu_free(b_scale_ptr);
    gpu::gpu_free(c_ptr);
    gpu::gpu_free(offsets_ptr);

    group.finish();
}

criterion_group!(benches, bench_moe_w4a16_grouped);
criterion_main!(benches);
