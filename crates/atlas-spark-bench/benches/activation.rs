// SPDX-License-Identifier: AGPL-3.0-only

//! SiLU×Mul activation kernel microbenchmarks.
//!
//! Shapes match Qwen3-Next-80B-A3B MoE intermediate (inter_size=512).
//! Requires a GPU — skips gracefully if unavailable.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

static SILU_MUL_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// SiLU×Mul: silu_mul_separate(gate, up, output, n)
/// Grid: (ceil(n/256), 1, 1)  Block: (256, 1, 1)
fn bench_silu_mul(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(reg, &SILU_MUL_FN, "residual_add", "silu_mul_separate");

    let inter_size: u32 = 512;
    let elem_bytes = 2_usize;

    let mut group = c.benchmark_group("silu_mul");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &num_tokens in &[16u32, 80, 800] {
        let gate_bytes = num_tokens as usize * inter_size as usize * elem_bytes;
        let up_bytes = gate_bytes;
        let out_bytes = gate_bytes;
        let num_elements = num_tokens * inter_size;

        let gate_ptr = gpu::gpu_alloc_zeroed(stream, gate_bytes).unwrap();
        let up_ptr = gpu::gpu_alloc_zeroed(stream, up_bytes).unwrap();
        let output = gpu::gpu_alloc_zeroed(stream, out_bytes).unwrap();
        gpu::gpu_sync(stream).unwrap();

        let label = format!("[{num_tokens},{inter_size}]");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                    let mut params: Vec<*mut c_void> = vec![
                        &gate_ptr as *const u64 as *mut c_void,
                        &up_ptr as *const u64 as *mut c_void,
                        &output as *const u64 as *mut c_void,
                        &num_elements as *const u32 as *mut c_void,
                    ];
                    unsafe {
                        gpu::launch(
                            reg,
                            kernel,
                            (num_elements.div_ceil(256), 1, 1),
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

        gpu::gpu_free(gate_ptr);
        gpu::gpu_free(up_ptr);
        gpu::gpu_free(output);
    }
    group.finish();
}

criterion_group!(benches, bench_silu_mul);
criterion_main!(benches);
