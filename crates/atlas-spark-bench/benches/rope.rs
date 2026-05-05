// SPDX-License-Identifier: AGPL-3.0-only

//! RoPE kernel microbenchmark.
//!
//! Applies rotary embeddings to Q and K tensors in-place.
//! Shapes match Qwen3-Next-80B-A3B: head_dim=256, rotary_dim=64,
//! num_q_heads=16, num_kv_heads=2 (GQA 8:1).

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

static ROPE_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// rope_forward(Q, K, positions, seq_len, num_q_heads, num_kv_heads, head_dim, rotary_dim, theta)
/// Grid: (num_q_heads + num_kv_heads, ceil(seq_len / (128 / (rotary_dim / 2))), 1)
/// Block: (128, 1, 1)
/// Note: Z-dim corresponds to batch which is 1.
fn bench_rope(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(reg, &ROPE_FN, "rope", "rope_forward");

    let num_q_heads: u32 = 16;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 256;
    let rotary_dim: u32 = 64;
    let theta: f32 = 10_000_000.0;
    let elem_bytes = 2_usize;

    let mut group = c.benchmark_group("rope");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let shapes: Vec<u32> = vec![64, 256, 1024, 4096];

    for seq_len in shapes {
        let q_bytes = seq_len as usize * num_q_heads as usize * head_dim as usize * elem_bytes;
        let k_bytes = seq_len as usize * num_kv_heads as usize * head_dim as usize * elem_bytes;
        let pos_bytes = seq_len as usize * 4; // u32

        let q_ptr = gpu::gpu_alloc_zeroed(stream, q_bytes).unwrap();
        let k_ptr = gpu::gpu_alloc_zeroed(stream, k_bytes).unwrap();
        let pos_ptr = gpu::gpu_alloc_zeroed(stream, pos_bytes).unwrap();
        gpu::gpu_sync(stream).unwrap();

        // seq_blocks calculation from ops.rs
        let pos_per_block = 128 / (rotary_dim / 2);
        let seq_blocks = seq_len.div_ceil(pos_per_block);

        let label = format!("seq={seq_len}");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                    let mut params: Vec<*mut c_void> = vec![
                        &q_ptr as *const u64 as *mut c_void,
                        &k_ptr as *const u64 as *mut c_void,
                        &pos_ptr as *const u64 as *mut c_void,
                        &seq_len as *const u32 as *mut c_void,
                        &num_q_heads as *const u32 as *mut c_void,
                        &num_kv_heads as *const u32 as *mut c_void,
                        &head_dim as *const u32 as *mut c_void,
                        &rotary_dim as *const u32 as *mut c_void,
                        &theta as *const f32 as *mut c_void,
                    ];
                    unsafe {
                        gpu::launch(
                            reg,
                            kernel,
                            (num_q_heads + num_kv_heads, seq_blocks, 1),
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

        gpu::gpu_free(q_ptr);
        gpu::gpu_free(k_ptr);
        gpu::gpu_free(pos_ptr);
    }
    group.finish();
}

criterion_group!(benches, bench_rope);
criterion_main!(benches);
