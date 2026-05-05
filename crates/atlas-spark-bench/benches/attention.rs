// SPDX-License-Identifier: AGPL-3.0-only

//! Attention kernel microbenchmarks.
//!
//! - `paged_decode_fp8`: Qwen3-Next-80B-A3B decode shapes.
//! - `prefill_attn_64`: MiniMax M2.7 TP=2 prefill shapes — Block A
//!   (FlashInfer-grade attention kernel tuning). Measures per-call latency
//!   at the shapes the cold-TTFT-dominant attention spends ~190 ms across
//!   62 layers. Provides the reference point against which a tuned kernel
//!   (or FlashInfer reference run) should be compared.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

static PAGED_DECODE_FP8_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static PREFILL_ATTN_64_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// paged_decode_attn_fp8(Q, K_cache, V_cache, O, block_tables, seq_lens, max_blocks_per_seq,
///   num_q_heads, num_kv_heads, head_dim, block_size, inv_sqrt_d, k_scale, v_scale,
///   q_stride, cache_stride)
fn bench_paged_decode(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(
        reg,
        &PAGED_DECODE_FP8_FN,
        "paged_decode_fp8",
        "paged_decode_attn_fp8",
    );

    let num_seqs: u32 = 1;
    let num_q_heads: u32 = 16;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 256;
    let block_size: u32 = 16; // 16 tokens per block
    let inv_sqrt_d: f32 = 1.0 / (head_dim as f32).sqrt();
    let k_scale: f32 = 1.0;
    let v_scale: f32 = 1.0;

    let q_stride: u32 = num_q_heads * head_dim;
    let cache_stride: u64 = block_size as u64 * num_kv_heads as u64 * head_dim as u64;

    let mut group = c.benchmark_group("paged_decode_fp8");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let seq_lens: Vec<u32> = vec![64, 256, 1024, 4096];

    for seq_len in seq_lens {
        let max_blocks_per_seq = seq_len.div_ceil(block_size);

        let q_bytes = num_seqs as usize * num_q_heads as usize * head_dim as usize * 2;
        let o_bytes = q_bytes;
        // Mock a pool size large enough to hold our max blocks
        let cache_pool_bytes = max_blocks_per_seq as usize * cache_stride as usize; // FP8 is 1 byte

        let block_tables_bytes = num_seqs as usize * max_blocks_per_seq as usize * 4;
        let seq_lens_bytes = num_seqs as usize * 4;

        let q_ptr = gpu::gpu_alloc_zeroed(stream, q_bytes).unwrap();
        let k_cache_ptr = gpu::gpu_alloc_zeroed(stream, cache_pool_bytes).unwrap();
        let v_cache_ptr = gpu::gpu_alloc_zeroed(stream, cache_pool_bytes).unwrap();
        let o_ptr = gpu::gpu_alloc_zeroed(stream, o_bytes).unwrap();
        let block_tables_ptr = gpu::gpu_alloc_zeroed(stream, block_tables_bytes).unwrap();
        let seq_lens_ptr = gpu::gpu_alloc_zeroed(stream, seq_lens_bytes).unwrap();
        gpu::gpu_sync(stream).unwrap();

        let label = format!("seq={seq_len}");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                    let mut params: Vec<*mut c_void> = vec![
                        &q_ptr as *const u64 as *mut c_void,
                        &k_cache_ptr as *const u64 as *mut c_void,
                        &v_cache_ptr as *const u64 as *mut c_void,
                        &o_ptr as *const u64 as *mut c_void,
                        &block_tables_ptr as *const u64 as *mut c_void,
                        &seq_lens_ptr as *const u64 as *mut c_void,
                        &max_blocks_per_seq as *const u32 as *mut c_void,
                        &num_q_heads as *const u32 as *mut c_void,
                        &num_kv_heads as *const u32 as *mut c_void,
                        &head_dim as *const u32 as *mut c_void,
                        &block_size as *const u32 as *mut c_void,
                        &inv_sqrt_d as *const f32 as *mut c_void,
                        &k_scale as *const f32 as *mut c_void,
                        &v_scale as *const f32 as *mut c_void,
                        &q_stride as *const u32 as *mut c_void,
                        &cache_stride as *const u64 as *mut c_void,
                    ];
                    unsafe {
                        gpu::launch(
                            reg,
                            kernel,
                            (num_q_heads, num_seqs, 1),
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

        gpu::gpu_free(q_ptr);
        gpu::gpu_free(k_cache_ptr);
        gpu::gpu_free(v_cache_ptr);
        gpu::gpu_free(o_ptr);
        gpu::gpu_free(block_tables_ptr);
        gpu::gpu_free(seq_lens_ptr);
    }
    group.finish();
}

/// Block A baseline: contiguous prefill flash attention at MiniMax M2.7
/// TP=2 shapes. Q[N, 24, 128] BF16, K/V[N, 4, 128] BF16, page-less.
///
/// Roadmap target: ~190 ms total across 62 layers at N=4096 cold prefill.
/// Per-layer = 3.0–3.4 ms — this microbench isolates one layer to enable
/// kernel-level tuning iterations against a fixed shape.
fn bench_prefill_attn_64(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(
        reg,
        &PREFILL_ATTN_64_FN,
        "inferspark_prefill",
        "inferspark_prefill_64",
    );

    // MiniMax M2.7 TP=2 attention shape (nq=48 / 2 ranks = 24).
    let num_q_heads: u32 = 24;
    let num_kv_heads: u32 = 4;
    let head_dim: u32 = 128;
    let inv_sqrt_d: f32 = 1.0 / (head_dim as f32).sqrt();
    let causal: u32 = 1;
    let sliding_window: u32 = 0;

    let mut group = c.benchmark_group("prefill_attn_64_minimax_tp2");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));

    // Prefill chunk sizes seen in production: 1024 (one Atlas chunk),
    // 4096 (cold-TTFT-dominant size), 8192 (long-context limit).
    let seq_lens: Vec<u32> = vec![1024, 4096, 8192];

    let bf16 = 2usize;

    for seq_len in seq_lens {
        let q_bytes = seq_len as usize * num_q_heads as usize * head_dim as usize * bf16;
        let kv_bytes = seq_len as usize * num_kv_heads as usize * head_dim as usize * bf16;
        let o_bytes = q_bytes;

        let q_ptr = gpu::gpu_alloc_zeroed(stream, q_bytes).unwrap();
        let k_ptr = gpu::gpu_alloc_zeroed(stream, kv_bytes).unwrap();
        let v_ptr = gpu::gpu_alloc_zeroed(stream, kv_bytes).unwrap();
        let o_ptr = gpu::gpu_alloc_zeroed(stream, o_bytes).unwrap();
        gpu::gpu_sync(stream).unwrap();

        let br: u32 = 64;
        let label = format!("seq={seq_len}");
        group.bench_function(&label, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                    let mut params: Vec<*mut c_void> = vec![
                        &q_ptr as *const u64 as *mut c_void,
                        &k_ptr as *const u64 as *mut c_void,
                        &v_ptr as *const u64 as *mut c_void,
                        &o_ptr as *const u64 as *mut c_void,
                        &seq_len as *const u32 as *mut c_void,
                        &num_q_heads as *const u32 as *mut c_void,
                        &num_kv_heads as *const u32 as *mut c_void,
                        &head_dim as *const u32 as *mut c_void,
                        &inv_sqrt_d as *const f32 as *mut c_void,
                        &causal as *const u32 as *mut c_void,
                        &sliding_window as *const u32 as *mut c_void,
                    ];
                    unsafe {
                        gpu::launch(
                            reg,
                            kernel,
                            (num_q_heads, seq_len.div_ceil(br), 1),
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

        gpu::gpu_free(q_ptr);
        gpu::gpu_free(k_ptr);
        gpu::gpu_free(v_ptr);
        gpu::gpu_free(o_ptr);
    }
    group.finish();
}

criterion_group!(benches, bench_paged_decode, bench_prefill_attn_64);
criterion_main!(benches);
