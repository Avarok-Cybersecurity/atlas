// SPDX-License-Identifier: AGPL-3.0-only

//! SSM (Gated Delta Net) kernel microbenchmarks.
//!
//! Includes causal conv1d update and gated delta rule decode.
//! Shapes match Qwen3-Next-80B-A3B: d_inner=8192, d_conv=4,
//! gdn_num_k=16, gdn_num_v=32, dim=128.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

static CONV1D_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static GDN_DECODE_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// causal_conv1d_update(conv_state, new_input, weight, bias, output, batch, dim, d_conv)
/// Grid: (ceil(dim/256), batch, 1)  Block: (256, 1, 1)
fn bench_conv1d(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(reg, &CONV1D_FN, "causal_conv1d", "causal_conv1d_update");

    let batch: u32 = 1;
    let d_inner: u32 = 8192;
    let d_conv: u32 = 4;
    let elem_bytes = 2_usize;

    // conv_state: [batch, d_inner, d_conv] FP32 (4 bytes)
    let state_bytes = batch as usize * d_inner as usize * d_conv as usize * 4;
    let input_bytes = batch as usize * d_inner as usize * elem_bytes;
    let weight_bytes = d_inner as usize * d_conv as usize * 4;
    let bias_bytes = d_inner as usize * 4;
    let output_bytes = batch as usize * d_inner as usize * elem_bytes;

    let state_ptr = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let input_ptr = gpu::gpu_alloc_zeroed(stream, input_bytes).unwrap();
    let weight_ptr = gpu::gpu_alloc_zeroed(stream, weight_bytes).unwrap();
    let bias_ptr = gpu::gpu_alloc_zeroed(stream, bias_bytes).unwrap();
    let output_ptr = gpu::gpu_alloc_zeroed(stream, output_bytes).unwrap();
    gpu::gpu_sync(stream).unwrap();

    let grid_x = d_inner.div_ceil(256);

    let mut group = c.benchmark_group("conv1d");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let label = format!("decode dim={d_inner}");
    group.bench_function(&label, |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &state_ptr as *const u64 as *mut c_void,
                    &input_ptr as *const u64 as *mut c_void,
                    &weight_ptr as *const u64 as *mut c_void,
                    &bias_ptr as *const u64 as *mut c_void,
                    &output_ptr as *const u64 as *mut c_void,
                    &batch as *const u32 as *mut c_void,
                    &d_inner as *const u32 as *mut c_void,
                    &d_conv as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel,
                        (grid_x, batch, 1),
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

    gpu::gpu_free(state_ptr);
    gpu::gpu_free(input_ptr);
    gpu::gpu_free(weight_ptr);
    gpu::gpu_free(bias_ptr);
    gpu::gpu_free(output_ptr);

    group.finish();
}

/// gated_delta_rule_decode(h_state, query, key, value, gate, beta, output,
///                         batch_size, num_k_heads, num_v_heads, k_dim, v_dim)
/// Grid: (num_v_heads, batch_size, 1)  Block: (128, 1, 1)
fn bench_gdn(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel = gpu::get_kernel(
        reg,
        &GDN_DECODE_FN,
        "gated_delta_rule",
        "gated_delta_rule_decode",
    );

    let batch: u32 = 1;
    let num_k_heads: u32 = 16;
    let num_v_heads: u32 = 32;
    let k_dim: u32 = 128;
    let v_dim: u32 = 128;
    let elem_bytes = 2_usize;

    // h_state: [batch, num_v_heads, k_dim, v_dim] FP32
    let state_bytes = batch as usize * num_v_heads as usize * k_dim as usize * v_dim as usize * 4;
    let q_bytes = batch as usize * num_k_heads as usize * k_dim as usize * elem_bytes;
    let k_bytes = batch as usize * num_k_heads as usize * k_dim as usize * elem_bytes;
    let v_bytes = batch as usize * num_v_heads as usize * v_dim as usize * elem_bytes;
    let gate_bytes = batch as usize * num_v_heads as usize * 4; // FP32
    let beta_bytes = gate_bytes;
    let output_bytes = batch as usize * num_v_heads as usize * v_dim as usize * elem_bytes;

    let h_state_ptr = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let q_ptr = gpu::gpu_alloc_zeroed(stream, q_bytes).unwrap();
    let k_ptr = gpu::gpu_alloc_zeroed(stream, k_bytes).unwrap();
    let v_ptr = gpu::gpu_alloc_zeroed(stream, v_bytes).unwrap();
    let gate_ptr = gpu::gpu_alloc_zeroed(stream, gate_bytes).unwrap();
    let beta_ptr = gpu::gpu_alloc_zeroed(stream, beta_bytes).unwrap();
    let output_ptr = gpu::gpu_alloc_zeroed(stream, output_bytes).unwrap();
    gpu::gpu_sync(stream).unwrap();

    let mut group = c.benchmark_group("gdn");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    let label = format!("decode {num_v_heads}vh dim={k_dim}");
    group.bench_function(&label, |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let mut params: Vec<*mut c_void> = vec![
                    &h_state_ptr as *const u64 as *mut c_void,
                    &q_ptr as *const u64 as *mut c_void,
                    &k_ptr as *const u64 as *mut c_void,
                    &v_ptr as *const u64 as *mut c_void,
                    &gate_ptr as *const u64 as *mut c_void,
                    &beta_ptr as *const u64 as *mut c_void,
                    &output_ptr as *const u64 as *mut c_void,
                    &batch as *const u32 as *mut c_void,
                    &num_k_heads as *const u32 as *mut c_void,
                    &num_v_heads as *const u32 as *mut c_void,
                    &k_dim as *const u32 as *mut c_void,
                    &v_dim as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel,
                        (num_v_heads, batch, 1),
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

    gpu::gpu_free(h_state_ptr);
    gpu::gpu_free(q_ptr);
    gpu::gpu_free(k_ptr);
    gpu::gpu_free(v_ptr);
    gpu::gpu_free(gate_ptr);
    gpu::gpu_free(beta_ptr);
    gpu::gpu_free(output_ptr);

    group.finish();
}

static GDN_CHUNK2_FN: OnceLock<RawCudaFunc> = OnceLock::new();

/// gated_delta_rule_chunk2 vs 2× sequential gated_delta_rule_decode.
/// Measures the kernel-level speedup of fused 2-token processing.
fn bench_gdn_chunk2(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();
    let kernel_seq = gpu::get_kernel(
        reg,
        &GDN_DECODE_FN,
        "gated_delta_rule",
        "gated_delta_rule_decode",
    );
    let kernel_chunk2 = gpu::get_kernel(
        reg,
        &GDN_CHUNK2_FN,
        "gated_delta_rule",
        "gated_delta_rule_chunk2",
    );

    let batch: u32 = 1;
    let num_k_heads: u32 = 16;
    let num_v_heads: u32 = 32;
    let k_dim: u32 = 128;
    let v_dim: u32 = 128;
    let bf16 = 2_usize;
    let fp32 = 4_usize;

    // Shared dimensions
    let key_dim = num_k_heads as usize * k_dim as usize; // 2048
    let value_dim = num_v_heads as usize * v_dim as usize; // 4096
    let conv_dim: usize = key_dim * 2 + value_dim; // 8192

    // h_state: [batch, num_v_heads, k_dim, v_dim] FP32
    let state_bytes = batch as usize * num_v_heads as usize * k_dim as usize * v_dim as usize * 4;

    // Chunk2 layout: Q/K/V interleaved per token with stride = conv_dim
    // Layout: [2, conv_dim] = [2, Q(2048) + K(2048) + V(4096)]
    let qkv_buf_bytes = 2 * conv_dim * bf16;
    // gate+beta: [2, nv + nv] FP32
    let gb_buf_bytes = 2 * num_v_heads as usize * 2 * fp32;
    // output: [2, value_dim] BF16
    let out_buf_bytes = 2 * value_dim * bf16;

    let h_state_ptr = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let h_state_copy = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let h_inter_ptr = gpu::gpu_alloc_zeroed(stream, state_bytes).unwrap();
    let qkv_buf = gpu::gpu_alloc_zeroed(stream, qkv_buf_bytes).unwrap();
    let gb_buf = gpu::gpu_alloc_zeroed(stream, gb_buf_bytes).unwrap();
    let out_buf = gpu::gpu_alloc_zeroed(stream, out_buf_bytes).unwrap();
    gpu::gpu_sync(stream).unwrap();

    // Strides for chunk2 kernel
    let qk_stride: u32 = conv_dim as u32;
    let v_stride_val: u32 = conv_dim as u32;
    let gb_stride: u32 = num_v_heads * 2;

    // Per-token offsets for sequential path
    let q0_offset = 0_u64;
    let k0_offset = (key_dim * bf16) as u64;
    let v0_offset = (key_dim * 2 * bf16) as u64;
    let q1_offset = (conv_dim * bf16) as u64;
    let k1_offset = (conv_dim * bf16 + key_dim * bf16) as u64;
    let v1_offset = (conv_dim * bf16 + key_dim * 2 * bf16) as u64;
    let gate0_offset = 0_u64;
    let beta0_offset = (num_v_heads as usize * fp32) as u64;
    let gate1_offset = (num_v_heads as usize * 2 * fp32) as u64;
    let beta1_offset = (num_v_heads as usize * 3 * fp32) as u64;
    let out0_offset = 0_u64;
    let out1_offset = (value_dim * bf16) as u64;

    let mut group = c.benchmark_group("gdn_chunk2");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    // Benchmark: 2× sequential gdn_decode
    let one: u32 = 1;
    group.bench_function("sequential_2x", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                // Token 0
                let q0 = qkv_buf + q0_offset;
                let k0 = qkv_buf + k0_offset;
                let v0 = qkv_buf + v0_offset;
                let g0 = gb_buf + gate0_offset;
                let b0 = gb_buf + beta0_offset;
                let o0 = out_buf + out0_offset;
                let mut params0: Vec<*mut c_void> = vec![
                    &h_state_ptr as *const u64 as *mut c_void,
                    &q0 as *const u64 as *mut c_void,
                    &k0 as *const u64 as *mut c_void,
                    &v0 as *const u64 as *mut c_void,
                    &g0 as *const u64 as *mut c_void,
                    &b0 as *const u64 as *mut c_void,
                    &o0 as *const u64 as *mut c_void,
                    &one as *const u32 as *mut c_void,
                    &num_k_heads as *const u32 as *mut c_void,
                    &num_v_heads as *const u32 as *mut c_void,
                    &k_dim as *const u32 as *mut c_void,
                    &v_dim as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel_seq,
                        (num_v_heads, one, 1),
                        (128, 1, 1),
                        0,
                        stream,
                        &mut params0,
                    )
                    .unwrap();
                }
                // Token 1
                let q1 = qkv_buf + q1_offset;
                let k1 = qkv_buf + k1_offset;
                let v1 = qkv_buf + v1_offset;
                let g1 = gb_buf + gate1_offset;
                let b1 = gb_buf + beta1_offset;
                let o1 = out_buf + out1_offset;
                let mut params1: Vec<*mut c_void> = vec![
                    &h_state_ptr as *const u64 as *mut c_void,
                    &q1 as *const u64 as *mut c_void,
                    &k1 as *const u64 as *mut c_void,
                    &v1 as *const u64 as *mut c_void,
                    &g1 as *const u64 as *mut c_void,
                    &b1 as *const u64 as *mut c_void,
                    &o1 as *const u64 as *mut c_void,
                    &one as *const u32 as *mut c_void,
                    &num_k_heads as *const u32 as *mut c_void,
                    &num_v_heads as *const u32 as *mut c_void,
                    &k_dim as *const u32 as *mut c_void,
                    &v_dim as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel_seq,
                        (num_v_heads, one, 1),
                        (128, 1, 1),
                        0,
                        stream,
                        &mut params1,
                    )
                    .unwrap();
                }
            });
            Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
        });
    });

    // Benchmark: 1× chunk2 gdn_decode
    group.bench_function("chunk2_fused", |b| {
        b.iter_custom(|iters| {
            let ms = gpu::bench_kernel_ms(stream, 50, iters as usize, || {
                let q_base = qkv_buf;
                let k_base = qkv_buf + k0_offset;
                let v_base = qkv_buf + v0_offset;
                let g_base = gb_buf;
                let b_base = gb_buf + beta0_offset;
                let mut params: Vec<*mut c_void> = vec![
                    &h_state_ptr as *const u64 as *mut c_void,
                    &q_base as *const u64 as *mut c_void,
                    &k_base as *const u64 as *mut c_void,
                    &v_base as *const u64 as *mut c_void,
                    &g_base as *const u64 as *mut c_void,
                    &b_base as *const u64 as *mut c_void,
                    &out_buf as *const u64 as *mut c_void,
                    &h_inter_ptr as *const u64 as *mut c_void,
                    &batch as *const u32 as *mut c_void,
                    &num_k_heads as *const u32 as *mut c_void,
                    &num_v_heads as *const u32 as *mut c_void,
                    &k_dim as *const u32 as *mut c_void,
                    &v_dim as *const u32 as *mut c_void,
                    &qk_stride as *const u32 as *mut c_void,
                    &v_stride_val as *const u32 as *mut c_void,
                    &gb_stride as *const u32 as *mut c_void,
                ];
                unsafe {
                    gpu::launch(
                        reg,
                        kernel_chunk2,
                        (num_v_heads, batch, 1),
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

    gpu::gpu_free(h_state_ptr);
    gpu::gpu_free(h_state_copy);
    gpu::gpu_free(h_inter_ptr);
    gpu::gpu_free(qkv_buf);
    gpu::gpu_free(gb_buf);
    gpu::gpu_free(out_buf);

    group.finish();
}

criterion_group!(benches, bench_conv1d, bench_gdn, bench_gdn_chunk2);
criterion_main!(benches);
