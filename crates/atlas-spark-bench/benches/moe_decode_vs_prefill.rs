// SPDX-License-Identifier: AGPL-3.0-only

//! MoE decode-vs-prefill microbenchmark (Phase 2.8 Phase 1).
//!
//! Compares the *scalar GEMV* decode kernel
//! (`moe_expert_gate_up_shared`) against the *tensor-core GEMM* prefill
//! kernel (`moe_w4a16_grouped_gemm_ptrtable`) at varying token counts M,
//! to determine where a tensor-core refactor of the decode/verify path
//! would materially improve TPS.
//!
//! Workload mirrors Qwen3.6-35B-A3B per-MoE-layer dimensions:
//!   K = 2048 (hidden), N = 512 (intermediate), top_k = 8, num_experts = 8
//!
//! For each M ∈ {1, 2, 3, 8, 17, 32, 64, 256}:
//!   * Scalar baseline: M sequential `moe_expert_gate_up_shared` launches
//!     (each producing top_k×N output for one decode token).
//!   * Tensor-core comparison: 1 `moe_w4a16_grouped_gemm_ptrtable` launch
//!     processing M tokens per expert across num_experts groups
//!     (M·num_experts total token-expert pairs).

use std::ffi::c_void;
use std::sync::OnceLock;
use std::time::Duration;

use atlas_core::registry::RawCudaFunc;
use atlas_spark_bench::gpu;
use criterion::{Criterion, criterion_group, criterion_main};

unsafe extern "C" {
    fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> i32;
}

fn h2d<T: Copy>(dev: u64, host: &[T]) {
    let bytes = std::mem::size_of_val(host);
    unsafe {
        let rc = cuMemcpyHtoD_v2(dev, host.as_ptr() as *const c_void, bytes);
        assert_eq!(rc, 0, "cuMemcpyHtoD failed: {rc}");
    }
}

static SCALAR_FN: OnceLock<RawCudaFunc> = OnceLock::new();
static TC_FN: OnceLock<RawCudaFunc> = OnceLock::new();

const K: u32 = 2048;
const N: u32 = 512;
const TOP_K: u32 = 8;
const NUM_EXPERTS: u32 = 8;
const GROUP_SIZE: u32 = 16;
const M_TILE: u32 = 64;
const N_TILE_SM: u32 = 64;

fn bench_moe_decode_vs_prefill(c: &mut Criterion) {
    let reg = gpu::ensure_registry();
    let stream = reg.raw_stream();

    let scalar_kernel = gpu::get_kernel(
        reg,
        &SCALAR_FN,
        "moe_shared_expert_fused",
        "moe_expert_gate_up_shared",
    );
    let tc_kernel = gpu::get_kernel(reg, &TC_FN, "moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable");

    // ── Per-expert weight buffers ────────────────────────────────────
    let per_expert_packed_bytes = (N as usize) * (K as usize / 2);
    let per_expert_scale_bytes = (N as usize) * (K as usize / GROUP_SIZE as usize);

    let mut packed_ptrs_host: Vec<u64> = Vec::with_capacity(NUM_EXPERTS as usize);
    let mut scale_ptrs_host: Vec<u64> = Vec::with_capacity(NUM_EXPERTS as usize);
    let scale2_host: Vec<f32> = vec![1.0f32; NUM_EXPERTS as usize];

    for _ in 0..NUM_EXPERTS {
        packed_ptrs_host.push(gpu::gpu_alloc_zeroed(stream, per_expert_packed_bytes).unwrap());
        scale_ptrs_host.push(gpu::gpu_alloc_zeroed(stream, per_expert_scale_bytes).unwrap());
    }
    gpu::gpu_sync(stream).unwrap();

    let ptr_tbl_bytes = (NUM_EXPERTS as usize) * std::mem::size_of::<u64>();
    let s2_tbl_bytes = (NUM_EXPERTS as usize) * std::mem::size_of::<f32>();
    let gate_packed_ptrs_dev = gpu::gpu_alloc_zeroed(stream, ptr_tbl_bytes).unwrap();
    let gate_scale_ptrs_dev = gpu::gpu_alloc_zeroed(stream, ptr_tbl_bytes).unwrap();
    let gate_scale2_dev = gpu::gpu_alloc_zeroed(stream, s2_tbl_bytes).unwrap();
    let up_packed_ptrs_dev = gpu::gpu_alloc_zeroed(stream, ptr_tbl_bytes).unwrap();
    let up_scale_ptrs_dev = gpu::gpu_alloc_zeroed(stream, ptr_tbl_bytes).unwrap();
    let up_scale2_dev = gpu::gpu_alloc_zeroed(stream, s2_tbl_bytes).unwrap();

    h2d(gate_packed_ptrs_dev, &packed_ptrs_host);
    h2d(gate_scale_ptrs_dev, &scale_ptrs_host);
    h2d(gate_scale2_dev, &scale2_host);
    h2d(up_packed_ptrs_dev, &packed_ptrs_host);
    h2d(up_scale_ptrs_dev, &scale_ptrs_host);
    h2d(up_scale2_dev, &scale2_host);
    gpu::gpu_sync(stream).unwrap();

    let expert_indices_host: Vec<u32> = (0..TOP_K).collect();
    let expert_indices_dev = gpu::gpu_alloc_zeroed(stream, expert_indices_host.len() * 4).unwrap();
    h2d(expert_indices_dev, &expert_indices_host);

    // Activation A (1 token × K BF16) — scalar kernel processes 1 token per launch
    let a_dev = gpu::gpu_alloc_zeroed(stream, (K as usize) * 2).unwrap();

    // Output buffers for scalar
    let out_per_call_bytes = ((TOP_K + 1) as usize) * (N as usize) * 2;
    let gate_out_dev = gpu::gpu_alloc_zeroed(stream, out_per_call_bytes).unwrap();
    let up_out_dev = gpu::gpu_alloc_zeroed(stream, out_per_call_bytes).unwrap();
    let sh_gate_out_dev = gpu::gpu_alloc_zeroed(stream, (N as usize) * 2).unwrap();
    let sh_up_out_dev = gpu::gpu_alloc_zeroed(stream, (N as usize) * 2).unwrap();

    // ── TC kernel buffers (max M = 256, M*top_k tokens) ─────────────
    const M_MAX: u32 = 256;
    let total_tokens_max = M_MAX * NUM_EXPERTS;
    let tc_a_dev =
        gpu::gpu_alloc_zeroed(stream, (total_tokens_max as usize) * (K as usize) * 2).unwrap();
    let tc_c_dev =
        gpu::gpu_alloc_zeroed(stream, (total_tokens_max as usize) * (N as usize) * 2).unwrap();
    let tc_offsets_dev = gpu::gpu_alloc_zeroed(stream, ((NUM_EXPERTS + 1) as usize) * 4).unwrap();
    let tc_sorted_ids_dev = gpu::gpu_alloc_zeroed(stream, (total_tokens_max as usize) * 4).unwrap();
    gpu::gpu_sync(stream).unwrap();

    let scale2_unit: f32 = 1.0;
    let scale2_zero: f32 = 0.0;
    let zero_ptr: u64 = 0;

    let m_set: &[u32] = &[1, 2, 3, 8, 17, 32, 64, 256];

    let mut group = c.benchmark_group("moe_decode_vs_prefill");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));

    // Pre-populate TC offsets/sorted_ids per M (host-side prep, uploaded inside loop)
    for &m in m_set {
        // expert_offsets: each expert gets M tokens → offsets [0, M, 2M, ..., 8M]
        let offsets_host: Vec<i32> = (0..=NUM_EXPERTS as i32).map(|i| i * m as i32).collect();
        h2d(tc_offsets_dev, &offsets_host);
        // sorted_token_ids: just 0..M*8 (identity mapping)
        let sorted_ids_host: Vec<i32> = (0..(m * NUM_EXPERTS) as i32).collect();
        h2d(tc_sorted_ids_dev, &sorted_ids_host);
        gpu::gpu_sync(stream).unwrap();

        // ── Scalar baseline ────────────────────────────────────────
        // Grid: (N/8, top_k+1, 2)  Block: (128, 1, 1)
        let s_grid_x = N.div_ceil(8);
        let s_grid_y = TOP_K + 1;
        let s_grid_z = 2;

        let label_scalar = format!("M={m:03}_scalar");
        group.bench_function(&label_scalar, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 5, iters as usize, || {
                    for _ in 0..m {
                        let mut params: Vec<*mut c_void> = vec![
                            &a_dev as *const u64 as *mut c_void,
                            &gate_packed_ptrs_dev as *const u64 as *mut c_void,
                            &gate_scale_ptrs_dev as *const u64 as *mut c_void,
                            &gate_scale2_dev as *const u64 as *mut c_void,
                            &gate_out_dev as *const u64 as *mut c_void,
                            &up_packed_ptrs_dev as *const u64 as *mut c_void,
                            &up_scale_ptrs_dev as *const u64 as *mut c_void,
                            &up_scale2_dev as *const u64 as *mut c_void,
                            &up_out_dev as *const u64 as *mut c_void,
                            &expert_indices_dev as *const u64 as *mut c_void,
                            // Shared expert: NULL sentinel — kernel writes zeros and returns.
                            &zero_ptr as *const u64 as *mut c_void,
                            &zero_ptr as *const u64 as *mut c_void,
                            &scale2_zero as *const f32 as *mut c_void,
                            &sh_gate_out_dev as *const u64 as *mut c_void,
                            &zero_ptr as *const u64 as *mut c_void,
                            &zero_ptr as *const u64 as *mut c_void,
                            &scale2_zero as *const f32 as *mut c_void,
                            &sh_up_out_dev as *const u64 as *mut c_void,
                            &N as *const u32 as *mut c_void,
                            &K as *const u32 as *mut c_void,
                            &TOP_K as *const u32 as *mut c_void,
                        ];
                        unsafe {
                            gpu::launch(
                                reg,
                                scalar_kernel,
                                (s_grid_x, s_grid_y, s_grid_z),
                                (128, 1, 1),
                                0,
                                stream,
                                &mut params,
                            )
                            .unwrap();
                        }
                    }
                });
                Duration::from_secs_f64(ms as f64 / 1000.0 * iters as f64)
            });
        });

        // ── TC comparison: 1× moe_w4a16_grouped_gemm_ptrtable ───────
        // Grid: (N/N_TILE_SM, ceil(M_per_expert/M_TILE), num_experts)
        // Block: (128, 1, 1) — kernel uses 4 warps × 32 = 128 threads
        let tc_grid_x = N.div_ceil(N_TILE_SM);
        let tc_grid_y = m.div_ceil(M_TILE).max(1);
        let tc_grid_z = NUM_EXPERTS;

        let label_tc = format!("M={m:03}_tc");
        group.bench_function(&label_tc, |b| {
            b.iter_custom(|iters| {
                let ms = gpu::bench_kernel_ms(stream, 5, iters as usize, || {
                    let mut params: Vec<*mut c_void> = vec![
                        &tc_a_dev as *const u64 as *mut c_void,
                        &gate_packed_ptrs_dev as *const u64 as *mut c_void,
                        &gate_scale_ptrs_dev as *const u64 as *mut c_void,
                        &gate_scale2_dev as *const u64 as *mut c_void,
                        &tc_c_dev as *const u64 as *mut c_void,
                        &tc_offsets_dev as *const u64 as *mut c_void,
                        &tc_sorted_ids_dev as *const u64 as *mut c_void,
                        &NUM_EXPERTS as *const u32 as *mut c_void,
                        &N as *const u32 as *mut c_void,
                        &K as *const u32 as *mut c_void,
                    ];
                    let _ = scale2_unit; // keep variable in scope
                    unsafe {
                        gpu::launch(
                            reg,
                            tc_kernel,
                            (tc_grid_x, tc_grid_y, tc_grid_z),
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
    }

    // Free
    for &p in &packed_ptrs_host {
        gpu::gpu_free(p);
    }
    for &p in &scale_ptrs_host {
        gpu::gpu_free(p);
    }
    gpu::gpu_free(gate_packed_ptrs_dev);
    gpu::gpu_free(gate_scale_ptrs_dev);
    gpu::gpu_free(gate_scale2_dev);
    gpu::gpu_free(up_packed_ptrs_dev);
    gpu::gpu_free(up_scale_ptrs_dev);
    gpu::gpu_free(up_scale2_dev);
    gpu::gpu_free(expert_indices_dev);
    gpu::gpu_free(a_dev);
    gpu::gpu_free(gate_out_dev);
    gpu::gpu_free(up_out_dev);
    gpu::gpu_free(sh_gate_out_dev);
    gpu::gpu_free(sh_up_out_dev);
    gpu::gpu_free(tc_a_dev);
    gpu::gpu_free(tc_c_dev);
    gpu::gpu_free(tc_offsets_dev);
    gpu::gpu_free(tc_sorted_ids_dev);

    group.finish();
}

criterion_group!(benches, bench_moe_decode_vs_prefill);
criterion_main!(benches);
