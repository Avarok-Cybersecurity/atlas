// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 Phase 2b: batched GDN prefill ops, hoisted from `ssm_gdn_a.rs`
//! to keep that file under the 500-LoC file-size cap.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

// ───────────────────────────────────────────────────────────────────────
// Q12 Phase 2b: batched GDN prefill ops.
//
// Same-chunk-len batched variants for the four GDN kernels patched in
// commits 37c44cc / 81d76fa / 5a93095 / 43f7c25. Caller is responsible
// for:
//   1. Uploading a device array `h_state_ptrs: [float*; batch_size]`
//      containing the per-stream `SsmLayerState::h_state` pointers.
//   2. Stacking QKV / gate / beta / output for all batched streams
//      contiguously as `[batch_size * seq_len, conv_dim]` etc. (each
//      stream's input lands at `b * seq_len * stride`).
//   3. Ensuring all batched streams share the same `seq_len` (the
//      scheduler `can_batch_prefill_only` gate enforces this).
//
// Validation status: kernels unvalidated against hardware. Wiring
// these ops into `Qwen3SsmLayer::prefill_batched` is the next step;
// they're committed first so the kernel patches and the Rust bindings
// stay in lockstep.
// ───────────────────────────────────────────────────────────────────────

/// Batched WY32 persistent GDN prefill (uses gated_delta_rule_prefill_wy64_batched).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_persistent_smem_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state_ptrs: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    smem: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state_ptrs)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// Batched persistent GDN prefill (uses gated_delta_rule_prefill_persistent_batched
/// or gated_delta_rule_prefill_persistent_wy4_batched).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_persistent_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state_ptrs: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    let smem = k_dim * v_dim * 4 + 4 * k_dim * 4;
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .shared_mem(smem)
        .arg_ptr(h_state_ptrs)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// Batched split4 GDN prefill (uses gated_delta_rule_prefill_split4_batched).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_split4_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state_ptrs: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads * 4, batch_size, 1])
        .block([32, 1, 1])
        .shared_mem(4 * k_dim * 4)
        .arg_ptr(h_state_ptrs)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}
