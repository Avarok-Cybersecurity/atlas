// SPDX-License-Identifier: AGPL-3.0-only

//! MoE token-routing reduce + CUTLASS grouped ops — extracted from
//! `moe_grouped_a.rs` during the ≤500-line split. All public items remain
//! available at `crate::layers::ops::*` via the re-export in `ops.rs`.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// Counting sort tokens by expert assignment.
///
/// Produces sorted_token_ids (grouped by expert), expert_offsets (prefix sum),
/// and token_to_perm (reverse map for unpermute).
///
/// Grid: (1, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_sort_by_expert(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    topk_ids: DevicePtr,
    sorted_token_ids: DevicePtr,
    sorted_expert_ids: DevicePtr,
    expert_offsets: DevicePtr,
    token_to_perm: DevicePtr,
    total_expanded: u32,
    num_experts: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(topk_ids)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(sorted_expert_ids)
        .arg_ptr(expert_offsets)
        .arg_ptr(token_to_perm)
        .arg_u32(total_expanded)
        .arg_u32(num_experts)
        .arg_u32(topk)
        .launch(stream)
}

/// Unpermute + weighted reduce with pre-built reverse map.
///
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn moe_unpermute_reduce_indexed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    expert_output: DevicePtr,
    output: DevicePtr,
    token_to_perm: DevicePtr,
    topk_weights: DevicePtr,
    hidden_size: u32,
    num_tokens: u32,
    topk: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_output)
        .arg_ptr(output)
        .arg_ptr(token_to_perm)
        .arg_ptr(topk_weights)
        .arg_u32(hidden_size)
        .arg_u32(num_tokens)
        .arg_u32(topk)
        .launch(stream)
}

/// Batched sigmoid blend: output += sigmoid(dot(normed, gate_weight)) * shared_out.
///
/// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
pub fn moe_batched_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    shared_out: DevicePtr,
    normed: DevicePtr,
    gate_weight: DevicePtr,
    hidden_size: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(shared_out)
        .arg_ptr(normed)
        .arg_ptr(gate_weight)
        .arg_u32(hidden_size)
        .arg_u32(num_tokens)
        .launch(stream)
}

/// Single-launch CUTLASS grouped NVFP4 fused gate_up GEMM (Phase-2).
///
/// Bridges the device-resident per-expert pointer/scale tables to the host-side
/// [`spark_runtime::cutlass::nvfp4_grouped_gate_up_fused`] entry. `a` is the
/// expert-contiguous bf16 activation `[total_expanded, k]`. `gate_packed`/
/// `gate_sfb`/`up_packed`/`up_sfb` are device `[num_experts]` u64 pointer arrays
/// (into the CUTLASS-layout weight + swizzled-SFB tables); `gate_scale2`/
/// `up_scale2` are device `[num_experts]` f32 arrays; `expert_offsets` is the
/// device i32 `[num_experts+1]` prefix sum. This op copies all of those host-side
/// (the C entry indexes offsets/scale2 on the host and needs the pointer values),
/// syncs the stream so they are valid, then dispatches the single grouped launch.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gate_up_cutlass(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    sorted_token_ids: DevicePtr,
    gate_packed: DevicePtr,
    gate_sfb: DevicePtr,
    gate_scale2: DevicePtr,
    up_packed: DevicePtr,
    up_sfb: DevicePtr,
    up_scale2: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    num_experts: usize,
    inter: u32,
    hidden: u32,
    stream: u64,
) -> Result<()> {
    let read_u64 = |p: DevicePtr| -> Result<Vec<u64>> {
        let mut raw = vec![0u8; num_experts * 8];
        gpu.copy_d2h_on_stream(p, &mut raw, stream)?;
        Ok(raw
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    };
    let read_f32 = |p: DevicePtr| -> Result<Vec<f32>> {
        let mut raw = vec![0u8; num_experts * 4];
        gpu.copy_d2h_on_stream(p, &mut raw, stream)?;
        Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };

    let gate_packed_h = read_u64(gate_packed)?;
    let gate_sfb_h = read_u64(gate_sfb)?;
    let up_packed_h = read_u64(up_packed)?;
    let up_sfb_h = read_u64(up_sfb)?;
    let gate_scale2_h = read_f32(gate_scale2)?;
    let up_scale2_h = read_f32(up_scale2)?;

    let mut off_raw = vec![0u8; (num_experts + 1) * 4];
    gpu.copy_d2h_on_stream(expert_offsets, &mut off_raw, stream)?;
    // The pointer/scale/offset host snapshots above are needed by the C entry
    // before it can launch — make sure the async D2H copies have landed.
    gpu.synchronize(stream)?;
    let eoff: Vec<i32> = off_raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    spark_runtime::cutlass::nvfp4_grouped_gate_up_fused(
        a.0,
        sorted_token_ids.0,
        &gate_packed_h,
        &gate_sfb_h,
        &gate_scale2_h,
        &up_packed_h,
        &up_sfb_h,
        &up_scale2_h,
        c_gate.0,
        c_up.0,
        &eoff,
        inter,
        hidden,
        stream,
    )
}

/// Single-launch CUTLASS grouped NVFP4 DOWN projection. `a` is the post-SiLU
/// intermediate `[total_expanded, inter]` (already expert-contiguous — no gather).
/// `packed`/`sfb` are device `[num_experts]` u64 pointer arrays into the
/// `[N=hidden,K/2]` down packed + swizzled-SFB tables; `scale2` is the device
/// `[num_experts]` f32 array; `expert_offsets` is the device i32 `[num_experts+1]`
/// prefix sum. Snapshots the pointer/offset tables host-side, then dispatches.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_down_cutlass(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    packed: DevicePtr,
    sfb: DevicePtr,
    scale2: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    num_experts: usize,
    hidden: u32,
    inter: u32,
    stream: u64,
) -> Result<()> {
    let mut praw = vec![0u8; num_experts * 8];
    gpu.copy_d2h_on_stream(packed, &mut praw, stream)?;
    let mut sraw = vec![0u8; num_experts * 8];
    gpu.copy_d2h_on_stream(sfb, &mut sraw, stream)?;
    let mut s2raw = vec![0u8; num_experts * 4];
    gpu.copy_d2h_on_stream(scale2, &mut s2raw, stream)?;
    let mut off_raw = vec![0u8; (num_experts + 1) * 4];
    gpu.copy_d2h_on_stream(expert_offsets, &mut off_raw, stream)?;
    gpu.synchronize(stream)?;
    let packed_h: Vec<u64> = praw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8")))
        .collect();
    let sfb_h: Vec<u64> = sraw
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().expect("8")))
        .collect();
    let scale2_h: Vec<f32> = s2raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4")))
        .collect();
    let eoff: Vec<i32> = off_raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().expect("4")))
        .collect();
    spark_runtime::cutlass::nvfp4_grouped_down(
        a.0, &packed_h, &sfb_h, &scale2_h, c.0, &eoff, hidden, inter, stream,
    )
}
