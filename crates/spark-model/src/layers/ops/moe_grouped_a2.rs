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

// Counting sort tokens by expert assignment.
//
// Produces sorted_token_ids (grouped by expert), expert_offsets (prefix sum),
// and token_to_perm (reverse map for unpermute).
//
// Grid: (1, 1, 1)  Block: (256, 1, 1)

/// Host snapshots of per-expert pointer/scale arrays, keyed by device pointer.
///
/// The CUTLASS grouped entry needs these on the HOST to build its per-group
/// problem shapes, so each call used to D2H six arrays that are IMMUTABLE after
/// `build_cutlass_grouped_sfb` — 6 copies x 2 calls x 47 MoE layers = 564
/// pointless transfers per prefill. Only `expert_offsets` genuinely changes per
/// call (it is produced by the sort), so only that one still needs a copy and
/// the synchronize that waits for it.
fn cached_u64(gpu: &dyn GpuBackend, p: DevicePtr, n: usize, stream: u64) -> Result<Vec<u64>> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, Vec<u64>>>> = OnceLock::new();
    let c = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = c.lock().unwrap().get(&p.0) {
        return Ok(v.clone());
    }
    let mut raw = vec![0u8; n * 8];
    gpu.copy_d2h_on_stream(p, &mut raw, stream)?;
    gpu.synchronize(stream)?;
    let v: Vec<u64> = raw
        .chunks_exact(8)
        .map(|x| u64::from_le_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]]))
        .collect();
    c.lock().unwrap().insert(p.0, v.clone());
    Ok(v)
}

fn cached_f32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize, stream: u64) -> Result<Vec<f32>> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<u64, Vec<f32>>>> = OnceLock::new();
    let c = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = c.lock().unwrap().get(&p.0) {
        return Ok(v.clone());
    }
    let mut raw = vec![0u8; n * 4];
    gpu.copy_d2h_on_stream(p, &mut raw, stream)?;
    gpu.synchronize(stream)?;
    let v: Vec<f32> = raw
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect();
    c.lock().unwrap().insert(p.0, v.clone());
    Ok(v)
}

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
/// Returns the host copy of `expert_offsets` so the paired `down` call can
/// reuse it instead of repeating the D2H + synchronize. The two calls share the
/// same offsets (both are driven by one `moe_sort_by_expert`), and each sync
/// blocks the host until the GPU drains — halving them halves that stall.
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
) -> Result<Vec<i32>> {
    // Static after load — snapshot once, not once per layer per prefill.
    let gate_packed_h = cached_u64(gpu, gate_packed, num_experts, stream)?;
    let gate_sfb_h = cached_u64(gpu, gate_sfb, num_experts, stream)?;
    let up_packed_h = cached_u64(gpu, up_packed, num_experts, stream)?;
    let up_sfb_h = cached_u64(gpu, up_sfb, num_experts, stream)?;
    let gate_scale2_h = cached_f32(gpu, gate_scale2, num_experts, stream)?;
    let up_scale2_h = cached_f32(gpu, up_scale2, num_experts, stream)?;

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
    )?;
    Ok(eoff)
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
    // Host `expert_offsets` from the paired gate_up call. When supplied, the
    // D2H + synchronize here is skipped entirely — the offsets are identical
    // (one sort feeds both projections).
    eoff_cached: Option<&[i32]>,
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
    // Static after load — see cached_u64/cached_f32.
    let packed_h = cached_u64(gpu, packed, num_experts, stream)?;
    let sfb_h = cached_u64(gpu, sfb, num_experts, stream)?;
    let scale2_h = cached_f32(gpu, scale2, num_experts, stream)?;
    // Offsets come from the paired gate_up when available; otherwise fetch them
    // (D2H + the sync that blocks the host until the GPU drains).
    let eoff: Vec<i32> = if let Some(e) = eoff_cached {
        e.to_vec()
    } else {
        let mut off_raw = vec![0u8; (num_experts + 1) * 4];
        gpu.copy_d2h_on_stream(expert_offsets, &mut off_raw, stream)?;
        gpu.synchronize(stream)?;
        off_raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().expect("4")))
            .collect()
    };
    spark_runtime::cutlass::nvfp4_grouped_down(
        a.0, &packed_h, &sfb_h, &scale2_h, c.0, &eoff, hidden, inter, stream,
    )
}
