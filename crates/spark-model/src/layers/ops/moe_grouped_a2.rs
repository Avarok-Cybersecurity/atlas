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

/// Host snapshots of the per-expert pointer/scale tables for the CUTLASS
/// grouped path, owned by the `MoeLayer` whose device tables they mirror.
///
/// The CUTLASS grouped entry needs these on the HOST to build its per-group
/// problem shapes. They are immutable once `build_cutlass_grouped_sfb` has
/// run, so re-reading them per call was 6 copies x 2 calls x 47 MoE layers =
/// 564 pointless D2H transfers per prefill. Only `expert_offsets` genuinely
/// changes per call (it is produced by the expert sort), so only that one is
/// still copied at dispatch time.
///
/// Why a layer-owned snapshot and not a process-global cache keyed on the
/// device pointer: an address is not an identity. An in-process model swap
/// (`model_swap::swap`) tears the outgoing model down — `cuMemFree_v2` on
/// every table cached here — and the incoming load's near-identical
/// `cuMemAlloc_v2` sequence reuses those virtual addresses. A pointer-keyed
/// static then hands the NEW model the OLD model's expert weight pointers,
/// and the grouped GEMM silently reads whatever now lives there as weights.
/// Owning the snapshot on the layer makes staleness structurally impossible:
/// the snapshot dies with the layer, with the model, at teardown.
pub struct MoeCutlassHostTables {
    pub gate_packed: Vec<u64>,
    pub gate_sfb: Vec<u64>,
    pub gate_scale2: Vec<f32>,
    pub up_packed: Vec<u64>,
    pub up_sfb: Vec<u64>,
    pub up_scale2: Vec<f32>,
    /// `None` when the checkpoint has no down-projection scale table — the
    /// grouped down branch is unreachable in that case.
    pub down: Option<MoeCutlassDownHostTables>,
}

/// Down-projection third of [`MoeCutlassHostTables`].
pub struct MoeCutlassDownHostTables {
    pub packed: Vec<u64>,
    pub sfb: Vec<u64>,
    pub scale2: Vec<f32>,
}

/// One blocking D2H of a device `[n]` u64 pointer table. Load-time only.
pub fn read_expert_ptrs_u64(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u64>> {
    let mut raw = vec![0u8; n * 8];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(8)
        .map(|x| u64::from_le_bytes(x.try_into().expect("8")))
        .collect())
}

/// One blocking D2H of a device `[n]` f32 scale table. Load-time only.
pub fn read_expert_scales_f32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut raw = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().expect("4")))
        .collect())
}

impl MoeCutlassHostTables {
    /// Snapshot the grouped-path tables at load. The SFB pointer vectors are
    /// taken by value because `build_cutlass_grouped_sfb` constructs them on
    /// the host in the first place — reading them back from the device would
    /// re-derive data this function's caller already holds. The packed/scale2
    /// tables exist only on the device (uploaded by the pointer-table build),
    /// so those are copied down once here.
    #[allow(clippy::too_many_arguments)]
    pub fn snapshot(
        gpu: &dyn GpuBackend,
        num_experts: usize,
        gate_packed: DevicePtr,
        gate_sfb: Vec<u64>,
        gate_scale2: DevicePtr,
        up_packed: DevicePtr,
        up_sfb: Vec<u64>,
        up_scale2: DevicePtr,
        down: Option<(DevicePtr, Vec<u64>, DevicePtr)>,
    ) -> Result<Self> {
        Ok(Self {
            gate_packed: read_expert_ptrs_u64(gpu, gate_packed, num_experts)?,
            gate_sfb,
            gate_scale2: read_expert_scales_f32(gpu, gate_scale2, num_experts)?,
            up_packed: read_expert_ptrs_u64(gpu, up_packed, num_experts)?,
            up_sfb,
            up_scale2: read_expert_scales_f32(gpu, up_scale2, num_experts)?,
            down: match down {
                Some((packed, sfb, scale2)) => Some(MoeCutlassDownHostTables {
                    packed: read_expert_ptrs_u64(gpu, packed, num_experts)?,
                    sfb,
                    scale2: read_expert_scales_f32(gpu, scale2, num_experts)?,
                }),
                None => None,
            },
        })
    }
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
/// Bridges the load-time host snapshot of the per-expert pointer/scale tables
/// ([`MoeCutlassHostTables`]) to the host-side
/// [`spark_runtime::cutlass::nvfp4_grouped_gate_up_fused`] entry. `a` is the
/// expert-contiguous bf16 activation `[total_expanded, k]`; `expert_offsets`
/// is the device i32 `[num_experts+1]` prefix sum — the only per-call table,
/// so the only one copied and the only reason for the synchronize (the C
/// entry indexes offsets on the host before it can launch).
#[allow(clippy::too_many_arguments)]
/// Returns the host copy of `expert_offsets` so the paired `down` call can
/// reuse it instead of repeating the D2H + synchronize. The two calls share the
/// same offsets (both are driven by one `moe_sort_by_expert`), and each sync
/// blocks the host until the GPU drains — halving them halves that stall.
pub fn moe_grouped_gate_up_cutlass(
    gpu: &dyn GpuBackend,
    host: &MoeCutlassHostTables,
    a: DevicePtr,
    sorted_token_ids: DevicePtr,
    c_gate: DevicePtr,
    c_up: DevicePtr,
    expert_offsets: DevicePtr,
    inter: u32,
    hidden: u32,
    stream: u64,
) -> Result<Vec<i32>> {
    let num_experts = host.gate_packed.len();
    let mut off_raw = vec![0u8; (num_experts + 1) * 4];
    gpu.copy_d2h_on_stream(expert_offsets, &mut off_raw, stream)?;
    // The offsets host copy is needed by the C entry before it can launch —
    // make sure the async D2H has landed.
    gpu.synchronize(stream)?;
    let eoff: Vec<i32> = off_raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    spark_runtime::cutlass::nvfp4_grouped_gate_up_fused(
        a.0,
        sorted_token_ids.0,
        &host.gate_packed,
        &host.gate_sfb,
        &host.gate_scale2,
        &host.up_packed,
        &host.up_sfb,
        &host.up_scale2,
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
/// intermediate `[total_expanded, inter]` (already expert-contiguous — no
/// gather). `host` is the down third of the load-time snapshot;
/// `expert_offsets` is the device i32 `[num_experts+1]` prefix sum.
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_down_cutlass(
    gpu: &dyn GpuBackend,
    // Host `expert_offsets` from the paired gate_up call. When supplied, the
    // D2H + synchronize here is skipped entirely — the offsets are identical
    // (one sort feeds both projections).
    eoff_cached: Option<&[i32]>,
    host: &MoeCutlassDownHostTables,
    a: DevicePtr,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    hidden: u32,
    inter: u32,
    stream: u64,
) -> Result<()> {
    let num_experts = host.packed.len();
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
        a.0,
        &host.packed,
        &host.sfb,
        &host.scale2,
        c.0,
        &eoff,
        hidden,
        inter,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn upload_u64(gpu: &MockGpuBackend, vals: &[u64]) -> DevicePtr {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let p = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(&bytes, p).unwrap();
        p
    }

    fn upload_f32(gpu: &MockGpuBackend, vals: &[f32]) -> DevicePtr {
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let p = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(&bytes, p).unwrap();
        p
    }

    /// The failure this pins: the previous implementation memoized these reads
    /// in a process-global map keyed on the device ADDRESS. After an
    /// in-process model swap the incoming load reuses the outgoing model's
    /// freed virtual addresses, and the pointer-keyed cache handed the new
    /// model the OLD model's expert weight pointers — the grouped GEMM then
    /// silently read unrelated tensors as weights. A read must reflect what
    /// the device holds NOW, so writing new contents to the same address and
    /// reading again must observe the new contents.
    #[test]
    fn read_reflects_current_device_contents_at_a_reused_address() {
        let gpu = MockGpuBackend::new();
        let old_model = [0x1111_u64, 0x2222, 0x3333];
        let p = upload_u64(&gpu, &old_model);
        assert_eq!(read_expert_ptrs_u64(&gpu, p, 3).unwrap(), old_model);

        // Same address, new contents — the swap's free/realloc collapsed to
        // its essence (the mock never moves an allocation, which is exactly
        // the driver's common case for identical alloc sequences).
        let new_model = [0xaaaa_u64, 0xbbbb, 0xcccc];
        let bytes: Vec<u8> = new_model.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d(&bytes, p).unwrap();
        assert_eq!(read_expert_ptrs_u64(&gpu, p, 3).unwrap(), new_model);

        let old_scales = [1.0_f32, 2.0];
        let ps = upload_f32(&gpu, &old_scales);
        assert_eq!(read_expert_scales_f32(&gpu, ps, 2).unwrap(), old_scales);
        let new_scales = [3.0_f32, 4.0];
        let sbytes: Vec<u8> = new_scales.iter().flat_map(|v| v.to_le_bytes()).collect();
        gpu.copy_h2d(&sbytes, ps).unwrap();
        assert_eq!(read_expert_scales_f32(&gpu, ps, 2).unwrap(), new_scales);
    }

    /// The old cache also ignored the requested length: a hit returned the
    /// first query's vector whatever `n` the caller asked for. A read of `n`
    /// elements must return exactly `n` elements.
    #[test]
    fn read_honors_the_requested_length() {
        let gpu = MockGpuBackend::new();
        let vals = [1_u64, 2, 3, 4];
        let p = upload_u64(&gpu, &vals);
        assert_eq!(read_expert_ptrs_u64(&gpu, p, 2).unwrap(), vals[..2]);
        assert_eq!(read_expert_ptrs_u64(&gpu, p, 4).unwrap(), vals);
    }

    /// Nine same-typed tables flow into `snapshot`; a transposition would
    /// type-check and quantize with the wrong expert scales. Pin each field
    /// to its source.
    #[test]
    fn snapshot_maps_every_table_to_its_field() {
        let gpu = MockGpuBackend::new();
        let n = 2;
        let gate_packed = upload_u64(&gpu, &[10, 11]);
        let gate_scale2 = upload_f32(&gpu, &[0.5, 0.25]);
        let up_packed = upload_u64(&gpu, &[20, 21]);
        let up_scale2 = upload_f32(&gpu, &[2.0, 4.0]);
        let down_packed = upload_u64(&gpu, &[30, 31]);
        let down_scale2 = upload_f32(&gpu, &[8.0, 16.0]);

        let t = MoeCutlassHostTables::snapshot(
            &gpu,
            n,
            gate_packed,
            vec![100, 101],
            gate_scale2,
            up_packed,
            vec![200, 201],
            up_scale2,
            Some((down_packed, vec![300, 301], down_scale2)),
        )
        .unwrap();

        assert_eq!(t.gate_packed, [10, 11]);
        assert_eq!(t.gate_sfb, [100, 101]);
        assert_eq!(t.gate_scale2, [0.5, 0.25]);
        assert_eq!(t.up_packed, [20, 21]);
        assert_eq!(t.up_sfb, [200, 201]);
        assert_eq!(t.up_scale2, [2.0, 4.0]);
        let d = t.down.expect("down tables were supplied");
        assert_eq!(d.packed, [30, 31]);
        assert_eq!(d.sfb, [300, 301]);
        assert_eq!(d.scale2, [8.0, 16.0]);
    }
}
