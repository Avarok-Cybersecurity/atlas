// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Slice a stacked + fused MTP MoE expert layout into per-expert
/// `DenseExpertWeight`s via DevicePtr offsets (zero-copy).
///
/// Expects two BF16 tensors in `store`:
///   `{mlp}.experts.gate_up_proj` shape `[E, 2*I, H]`
///       — first `I` rows of axis 1 are gate, next `I` rows are up
///   `{mlp}.experts.down_proj`    shape `[E, H, I]`
///
/// Each expert's `gate`, `up`, `down` are contiguous sub-tensors of the
/// stacked allocations, so we hand back DenseWeights pointing into the
/// same underlying GPU memory. The WeightStore retains ownership of the
/// stacked allocations; the offset pointers are aliases and must NEVER
/// be passed to `gpu.free()` (the loader doesn't, and ModelWeights drops
/// the WeightStore last in any case).
pub(super) fn load_mtp_experts_stacked(
    store: &WeightStore,
    mlp: &str,
    num_experts: usize,
) -> Result<Vec<DenseExpertWeight>> {
    let gate_up = store.get(&format!("{mlp}.experts.gate_up_proj"))?;
    let down = store.get(&format!("{mlp}.experts.down_proj"))?;

    ensure!(
        gate_up.shape.len() == 3,
        "MTP stacked experts.gate_up_proj: expected 3D [E,2I,H], got {:?}",
        gate_up.shape
    );
    ensure!(
        down.shape.len() == 3,
        "MTP stacked experts.down_proj: expected 3D [E,H,I], got {:?}",
        down.shape
    );
    ensure!(
        gate_up.shape[0] == num_experts,
        "MTP stacked experts.gate_up_proj: expert dim {} != num_experts {num_experts}",
        gate_up.shape[0]
    );
    ensure!(
        down.shape[0] == num_experts,
        "MTP stacked experts.down_proj: expert dim {} != num_experts {num_experts}",
        down.shape[0]
    );

    let two_inter = gate_up.shape[1];
    let hidden = gate_up.shape[2];
    ensure!(
        two_inter % 2 == 0,
        "MTP stacked experts.gate_up_proj: 2nd dim must be even (gate+up fused), got {two_inter}"
    );
    let intermediate = two_inter / 2;

    ensure!(
        down.shape[1] == hidden,
        "MTP stacked: gate_up_proj.hidden ({hidden}) != down_proj.hidden ({})",
        down.shape[1]
    );
    ensure!(
        down.shape[2] == intermediate,
        "MTP stacked: down_proj.intermediate ({}) != gate_up_proj/2 ({intermediate})",
        down.shape[2]
    );

    // Stacked tensors must be BF16 — the per-expert split path also returns
    // BF16 (norm/gate dense or dequanted projections), so we keep the
    // contract uniform downstream.
    ensure!(
        matches!(gate_up.dtype, WeightDtype::BF16),
        "MTP stacked experts.gate_up_proj: expected BF16, got {:?}",
        gate_up.dtype
    );
    ensure!(
        matches!(down.dtype, WeightDtype::BF16),
        "MTP stacked experts.down_proj: expected BF16, got {:?}",
        down.dtype
    );

    let elt = WeightDtype::BF16.byte_size();
    let half_bytes = intermediate * hidden * elt;
    let gate_up_stride = two_inter * hidden * elt;
    let down_stride = hidden * intermediate * elt;

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let base_gu = gate_up.ptr.offset(e * gate_up_stride);
        experts.push(DenseExpertWeight {
            gate_proj: DenseWeight { weight: base_gu },
            up_proj: DenseWeight {
                weight: base_gu.offset(half_bytes),
            },
            down_proj: DenseWeight {
                weight: down.ptr.offset(e * down_stride),
            },
        });
    }
    Ok(experts)
}

// ── Qwen3.5-MoE weight loaders ──
// Two NVFP4 naming conventions exist:
//   Standard (nvidia/txn545):  weight, weight_scale, weight_scale_2, input_scale
//   Sehyo (compressed-tensors): weight_packed, weight_scale, weight_global_scale, input_global_scale
// Additionally, Sehyo quantizes attention/SSM projections; standard keeps them BF16.

/// Weight quantization variant (on-disk format).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nvfp4Variant {
    /// Standard ModelOpt: weight, weight_scale, weight_scale_2, input_scale.
    /// Attention/SSM projections are BF16 dense.
    Standard,
    /// Sehyo/compressed-tensors: weight_packed, weight_global_scale, input_global_scale.
    /// Attention/SSM projections are NVFP4 quantized.
    CompressedTensors,
    /// FP8 block-scaled (e.g. Qwen/Qwen3.5-35B-A3B-FP8):
    /// weight (float8_e4m3fn) + weight_scale_inv (BF16 per-`[128,128]`-block).
    /// All quantized weights get dequanted to BF16 at load time, then runtime-quantized to NVFP4.
    Fp8Dequanted,
    /// Raw BF16/FP16 fine-tunes (e.g. samuelcardillo/Carnice-MoE-35B-A3B):
    /// only `.weight` tensors exist (no quantization metadata). Runtime-quantize
    /// from BF16 to NVFP4 at load time, like Fp8Dequanted but without the FP8
    /// dequant step. Quality is suboptimal vs. a pre-calibrated NVFP4 release —
    /// the user gets a warning at startup.
    Bf16Raw,
}
