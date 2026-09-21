// SPDX-License-Identifier: AGPL-3.0-only

//! K3 tensor-parallel storage plan, shared by host loading and device binding.
//! Config head counts are rank-local; expert and dense widths remain full.

use super::MixerKind;
use crate::config::ModelConfig;
use crate::mxfp4_e8m0::GROUP_SIZE;
use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpAxis {
    Replicated,
    Rows,
    Columns,
}

pub fn tensor_plan(name: &str, mixer: MixerKind, config: &ModelConfig) -> (TpAxis, usize, usize) {
    // Packed weights and E8M0 scales share the logical projection plan.
    let canonical;
    let name = if let Some(stem) = name
        .strip_suffix(".weight_packed")
        .or_else(|| name.strip_suffix(".weight_scale"))
    {
        canonical = format!("{stem}.weight");
        canonical.as_str()
    } else {
        name
    };
    let tp = config.tp_world_size.max(1);
    let h = config.hidden_size;
    let kda_heads = config.linear_num_key_heads * tp;
    let kda_d = config.linear_key_head_dim;
    let kda_q = kda_heads * kda_d;
    let conv_k = config.linear_conv_kernel_dim.max(1);
    let mla_heads = config.num_attention_heads * tp;
    let qk = config.qk_nope_head_dim + config.qk_rope_head_dim;
    let dv = mla_heads * config.v_head_dim;
    let kv_b = mla_heads * (config.qk_nope_head_dim + config.v_head_dim);
    let inter = config.intermediate_size;
    let eh = config.moe_intermediate_size;
    let lat = config.moe_latent_size;

    if name.ends_with(".self_attn.q_proj.weight")
        || name.ends_with(".self_attn.k_proj.weight")
        || name.ends_with(".self_attn.v_proj.weight")
    {
        return (TpAxis::Rows, kda_q, h);
    }
    if name.ends_with(".self_attn.q_conv1d.weight")
        || name.ends_with(".self_attn.k_conv1d.weight")
        || name.ends_with(".self_attn.v_conv1d.weight")
    {
        return (TpAxis::Rows, kda_q, conv_k);
    }
    if name.ends_with(".self_attn.g_proj.weight") {
        let n = match mixer {
            MixerKind::Kda => kda_q,
            MixerKind::Mla => dv,
        };
        return (TpAxis::Rows, n, h);
    }
    if name.ends_with(".self_attn.o_proj.weight") {
        let inn = match mixer {
            MixerKind::Kda => kda_q,
            MixerKind::Mla => dv,
        };
        return (TpAxis::Columns, h, inn);
    }
    if name.ends_with(".self_attn.b_proj.weight") {
        return (TpAxis::Rows, kda_heads, h);
    }
    if name.ends_with(".self_attn.A_log") {
        return (TpAxis::Rows, kda_heads, 1);
    }
    if name.ends_with(".self_attn.dt_bias") {
        return (TpAxis::Rows, kda_q, 1);
    }
    if name.ends_with(".self_attn.f_b_proj.weight") {
        return (TpAxis::Rows, kda_q, kda_d);
    }
    if name.ends_with(".self_attn.q_b_proj.weight") {
        return (TpAxis::Rows, mla_heads * qk, config.q_lora_rank);
    }
    if name.ends_with(".self_attn.kv_b_proj.weight") {
        return (TpAxis::Rows, kv_b, config.kv_lora_rank);
    }
    if name.ends_with(".mlp.gate_proj.weight") || name.ends_with(".mlp.up_proj.weight") {
        return (TpAxis::Rows, inter, h);
    }
    if name.ends_with(".mlp.down_proj.weight") {
        return (TpAxis::Columns, h, inter);
    }
    if name.contains(".block_sparse_moe.experts.") {
        if name.ends_with(".w1.weight") || name.ends_with(".w3.weight") {
            return (TpAxis::Rows, eh, lat);
        }
        if name.ends_with(".w2.weight") {
            return (TpAxis::Columns, lat, eh);
        }
    }
    (TpAxis::Replicated, 1, 1)
}

/// Byte ranges from one full checkpoint tensor that form this rank's tensor.
/// Ranges preserve row-major packed nibbles and their matching E8M0 groups.
#[derive(Debug, Clone)]
pub struct TpBytePlan {
    pub axis: TpAxis,
    pub local_shape: Vec<usize>,
    pub source_bytes: usize,
    offset: usize,
    rows: usize,
    stride: usize,
    row_bytes: usize,
    zero_f32_tail_from: Option<usize>,
}

impl TpBytePlan {
    pub fn local_bytes(&self) -> usize {
        self.rows * self.row_bytes
    }

    pub fn byte_ranges(&self) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
        (0..self.rows).map(|row| {
            let start = self.offset + row * self.stride;
            start..start + self.row_bytes
        })
    }

    /// Check storage-only padding before any device allocations. The official
    /// K3 export stores 96 trained A_log heads followed by 32 zero F32 values.
    /// Keep the logical head count; never truncate an unverified nonzero tail.
    pub fn validate_source(&self, source: &[u8]) -> Result<()> {
        ensure!(
            source.len() == self.source_bytes,
            "K3 TP source bytes {} != {}",
            source.len(),
            self.source_bytes
        );
        if let Some(start) = self.zero_f32_tail_from {
            ensure!(
                source[start..]
                    .chunks_exact(4)
                    .all(|v| f32::from_le_bytes(v.try_into().unwrap()) == 0.0),
                "K3 padded A_log has a nonzero/nonfinite tail; refusing to discard trained values"
            );
        }
        Ok(())
    }

    /// Gather only the local bytes from an mmap/read-only checkpoint view.
    pub fn copy_shard(&self, source: &[u8]) -> Result<Vec<u8>> {
        self.validate_source(source)?;
        let mut local = Vec::with_capacity(self.local_bytes());
        for range in self.byte_ranges() {
            local.extend_from_slice(&source[range]);
        }
        Ok(local)
    }
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |n, &v| {
        ensure!(v > 0, "K3 TP zero dimension/element width");
        n.checked_mul(v)
            .ok_or_else(|| anyhow::anyhow!("K3 TP byte-size overflow"))
    })
}

/// Plan checkpoint storage BEFORE allocating a device buffer.
///
/// BF16/F32 keep their dtype. Packed experts require `[N, K/2]` bytes plus
/// `[N, K/32]` E8M0 scales; a column split cannot bisect a 32-value group.
/// Unknown non-packed tensors remain replicated. Runtime must separately
/// enforce the model's tensor inventory and exclude unsupported modalities.
pub fn plan_tensor_bytes(
    name: &str,
    shape: &[usize],
    element_bytes: usize,
    mixer: MixerKind,
    config: &ModelConfig,
) -> Result<TpBytePlan> {
    let world = config.tp_world_size;
    let rank = config.tp_rank;
    ensure!(
        world > 0 && rank < world,
        "K3 TP rank {rank} invalid for world {world}"
    );
    ensure!(
        !shape.is_empty(),
        "{name}: K3 TP requires a non-scalar tensor"
    );
    let elements = product(shape)?;
    let source_bytes = product(&[elements, element_bytes])?;
    let (axis, n, logical_k) = tensor_plan(name, mixer, config);
    // Verified export exception, not a general shape relaxation. See HF
    // moonshotai/Kimi-K3 discussion #150 and NVIDIA Megatron Bridge K3 docs.
    if name.ends_with(".self_attn.A_log")
        && mixer == MixerKind::Kda
        && n == 96
        && config.linear_key_head_dim == 128
        && shape == [128]
        && element_bytes == 4
    {
        ensure!(
            n.is_multiple_of(world),
            "K3 A_log heads must divide TP world"
        );
        let row_bytes = n / world * element_bytes;
        return Ok(TpBytePlan {
            axis,
            local_shape: vec![n / world],
            source_bytes,
            offset: rank * row_bytes,
            rows: 1,
            stride: row_bytes,
            row_bytes,
            zero_f32_tail_from: Some(n * element_bytes),
        });
    }
    let packed = name.ends_with(".weight_packed");
    let scale = name.ends_with(".weight_scale");
    let storage_k = if packed || scale {
        ensure!(
            name.contains(".block_sparse_moe.experts.") && axis != TpAxis::Replicated,
            "{name}: unsupported K3 packed projection"
        );
        ensure!(
            element_bytes == 1,
            "{name}: packed MXFP4/E8M0 storage must be one byte"
        );
        ensure!(
            logical_k.is_multiple_of(GROUP_SIZE),
            "{name}: K {logical_k} must align to E8M0 groups of 32"
        );
        if axis == TpAxis::Columns {
            ensure!(
                logical_k.is_multiple_of(world) && (logical_k / world).is_multiple_of(GROUP_SIZE),
                "{name}: TP{world} column partition splits an E8M0 group of 32"
            );
        }
        let divisor = if packed { 2 } else { GROUP_SIZE };
        let k = logical_k / divisor;
        ensure!(
            shape == [n, k],
            "{name}: packed shape {shape:?} != [{n}, {k}]"
        );
        k
    } else {
        if axis != TpAxis::Replicated {
            ensure!(
                matches!(element_bytes, 2 | 4),
                "{name}: K3 dense TP requires BF16 or FP32"
            );
            ensure!(
                shape[0] == n && elements == product(&[n, logical_k])?,
                "{name}: dense shape {shape:?} != logical [{n}, {logical_k}]"
            );
        }
        logical_k
    };
    let mut local_shape = shape.to_vec();
    let (offset, rows, stride, row_bytes) = match axis {
        TpAxis::Replicated => (0, 1, source_bytes, source_bytes),
        TpAxis::Rows => {
            ensure!(
                n.is_multiple_of(world),
                "{name}: rows {n} not divisible by TP{world}"
            );
            local_shape[0] /= world;
            let row_bytes = source_bytes / world;
            (rank * row_bytes, 1, row_bytes, row_bytes)
        }
        TpAxis::Columns => {
            ensure!(
                shape.len() == 2 && storage_k.is_multiple_of(world),
                "{name}: columns {storage_k} not divisible by TP{world}"
            );
            local_shape[1] /= world;
            let stride = product(&[storage_k, element_bytes])?;
            let row_bytes = stride / world;
            (rank * row_bytes, n, stride, row_bytes)
        }
    };
    ensure!(
        product(&local_shape)? * element_bytes == rows * row_bytes,
        "{name}: local TP byte accounting mismatch"
    );
    Ok(TpBytePlan {
        axis,
        local_shape,
        source_bytes,
        offset,
        rows,
        stride,
        row_bytes,
        zero_f32_tail_from: None,
    })
}

#[cfg(test)]
mod tests;
