// SPDX-License-Identifier: AGPL-3.0-only
//! Kimi-K3 GGUF keep-packed upload: TP-slice before `gpu.alloc`.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};

use super::dequant_cpu::{self, GgmlType};
use super::kimi_tp_slice::{self, PackKind};
use super::names;
use super::GgufLoader;
use crate::gpu::GpuBackend;
use crate::weights::{WeightDtype, WeightTensor};

fn kimi_rows(name: &str) -> bool {
    name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("g_proj")
        || name.contains("q_b_proj")
        || name.contains("kv_b_proj")
        || name.contains("v_b_proj")
        || name.contains("gate_proj")
        || name.contains("up_proj")
        || name.contains(".w1.weight")
        || name.contains(".w3.weight")
        || name.contains("b_proj")
        || name.contains("f_b_proj")
}

fn kimi_cols(name: &str) -> bool {
    name.contains("o_proj")
        || name.contains("down_proj")
        || name.contains(".w2.weight")
}

fn dtype_for_id(id: u32) -> WeightDtype {
    match id {
        8 => WeightDtype::Q8_0,
        10 => WeightDtype::Q2K,
        11 => WeightDtype::Q3K,
        17 => WeightDtype::Iq2Xs,
        18 => WeightDtype::Iq3Xxs,
        _ => WeightDtype::Q8_0,
    }
}

fn pack_kind(id: u32) -> Result<PackKind> {
    kimi_tp_slice::pack_kind_from_ggml(id)
}

fn q8_to_bf16_bytes(raw: &[u8], n: usize, k: usize) -> Result<Vec<u8>> {
    ensure!(k % 32 == 0, "Q8_0 K={k} not multiple of 32");
    let mut bits = vec![0u16; n * k];
    dequant_cpu::dequant_to_bf16(GgmlType::Q8_0, raw, n * k, &mut bits)?;
    let mut out = Vec::with_capacity(bits.len() * 2);
    for b in bits {
        out.extend_from_slice(&b.to_le_bytes());
    }
    Ok(out)
}

/// Keep-packed Direct tensor: TP-slice, Q8→BF16, upload.
pub(super) fn upload_direct_packed(
    loader: &GgufLoader,
    gpu: &dyn GpuBackend,
    hf_name: &str,
    raw: &[u8],
    hf_shape: &[usize],
    id: u32,
    weights: &mut HashMap<String, WeightTensor>,
) -> Result<()> {
    let tp = loader.tp_world_size.max(1);
    let rank = loader.tp_rank;
    let mut shape = hf_shape.to_vec();
    let kind = if matches!(id, 8 | 17 | 18) {
        Some(pack_kind(id)?)
    } else {
        None
    };

    let upload: Vec<u8> = if tp > 1 && shape.len() == 2 {
        let (n, k) = (shape[0], shape[1]);
        if let Some(pk) = kind {
            if kimi_rows(hf_name) {
                let (bytes, ln, lk) = kimi_tp_slice::slice_rows_packed(raw, n, k, rank, tp, pk)?;
                shape = vec![ln, lk];
                bytes
            } else if kimi_cols(hf_name) {
                let (bytes, ln, lk) = kimi_tp_slice::slice_cols_packed(raw, n, k, rank, tp, pk)?;
                shape = vec![ln, lk];
                bytes
            } else {
                raw.to_vec()
            }
        } else if kimi_rows(hf_name) && n.is_multiple_of(tp) && raw.len().is_multiple_of(n) {
            let local_rows = n / tp;
            let row_bytes = raw.len() / n;
            let start = rank * local_rows * row_bytes;
            shape = vec![local_rows, k];
            raw[start..start + local_rows * row_bytes].to_vec()
        } else {
            raw.to_vec()
        }
    } else {
        raw.to_vec()
    };

    let (ptr, dtype, final_shape) = if id == 8 {
        let k = if shape.len() >= 2 {
            shape[1]
        } else {
            shape[0]
        };
        let n = if shape.len() >= 2 { shape[0] } else { 1 };
        let bf = q8_to_bf16_bytes(&upload, n, k)?;
        let ptr = gpu.alloc(bf.len())?;
        gpu.copy_h2d(&bf, ptr)?;
        (ptr, WeightDtype::BF16, shape)
    } else {
        let dtype = dtype_for_id(id);
        let ptr = gpu.alloc(upload.len())?;
        gpu.copy_h2d(&upload, ptr)?;
        (ptr, dtype, shape)
    };

    weights.insert(
        hf_name.to_string(),
        WeightTensor {
            ptr,
            shape: final_shape.clone(),
            dtype,
        },
    );
    if let Some(stem) = hf_name.strip_suffix("_res_proj.weight") {
        weights.insert(
            format!("{stem}_res_norm.weight"),
            WeightTensor {
                ptr,
                shape: final_shape,
                dtype,
            },
        );
    } else if hf_name.ends_with("output_attn_res_proj.weight") {
        let stem = &hf_name[..hf_name.len() - "output_attn_res_proj.weight".len()];
        weights.insert(
            format!("{stem}output_attn_res_norm.weight"),
            WeightTensor {
                ptr,
                shape: final_shape,
                dtype,
            },
        );
    }
    Ok(())
}

/// Upload every routed expert, TP-sliced, one alloc per expert.
pub(super) fn upload_expert_stack(
    loader: &GgufLoader,
    gpu: &dyn GpuBackend,
    tensor_name: &str,
    raw: &[u8],
    dims: &[usize],
    id: u32,
    weights: &mut HashMap<String, WeightTensor>,
) -> Result<()> {
    let mut hf_shape: Vec<usize> = dims.to_vec();
    hf_shape.reverse();
    let count = *hf_shape
        .first()
        .context("kimi expert stack missing leading expert dim")?;
    ensure!(hf_shape.len() == 3, "{tensor_name}: want [E, out, in], got {hf_shape:?}");
    let out = hf_shape[1];
    let inn = hf_shape[2];
    let dtype = dtype_for_id(id);
    let pk = pack_kind(id)?;
    let per_elems = out * inn;
    let per_bytes = (per_elems / pk.qk()) * pk.block_bytes();
    ensure!(
        raw.len() >= count * per_bytes,
        "{tensor_name}: raw {} < {} experts * {per_bytes}",
        raw.len(),
        count
    );

    let proj = if tensor_name.ends_with("ffn_gate_exps.weight") {
        "gate"
    } else if tensor_name.ends_with("ffn_up_exps.weight") {
        "up"
    } else {
        "down"
    };
    let layer: usize = tensor_name
        .split('.')
        .nth(1)
        .context("kimi expert layer")?
        .parse()
        .context("kimi expert layer id")?;

    let tp = loader.tp_world_size.max(1);
    let rank = loader.tp_rank;
    let rows = proj != "down";

    for e in 0..count {
        let expert_raw = &raw[e * per_bytes..(e + 1) * per_bytes];
        let (upload, local_out, local_in) = if tp <= 1 {
            (expert_raw.to_vec(), out, inn)
        } else if rows {
            kimi_tp_slice::slice_rows_packed(expert_raw, out, inn, rank, tp, pk)?
        } else {
            kimi_tp_slice::slice_cols_packed(expert_raw, out, inn, rank, tp, pk)?
        };
        let ptr = gpu.alloc(upload.len())?;
        gpu.copy_h2d(&upload, ptr)?;
        let name = names::kimi_k3_expert_name(layer, proj, e);
        weights.insert(
            name,
            WeightTensor {
                ptr,
                shape: vec![local_out, local_in],
                dtype,
            },
        );
    }
    Ok(())
}

pub(super) fn estimate_resident_bytes(tp_world: usize) -> usize {
    let tp = tp_world.max(1);
    let disk = 802usize * 1024 * 1024 * 1024;
    disk / tp + 12usize * 1024 * 1024 * 1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_and_cols_detect() {
        assert!(kimi_rows("model.layers.0.self_attn.q_proj.weight"));
        assert!(kimi_cols("model.layers.0.self_attn.o_proj.weight"));
        assert!(kimi_rows(
            "model.layers.1.block_sparse_moe.experts.0.w1.weight"
        ));
        assert!(kimi_cols(
            "model.layers.1.block_sparse_moe.experts.0.w2.weight"
        ));
    }
}
