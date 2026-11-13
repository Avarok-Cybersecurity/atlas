// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the dense FFN of a checkpoint whose 4-bit tensors the weight store keeps PACKED (GGUF
//! IQ4_XS, Q4_K and Q6_K, kept packed by the GGUF loader under lane L19's `gguf_native_iq4` row): which
//! layers carry them, the refusals, and the `W4Linear`s that serve them (`port::gguf_q4`, borrowing the
//! store's buffers, so the install copies and frees nothing).
//!
//! **WHY HERE.** Without this, a dense layer whose gate, up or down is a packed tensor reaches the BF16
//! (`dense_auto`) or NVFP4 (`load_dense_ffn`) arm, neither of which can read GGUF super-blocks. The loader
//! asks [`ffn_any_packed`] before it builds either, and a layer that answers yes is built with a NULL NVFP4
//! fallback and no BF16 overlay, then served by `DenseFfnLayer::set_w4` (lane L21's dispatch).
//!
//! **NO ROW OF ITS OWN.** Packed tensors exist in the store only when the GGUF loader kept them packed
//! (L19's row, off on every target); with it off every tensor arrives as BF16, [`ffn_any_packed`] is false
//! for every layer, and the loader is byte-identical.
//!
//! **SCOPE.** The dense FFN (three plain row-major `[n, k]` super-block rows) and the full-attention
//! q/k/v/o, which lane L19's loader keeps packed as transform-free rows (`is_keep_packed_proj`; the
//! attention layer is built with NULL slots like the keep-packed Q2 arm and served by `set_w4_proj`).
//! The GDN input projections are also kept packed by the GGUF loader (with the value-head row reorder
//! already applied, L19's R1), but the GDN layer takes a PAIR (`in_proj_qkvz`, `out_proj`) and `out_proj`
//! stays BF16 (its column reorder is not a row copy), so a GDN layer with packed inputs is refused by name
//! ([`gdn_packed_refusal`]); the row is not usable on a file whose GDN projections are packed until that
//! pair has an install (open item, queue row Q14). GPTQ and EXL3 checkpoints are not readable into the
//! store today (the store has no I32 dtype, so a GPTQ `qweight` cannot enter it; the EXL3 reader has no
//! store arm), so `port::marlin::from_gptq` / `from_awq` have no install site here (lane L18b's R3, open).

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::weights::{WeightStore, WeightTensor};

use crate::port::w4::W4Box;

/// The FFN projections, in the install order (`DenseFfnLayer::set_w4(gate, up, down)`).
pub const FFN_NAMES: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// The store dtypes this module serves packed.
pub fn is_packed(t: &WeightTensor) -> bool {
    crate::port::gguf_q4::q4_type_of_dtype(t.dtype).is_some()
}

fn tensor_name(lp: &str, proj: &str) -> String {
    format!("{lp}.mlp.{proj}.weight")
}

/// Any of the layer's three FFN projections is a packed tensor. The loader must then not read them as
/// BF16 or NVFP4; [`ffn_boxes`] refuses a layer that is only partly packed.
pub fn ffn_any_packed(store: &WeightStore, lp: &str) -> bool {
    FFN_NAMES
        .iter()
        .any(|p| store.get(&tensor_name(lp, p)).is_ok_and(is_packed))
}

/// `[n, k]` of gate, up and down.
pub fn ffn_shapes(hidden: usize, inter: usize) -> [[usize; 2]; 3] {
    [[inter, hidden], [inter, hidden], [hidden, inter]]
}

/// The layer's three packed FFN weights, built by `build` (production: `port::gguf_q4::from_weight_tensor`)
/// in the order gate, up, down. Refused, with the reason, when the load is tensor parallel (packed blocks
/// have no shard path), a LoRA adapter is pending (the adapter install has no packed arm), a projection is
/// not packed (a partly packed layer has no serving path), a shape differs from the config's, or the
/// builder declines a tensor. Every refusal is a load error: nothing else can serve these tensors.
#[allow(clippy::too_many_arguments)]
pub fn ffn_boxes(
    store: &WeightStore,
    lp: &str,
    hidden: usize,
    inter: usize,
    tp_world_size: usize,
    lora: bool,
    mut build: impl FnMut(&WeightTensor) -> Result<Option<W4Box>>,
) -> Result<[W4Box; 3]> {
    ensure!(
        tp_world_size.max(1) == 1,
        "{lp}: keep-packed GGUF FFN tensors cannot be sharded (tensor parallel {tp_world_size}); \
         serve on one GPU, or load the checkpoint without keep-packed tensors"
    );
    ensure!(
        !lora,
        "{lp}: a LoRA adapter cannot be applied to a keep-packed GGUF FFN; load the checkpoint without \
         keep-packed tensors to use an adapter"
    );
    let mut out: Vec<W4Box> = Vec::with_capacity(3);
    for (proj, shape) in FFN_NAMES.iter().zip(ffn_shapes(hidden, inter)) {
        let name = tensor_name(lp, proj);
        let t = store
            .get(&name)
            .with_context(|| format!("keep-packed FFN: {name}"))?;
        if !is_packed(t) {
            bail!(
                "{name} is {:?} while another projection of the layer is kept packed; a partly packed FFN \
                 has no serving path",
                t.dtype
            );
        }
        ensure!(
            t.shape == shape,
            "{name}: packed shape {:?}, the config needs {shape:?}",
            t.shape
        );
        let w =
            build(t)?.with_context(|| format!("{name}: no 4-bit kernel serves {:?}", t.dtype))?;
        ensure!(
            (w.n(), w.k()) == (shape[0], shape[1]),
            "{name}: the 4-bit weight is [{}x{}], the config needs {shape:?}",
            w.n(),
            w.k()
        );
        out.push(w);
    }
    let three: [W4Box; 3] = out
        .try_into()
        .map_err(|_| anyhow::anyhow!("{lp}: three packed FFN weights expected"))?;
    Ok(three)
}

/// The full-attention projections, in the install order (`Qwen3AttentionLayer::set_w4_proj(q, k, v, o)`).
pub const ATTN_NAMES: [&str; 4] = ["q_proj", "k_proj", "v_proj", "o_proj"];

/// The GDN input projections the keep-packed route tags (`in_proj_qkv` and `in_proj_z`).
pub const GDN_NAMES: [&str; 2] = ["in_proj_qkv", "in_proj_z"];

fn proj_tensor_name(prefix: &str, proj: &str) -> String {
    format!("{prefix}.{proj}.weight")
}

/// Any of the layer's q/k/v/o (`p` is `<layer>.self_attn`) is a packed tensor. The loader must then not
/// read them as BF16 or NVFP4; [`attn_boxes`] refuses a layer that is only partly packed.
pub fn attn_any_packed(store: &WeightStore, p: &str) -> bool {
    ATTN_NAMES
        .iter()
        .any(|n| store.get(&proj_tensor_name(p, n)).is_ok_and(is_packed))
}

/// Any of the GDN layer's `in_proj_qkv` / `in_proj_z` (`la` is `<layer>.linear_attn`) is a packed tensor.
///
/// There is no serving path for these: the pair a GDN layer takes (`Qwen3SsmLayer::set_w4_proj`) needs a
/// packed `out_proj` too, and the GGUF converter reorders `out_proj` within each row (a column permutation
/// by value head), which the loader does not keep packed. The loader refuses the layer with
/// [`gdn_packed_refusal`] instead of failing inside a BF16 read.
pub fn gdn_any_packed(store: &WeightStore, la: &str) -> bool {
    GDN_NAMES
        .iter()
        .any(|n| store.get(&proj_tensor_name(la, n)).is_ok_and(is_packed))
}

/// The load error for a keep-packed GDN projection.
pub fn gdn_packed_refusal(la: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "{la}: the checkpoint keeps in_proj_qkv / in_proj_z packed (GGUF IQ4_XS / Q4_K / Q6_K), and the \
         GDN layer has no packed install (its out_proj is not kept packed, and the pair needs both); \
         load the GGUF without keep-packed 4-bit tensors (AVAROK_GGUF_NATIVE_IQ4=0)"
    )
}

/// The layer's four packed attention weights, built by `build` in the order q, k, v, o. `shapes` is
/// `[n, k]` of each (`AttnShapes::projections`). Refused, with the reason, on the same grounds as
/// [`ffn_boxes`]: tensor parallel, a pending LoRA adapter, a partly packed layer, a shape that differs from
/// the config's, or a builder that declines a tensor. Every refusal is a load error.
pub fn attn_boxes(
    store: &WeightStore,
    p: &str,
    shapes: [(usize, usize); 4],
    tp_world_size: usize,
    lora: bool,
    mut build: impl FnMut(&WeightTensor) -> Result<Option<W4Box>>,
) -> Result<[W4Box; 4]> {
    ensure!(
        tp_world_size.max(1) == 1,
        "{p}: keep-packed GGUF attention tensors cannot be sharded (tensor parallel {tp_world_size}); \
         serve on one GPU, or load the checkpoint without keep-packed tensors"
    );
    ensure!(
        !lora,
        "{p}: a LoRA adapter cannot be applied to keep-packed GGUF attention weights; load the checkpoint \
         without keep-packed tensors to use an adapter"
    );
    let mut out: Vec<W4Box> = Vec::with_capacity(4);
    for (proj, (n, k)) in ATTN_NAMES.iter().zip(shapes) {
        let name = proj_tensor_name(p, proj);
        let t = store
            .get(&name)
            .with_context(|| format!("keep-packed attention: {name}"))?;
        if !is_packed(t) {
            bail!(
                "{name} is {:?} while another projection of the layer is kept packed; a partly packed \
                 attention layer has no serving path",
                t.dtype
            );
        }
        ensure!(
            t.shape == [n, k],
            "{name}: packed shape {:?}, the config needs {:?}",
            t.shape,
            [n, k]
        );
        let w =
            build(t)?.with_context(|| format!("{name}: no 4-bit kernel serves {:?}", t.dtype))?;
        ensure!(
            (w.n(), w.k()) == (n, k),
            "{name}: the 4-bit weight is [{}x{}], the config needs [{n}x{k}]",
            w.n(),
            w.k()
        );
        out.push(w);
    }
    let four: [W4Box; 4] = out
        .try_into()
        .map_err(|_| anyhow::anyhow!("{p}: four packed attention weights expected"))?;
    Ok(four)
}

#[cfg(test)]
#[path = "packed_install_tests.rs"]
mod tests;
