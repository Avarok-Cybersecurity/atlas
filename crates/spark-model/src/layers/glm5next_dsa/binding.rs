// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 DSA checkpoint binding: the 14 `self_attn` tensors of a DSA block,
//! their expected dtype and shape, and a verifier that fails loudly.
//!
//! Scoped to `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`. Mirrors
//! [`crate::layers::glm5next_kda::binding`] and deliberately reuses its
//! [`RawTensor`] / [`TensorSource`] / dtype types instead of growing a parallel set.
//!
//! Shapes are expressed against [`Glm5NextDsaConfig`], so a geometry change fails
//! here rather than at kernel launch — the same contract the KDA binder holds.
//!
//! # 🪤 What this exists to catch
//!
//! * **`indexer.k_norm` has a `bias`.** It is an `nn.LayerNorm`, not an RMSNorm —
//!   the only other norm in GLM-5.3 with a bias. A binder that loads only
//!   `.weight` drops mean-subtraction *and* the bias, and nothing about the shapes
//!   says so. The spec table lists the bias as REQUIRED so its absence is an error,
//!   not a silent zero.
//! * **`index_kpool_compress_ape` is `[kpool, index_head_dim]`** — indexed by pool
//!   SLOT, not by head and not by pool. Its 1 KB size makes a wrong-axis bind easy
//!   to miss.
//! * **`q_b_proj` and `kv_b_proj` carry different per-head widths** —
//!   `qk_head_dim` (256) vs `nope + v_head_dim` (512). Both are `[heads * w, lora]`,
//!   so a swapped width still yields a well-formed 2-D tensor.
//! * **NoPE**: there is no `wkv_a_rope` / `wq_b_rope` here at all, and
//!   `kv_a_proj_with_mqa` is `kv_lora_rank` wide (512), not `+ rope` (576).
//!   A spec that expects 576 rejects the real checkpoint.

use std::collections::BTreeSet;

use anyhow::{Result, bail};

use super::Glm5NextDsaConfig;
use crate::layers::glm5next_kda::binding::{KdaDtype as Dtype, KdaTensorSource as TensorSource};

/// One expected tensor: layer-relative name, dtype, and full (unsharded) shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaSpec {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

/// Every `self_attn` tensor a DSA block has — and the complete list of what it may
/// have. `full_heads` is the pre-shard head count; the checkpoint is never sharded
/// on disk, so binding always validates against the full geometry and slices after.
pub fn dsa_tensor_specs(cfg: &Glm5NextDsaConfig, full_heads: usize) -> Vec<DsaSpec> {
    let x = cfg.hidden;
    let ql = cfg.q_lora_rank;
    let kvl = cfg.kv_lora_rank;
    let qk = cfg.qk_head_dim();
    let kvb = cfg.qk_nope_head_dim + cfg.v_head_dim;
    let ihd = cfg.index_head_dim;
    let ih = cfg.index_heads;

    let s = |name: &str, shape: Vec<usize>| DsaSpec {
        name: name.to_string(),
        dtype: Dtype::Bf16,
        shape,
    };

    vec![
        // ── MLA ──
        s("self_attn.q_a_proj.weight", vec![ql, x]),
        s("self_attn.q_a_layernorm.weight", vec![ql]),
        s("self_attn.q_b_proj.weight", vec![full_heads * qk, ql]),
        // 🪤 NoPE: kv_cache_dim == kv_lora_rank, no rope section.
        s(
            "self_attn.kv_a_proj_with_mqa.weight",
            vec![cfg.kv_cache_dim(), x],
        ),
        s("self_attn.kv_a_layernorm.weight", vec![kvl]),
        s("self_attn.kv_b_proj.weight", vec![full_heads * kvb, kvl]),
        s(
            "self_attn.o_proj.weight",
            vec![x, full_heads * cfg.v_head_dim],
        ),
        // ── indexer ──
        s("self_attn.indexer.wq_b.weight", vec![ih * ihd, ql]),
        s("self_attn.indexer.wk.weight", vec![ihd, x]),
        s("self_attn.indexer.k_norm.weight", vec![ihd]),
        // 🪤 REQUIRED: LayerNorm bias. Its absence is an error, never a silent zero.
        s("self_attn.indexer.k_norm.bias", vec![ihd]),
        s("self_attn.indexer.weights_proj.weight", vec![ih, x]),
        s("self_attn.indexer.index_kpool_compress_gate", vec![ihd, x]),
        // 🪤 indexed by pool SLOT: [kpool, index_head_dim].
        s(
            "self_attn.indexer.index_kpool_compress_ape",
            vec![cfg.index_kpool, ihd],
        ),
    ]
}

/// What a successful bind saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaBindReport {
    pub bound: usize,
    pub total_bytes: usize,
    /// `self_attn.*` names present in the source that no spec claims. Never empty-
    /// tolerated: an unclaimed attention tensor means the architecture moved.
    pub unclaimed: Vec<String>,
}

/// Verify one DSA block against the spec table.
///
/// Every spec must be present with the exact dtype and shape, and no `self_attn.*`
/// tensor may be left unclaimed. Both directions matter: a missing tensor is a
/// broken layer, and an unexpected one means the checkpoint is not the model we
/// think it is.
pub fn verify_dsa_block(
    cfg: &Glm5NextDsaConfig,
    full_heads: usize,
    source: &dyn TensorSource,
) -> Result<DsaBindReport> {
    cfg.validate()?;
    let specs = dsa_tensor_specs(cfg, full_heads);
    let mut total_bytes = 0usize;

    for spec in &specs {
        let Some(raw) = source.get(&spec.name) else {
            bail!("DSA bind: missing required tensor `{}`", spec.name);
        };
        if raw.dtype != spec.dtype {
            bail!(
                "DSA bind: `{}` is {:?}, expected {:?}",
                spec.name,
                raw.dtype,
                spec.dtype
            );
        }
        if raw.shape != spec.shape {
            bail!(
                "DSA bind: `{}` has shape {:?}, expected {:?}",
                spec.name,
                raw.shape,
                spec.shape
            );
        }
        let elems: usize = spec.shape.iter().product();
        if raw.bytes.len() != elems * 2 {
            bail!(
                "DSA bind: `{}` carries {} bytes, expected {} for {:?} BF16",
                spec.name,
                raw.bytes.len(),
                elems * 2,
                spec.shape
            );
        }
        total_bytes += raw.bytes.len();
    }

    let claimed: BTreeSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    let unclaimed: Vec<String> = source
        .names()
        .into_iter()
        .filter(|n| n.starts_with("self_attn.") && !claimed.contains(n.as_str()))
        .collect();
    if !unclaimed.is_empty() {
        bail!(
            "DSA bind: {} unclaimed self_attn tensor(s), first: {:?}. An unexpected \
             attention tensor means the architecture moved — do not skip it.",
            unclaimed.len(),
            &unclaimed[..unclaimed.len().min(5)]
        );
    }

    Ok(DsaBindReport {
        bound: specs.len(),
        total_bytes,
        unclaimed,
    })
}

#[cfg(test)]
mod tests;
