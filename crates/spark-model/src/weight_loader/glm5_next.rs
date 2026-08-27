// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash (`glm5_next`) tensor accounting.
//!
//! Slice 1 scope: **classify every tensor in the checkpoint, deliberately.**
//! No loading, no device work, no forward pass.
//!
//! Reference checkpoint: `LibertAIDAI/GLM-5.3-Flash-NVFP4` snapshot
//! `9e0d74e3cef17f634e84fb8e2223707e02616290` — 120 shards, **113,074 tensors**,
//! 407 distinct name patterns. Every claim here is scoped to that checkpoint.
//!
//! The point of this module is that "we loaded the model" and "we accounted for
//! the checkpoint" are different statements. A loader that silently ignores an
//! unrecognised tensor will happily produce a model that is quietly wrong — the
//! failure mode that cost the DS4F campaign weeks. So the contract is:
//! `classify` returns `None` for anything it has not been taught, and the test
//! suite fails if a single tensor in the reference checkpoint lands there.

use std::collections::BTreeMap;

/// What a tensor is *for*. Deliberately coarse — Slice 1 proves coverage, not
/// placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorRole {
    /// Token embedding / final norm / lm_head.
    Embedding,
    LmHead,
    FinalNorm,
    /// KDA linear-attention mixer (34 of the 45 text layers).
    KdaProjection,
    KdaConv,
    KdaDecay,
    KdaGate,
    KdaNorm,
    /// NoPE sparse MLA (11 text layers + the MTP layer).
    MlaProjection,
    MlaNorm,
    /// DSA top-k indexer that fronts each sparse-MLA layer.
    Indexer,
    IndexerNorm,
    /// MoE router + experts + shared expert.
    MoeRouter,
    MoeExpert,
    MoeShared,
    /// Dense FFN (layers 0..2, `first_k_dense_replace = 3`).
    DenseFfn,
    /// Per-layer norms and mHC (hyper-connection) parameters.
    LayerNorm,
    HyperConnection,
    /// MTP head at layer 45. 🪤 GLM does NOT use `mtp.0.*`.
    MtpProjection,
    MtpNorm,
    /// Vision tower — present in the checkpoint, out of scope for the text port.
    Vision,
}

impl TensorRole {
    /// Norm-family roles. Used by the RMSNorm sanity pass: a tensor whose name
    /// says "norm" must land in one of these, or the classifier is lying about
    /// something. (Hazard carried from `notavault-atlas`: RMSNorm weights have
    /// silently corrupted Atlas numbers before.)
    pub fn is_norm(self) -> bool {
        matches!(
            self,
            TensorRole::FinalNorm
                | TensorRole::KdaNorm
                | TensorRole::MlaNorm
                | TensorRole::IndexerNorm
                | TensorRole::LayerNorm
                | TensorRole::MtpNorm
        )
    }

    /// Text-model roles Atlas must eventually implement. Vision is excluded.
    pub fn is_text_model(self) -> bool {
        !matches!(self, TensorRole::Vision)
    }
}

/// Replace every all-numeric path segment with `#`, so `layers.7.` and
/// `layers.N.` (our census canonicalisation) collapse to the same key.
fn normalize(name: &str) -> String {
    name.split('.')
        .map(|seg| {
            if seg.is_empty() {
                seg
            } else if seg.chars().all(|c| c.is_ascii_digit()) || seg == "N" || seg == "E" {
                "#"
            } else {
                seg
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Classify one checkpoint tensor name.
///
/// Returns `None` for anything unrecognised — callers MUST treat that as a
/// hard error, never as "skip it".
pub fn classify(name: &str) -> Option<TensorRole> {
    let n = normalize(name);
    let s = n.as_str();

    // ---- non-layer -------------------------------------------------------
    if s == "lm_head.weight" {
        return Some(TensorRole::LmHead);
    }
    if s == "model.language_model.embed_tokens.weight" {
        return Some(TensorRole::Embedding);
    }
    if s == "model.language_model.norm.weight" {
        return Some(TensorRole::FinalNorm);
    }
    if s.starts_with("model.visual.") || s.starts_with("model.vision") {
        return Some(TensorRole::Vision);
    }

    let rest = s.strip_prefix("model.language_model.layers.#.")?;

    // ---- MTP head (layer 45) --------------------------------------------
    // These names appear ONLY on the MTP layer. Layer-index checking is the
    // caller's job (see `is_mtp_only_name`); here we classify by role.
    match rest {
        "eh_proj.weight" => return Some(TensorRole::MtpProjection),
        "enorm.weight" | "hnorm.weight" | "shared_head.norm.weight" => {
            return Some(TensorRole::MtpNorm);
        }
        _ => {}
    }

    // ---- KDA linear attention -------------------------------------------
    match rest {
        "self_attn.q_proj.weight"
        | "self_attn.k_proj.weight"
        | "self_attn.v_proj.weight"
        | "self_attn.b_proj.weight"
        | "self_attn.f_a_proj.weight"
        | "self_attn.f_b_proj.weight"
        | "self_attn.g_a_proj.weight"
        | "self_attn.g_b_proj.weight" => return Some(TensorRole::KdaProjection),
        "self_attn.q_conv1d.weight" | "self_attn.k_conv1d.weight" | "self_attn.v_conv1d.weight" => {
            return Some(TensorRole::KdaConv);
        }
        "self_attn.A_log" | "self_attn.dt_bias" => return Some(TensorRole::KdaDecay),
        "self_attn.o_norm.weight" => return Some(TensorRole::KdaNorm),
        _ => {}
    }

    // ---- NoPE sparse MLA -------------------------------------------------
    match rest {
        "self_attn.q_a_proj.weight"
        | "self_attn.q_b_proj.weight"
        | "self_attn.kv_a_proj_with_mqa.weight"
        | "self_attn.kv_b_proj.weight" => return Some(TensorRole::MlaProjection),
        "self_attn.q_a_layernorm.weight" | "self_attn.kv_a_layernorm.weight" => {
            return Some(TensorRole::MlaNorm);
        }
        // o_proj is shared by both mixer families (46 occurrences = 34 KDA +
        // 11 DSA + 1 MTP), so it cannot discriminate; it is an output
        // projection either way.
        "self_attn.o_proj.weight" => return Some(TensorRole::MlaProjection),
        _ => {}
    }

    // ---- DSA indexer -----------------------------------------------------
    if let Some(idx) = rest.strip_prefix("self_attn.indexer.") {
        return Some(match idx {
            "k_norm.weight" | "k_norm.bias" => TensorRole::IndexerNorm,
            _ => TensorRole::Indexer,
        });
    }

    // ---- mHC hyper-connections ------------------------------------------
    if rest.starts_with("hc_") {
        return Some(TensorRole::HyperConnection);
    }

    // ---- norms -----------------------------------------------------------
    if rest == "input_layernorm.weight" || rest == "post_attention_layernorm.weight" {
        return Some(TensorRole::LayerNorm);
    }

    // ---- MoE / dense FFN -------------------------------------------------
    if let Some(mlp) = rest.strip_prefix("mlp.") {
        if mlp.starts_with("gate.") || mlp == "gate.weight" {
            return Some(TensorRole::MoeRouter);
        }
        if mlp.starts_with("experts.#.") {
            return Some(TensorRole::MoeExpert);
        }
        if mlp.starts_with("shared_experts.") {
            return Some(TensorRole::MoeShared);
        }
        // Bare gate/up/down on a layer with no expert dimension = dense FFN.
        if mlp.starts_with("gate_proj.")
            || mlp.starts_with("up_proj.")
            || mlp.starts_with("down_proj.")
        {
            return Some(TensorRole::DenseFfn);
        }
    }

    None
}

/// Names that exist ONLY on the MTP layer. Used to locate the MTP layer index
/// from the checkpoint itself rather than assuming one.
///
/// 🪤 Do not look for `mtp.0.*`: GLM-5.3 has **zero** such tensors. Verified by
/// scanning all 120 shard headers of the reference checkpoint.
pub fn is_mtp_only_name(name: &str) -> bool {
    let n = normalize(name);
    let Some(rest) = n.strip_prefix("model.language_model.layers.#.") else {
        return false;
    };
    matches!(
        rest,
        "eh_proj.weight" | "enorm.weight" | "hnorm.weight" | "shared_head.norm.weight"
    )
}

/// Result of accounting a whole checkpoint's tensor-name list.
#[derive(Debug, Default)]
pub struct Accounting {
    pub total: usize,
    pub by_role: BTreeMap<String, usize>,
    /// Anything `classify` refused. MUST be empty.
    pub unknown: Vec<String>,
    /// Layer indices that carry MTP-only tensors.
    pub mtp_layers: Vec<usize>,
}

/// Account for a list of `(tensor_name, count)` pairs.
pub fn account<'a, I>(names: I) -> Accounting
where
    I: IntoIterator<Item = (&'a str, usize)>,
{
    let mut acc = Accounting::default();
    let mut mtp = std::collections::BTreeSet::new();
    for (name, count) in names {
        acc.total += count;
        match classify(name) {
            Some(role) => {
                *acc.by_role.entry(format!("{role:?}")).or_insert(0) += count;
            }
            None => acc.unknown.push(name.to_string()),
        }
        if is_mtp_only_name(name)
            && let Some(i) = layer_index(name)
        {
            mtp.insert(i);
        }
    }
    acc.mtp_layers = mtp.into_iter().collect();
    acc
}

/// Extract the numeric layer index from a real tensor name (`None` for the
/// canonicalised `layers.N.` form, which carries no index).
pub fn layer_index(name: &str) -> Option<usize> {
    let mut it = name.split('.');
    while let Some(seg) = it.next() {
        if seg == "layers" {
            return it.next().and_then(|s| s.parse().ok());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_numeric_and_canonical_segments_alike() {
        assert_eq!(
            normalize("model.language_model.layers.7.self_attn.o_proj.weight"),
            normalize("model.language_model.layers.N.self_attn.o_proj.weight")
        );
        assert_eq!(
            normalize("model.language_model.layers.45.mlp.experts.12.down_proj.weight"),
            normalize("model.language_model.layers.N.mlp.experts.E.down_proj.weight")
        );
    }

    #[test]
    fn mtp_is_found_by_layer_name_not_by_mtp_prefix() {
        assert!(is_mtp_only_name(
            "model.language_model.layers.45.eh_proj.weight"
        ));
        assert!(!is_mtp_only_name("model.language_model.layers.3.eh_proj"));
        // The DeepSeek convention must NOT be what we key on.
        assert!(!is_mtp_only_name("model.layers.mtp.0.eh_proj.weight"));
        assert_eq!(
            layer_index("model.language_model.layers.45.eh_proj.weight"),
            Some(45)
        );
    }

    #[test]
    fn unknown_tensor_is_refused_not_skipped() {
        assert_eq!(
            classify("model.language_model.layers.4.self_attn.wat"),
            None
        );
        let acc = account([("model.language_model.layers.4.self_attn.wat", 1)]);
        assert_eq!(acc.unknown.len(), 1);
    }

    #[test]
    fn norm_tensors_land_in_norm_roles() {
        for n in [
            "model.language_model.norm.weight",
            "model.language_model.layers.0.self_attn.o_norm.weight",
            "model.language_model.layers.3.self_attn.q_a_layernorm.weight",
            "model.language_model.layers.3.self_attn.indexer.k_norm.weight",
            "model.language_model.layers.5.input_layernorm.weight",
            "model.language_model.layers.45.enorm.weight",
        ] {
            let r = classify(n).unwrap_or_else(|| panic!("unclassified: {n}"));
            assert!(r.is_norm(), "{n} classified as non-norm {r:?}");
        }
    }
}
