// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the Qwen3.5 / Qwen3.8 MTP head names of a GGUF file (port lane L19, gated by the
//! `gguf_native_iq4` row through [`super::super::keep_packed_iq4`]).
//!
//! llama.cpp's converter writes a model's multi-token-prediction head as ONE extra block after the trunk,
//! `blk.<L>.*` with `L = num_hidden_layers` (`blk.64` on the 27B), in two parts:
//!   * the head's own tensors under `blk.<L>.nextn.*`: `eh_proj` (the HF `mtp.fc`, `[H, 2H]`), `enorm` and
//!     `hnorm` (`mtp.pre_fc_norm_embedding` / `mtp.pre_fc_norm_hidden`), `shared_head_norm` (`mtp.norm`),
//!     and, for families whose head has its own, `embed_tokens` and `shared_head_head`;
//!   * the head's transformer block under the ordinary per-layer names (`blk.<L>.attn_q.weight`, ...).
//!
//! Atlas's Qwen MTP loader (`spark-model` `weight_map/loaders_moe.rs`, dense arm) reads the HF names
//! `mtp.fc.weight`, `mtp.pre_fc_norm_*.weight`, `mtp.layers.0.*` and `mtp.norm.weight`, so this module maps
//! the first part by name and re-roots the second part from `model.layers.<L>.` to `mtp.layers.0.`. A
//! Qwen MTP head shares the trunk's embedding and output head, so a `nextn.embed_tokens` or
//! `nextn.shared_head_head` copy is dropped rather than loaded twice.
//!
//! `L` is not in the tensor name of the block part, so the caller finds it once per file
//! ([`mtp_layer`]: the index of the first `blk.N.nextn.*` tensor) and passes it to
//! [`translate_with_mtp`]. With `None` (the row off, no MTP head, or another architecture),
//! [`translate_with_mtp`] is exactly [`super::translate`], so files load as they always have.
//!
//! Scope: the dense `qwen35` architecture (`qwen35` / `qwen3_5`). A `qwen35moe` head's routed experts
//! would need the expert fan-out re-rooted too; it is not mapped here and keeps today's behavior.

use super::GgufName;

/// The HF names of the head's own tensors, by the `blk.<L>.nextn.` suffix.
fn nextn_tensor(sub: &str) -> Option<GgufName> {
    let hf = match sub {
        "eh_proj.weight" => "mtp.fc.weight",
        "enorm.weight" => "mtp.pre_fc_norm_embedding.weight",
        "hnorm.weight" => "mtp.pre_fc_norm_hidden.weight",
        "shared_head_norm.weight" => "mtp.norm.weight",
        // The trunk's embedding and output head serve the Qwen MTP head.
        "embed_tokens.weight" | "shared_head_head.weight" => return Some(GgufName::Drop),
        _ => return None,
    };
    Some(GgufName::Direct(hf.to_string()))
}

/// True for the dense Qwen3.5-family architecture strings this mapping covers.
pub fn is_dense_qwen35(arch: &str) -> bool {
    matches!(arch, "qwen35" | "qwen3_5")
}

/// The block index that carries the MTP head: `N` of the first `blk.N.nextn.*` name, if any.
pub fn mtp_layer<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<usize> {
    names.into_iter().find_map(|n| {
        let rest = n.strip_prefix("blk.")?;
        let (layer, sub) = rest.split_once('.')?;
        sub.starts_with("nextn.").then(|| layer.parse().ok())?
    })
}

/// Re-root a trunk name of block `layer` under the MTP head's single block.
fn reroot(hf: &str, layer: usize) -> Option<String> {
    let prefix = format!("{}.layers.{layer}.", super::HF_PREFIX);
    hf.strip_prefix(&prefix)
        .map(|rest| format!("mtp.layers.0.{rest}"))
}

/// [`super::translate`] with the MTP head of block `mtp_layer` mapped to Atlas's `mtp.*` names. With
/// `mtp_layer = None`, or for an architecture other than dense qwen35, it is [`super::translate`].
pub fn translate_with_mtp(
    gguf_name: &str,
    arch: &str,
    mtp_layer: Option<usize>,
) -> Option<GgufName> {
    if let Some(layer) = mtp_layer
        && is_dense_qwen35(arch)
        && let Some(rest) = gguf_name
            .strip_prefix("blk.")
            .and_then(|r| r.strip_prefix(&format!("{layer}.")))
    {
        if let Some(sub) = rest.strip_prefix("nextn.") {
            return nextn_tensor(sub);
        }
        return match super::translate(gguf_name, arch)? {
            GgufName::Direct(hf) => reroot(&hf, layer).map(GgufName::Direct),
            GgufName::Drop => Some(GgufName::Drop),
            // A dense head has no stacked experts; an unexpected one is left unmapped.
            GgufName::ExpertStack { .. } => None,
        };
    }
    super::translate(gguf_name, arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(s: &str) -> Option<GgufName> {
        Some(GgufName::Direct(s.to_string()))
    }

    #[test]
    fn the_head_layer_is_found_from_any_nextn_name() {
        let names = [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.63.ffn_down.weight",
            "blk.64.attn_q.weight",
            "blk.64.nextn.eh_proj.weight",
        ];
        assert_eq!(mtp_layer(names), Some(64));
        assert_eq!(mtp_layer(["blk.3.attn_q.weight", "output.weight"]), None);
        // `nextn` must be the whole sub-name prefix, not a substring of another stem.
        assert_eq!(mtp_layer(["blk.7.ffn_nextn_x.weight"]), None);
        assert_eq!(mtp_layer(["blk.x.nextn.eh_proj.weight"]), None);
    }

    #[test]
    fn nextn_tensors_map_to_the_hf_mtp_names() {
        let t = |n: &str| translate_with_mtp(n, "qwen35", Some(64));
        assert_eq!(t("blk.64.nextn.eh_proj.weight"), direct("mtp.fc.weight"));
        assert_eq!(
            t("blk.64.nextn.enorm.weight"),
            direct("mtp.pre_fc_norm_embedding.weight")
        );
        assert_eq!(
            t("blk.64.nextn.hnorm.weight"),
            direct("mtp.pre_fc_norm_hidden.weight")
        );
        assert_eq!(
            t("blk.64.nextn.shared_head_norm.weight"),
            direct("mtp.norm.weight")
        );
        assert_eq!(t("blk.64.nextn.embed_tokens.weight"), Some(GgufName::Drop));
        assert_eq!(
            t("blk.64.nextn.shared_head_head.weight"),
            Some(GgufName::Drop)
        );
        assert_eq!(t("blk.64.nextn.unknown.weight"), None);
    }

    #[test]
    fn the_head_block_is_rerooted_under_mtp_layers_0() {
        let t = |n: &str| translate_with_mtp(n, "qwen35", Some(64));
        assert_eq!(
            t("blk.64.attn_q.weight"),
            direct("mtp.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            t("blk.64.attn_output.weight"),
            direct("mtp.layers.0.self_attn.o_proj.weight")
        );
        assert_eq!(
            t("blk.64.attn_q_norm.weight"),
            direct("mtp.layers.0.self_attn.q_norm.weight")
        );
        assert_eq!(
            t("blk.64.attn_norm.weight"),
            direct("mtp.layers.0.input_layernorm.weight")
        );
        assert_eq!(
            t("blk.64.post_attention_norm.weight"),
            direct("mtp.layers.0.post_attention_layernorm.weight")
        );
        assert_eq!(
            t("blk.64.ffn_gate.weight"),
            direct("mtp.layers.0.mlp.gate_proj.weight")
        );
        assert_eq!(
            t("blk.64.ffn_down.weight"),
            direct("mtp.layers.0.mlp.down_proj.weight")
        );
    }

    #[test]
    fn everything_else_is_the_plain_translation() {
        for (name, arch, layer) in [
            ("blk.63.attn_q.weight", "qwen35", Some(64)),
            ("blk.640.attn_q.weight", "qwen35", Some(64)),
            ("blk.64.attn_q.weight", "qwen35", None),
            ("blk.64.attn_q.weight", "qwen35moe", Some(64)),
            ("blk.64.attn_q.weight", "llama", Some(64)),
            ("token_embd.weight", "qwen35", Some(64)),
            ("output.weight", "qwen35", Some(64)),
            ("blk.64.nextn.eh_proj.weight", "qwen35", None),
        ] {
            assert_eq!(
                translate_with_mtp(name, arch, layer),
                super::super::translate(name, arch),
                "{name} {arch} {layer:?}"
            );
        }
    }
}
