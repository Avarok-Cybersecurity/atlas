// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 Flash (`general.architecture = deepseek41`) tensor-name
//! remaps, split out of the generic translator in the parent module.

use super::{GgufName, HF_PREFIX};

/// DeepSeek-V4.1 Flash (`general.architecture = deepseek41`) name remaps.
///
/// The GGUF converter emits one flat `blk.N.*` namespace; Atlas's DeepSeek
/// loaders index by sub-module (`…attn.wq_a`, `…compressor.wkv`, `…indexer.wk`),
/// matching the V4 loader this model's loader is modelled on.
///
/// Three things here are NOT true of the decoder-only default:
///
///   * **Experts are stacked.** `ffn_{gate,up,down}_exps.weight` is 3-D
///     (`[5120, 2304, 384]`), one tensor per layer holding all 384 experts, so
///     those resolve to [`GgufName::ExpertStack`] and the loader fans them out.
///     🪤 [`expert_name`](super::expert_name) currently emits the `mlp.experts.{e}.{proj}_proj`
///     convention; the V4-family loaders read `ffn.experts.{e}.w{1,2,3}`. S2
///     has to reconcile those two, either by teaching `expert_name` the arch or
///     by having the V4.1 loader read the `mlp.*` spelling.
///
///   * **Several tensors exist on only SOME layers**, because V4.1 shares
///     compressed attention: `attn_compressor_kv` appears 4 times (the
///     `kv_source_layer_ids` `[2, 8, 14, 20]`), `indexer.attn_k` 4 times,
///     `indexer.proj` 8 times, and `engram_*` twice (layers 1 and 14). A
///     translation that assumed one of each per block would be wrong; this
///     function is per-name and makes no such assumption.
///
///   * **`exp_probs_b_vl.bias` is the VISION routing bias.** It ships on all 40
///     layers even in this text-only quant (the file is tagged
///     `image-text-to-text`). It is mapped rather than dropped so a later
///     multimodal path does not have to re-convert the checkpoint.
///
/// DeepSeek-V4.1 tensors that are NEVER uploaded: the routed expert stacks
/// (40 x 3 x 384 x 12.22 MiB) and the two ~30 GiB engram tables. They are
/// recorded as deferred with their on-disk location and served by
/// `expert_stream` (pread into a device-visible cache / rows on demand).
/// Returns the store name the loader looks them up under.
pub fn deepseek41_deferred_name(gguf_name: &str) -> Option<String> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let (layer, tail) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let lp = format!("{HF_PREFIX}.layers.{layer}");
    Some(match tail {
        "ffn_gate_exps.weight" => format!("{lp}.ffn.experts_stack.gate"),
        "ffn_up_exps.weight" => format!("{lp}.ffn.experts_stack.up"),
        "ffn_down_exps.weight" => format!("{lp}.ffn.experts_stack.down"),
        "engram_embd.weight" => format!("{lp}.engram.embd"),
        _ => return None,
    })
}

pub(super) fn translate_deepseek41(gguf_name: &str) -> Option<GgufName> {
    // Top-level tensors.
    match gguf_name {
        "token_embd.weight" => {
            return Some(GgufName::Direct(format!("{HF_PREFIX}.embed_tokens.weight")));
        }
        "output_norm.weight" => return Some(GgufName::Direct(format!("{HF_PREFIX}.norm.weight"))),
        "output.weight" => return Some(GgufName::Direct("lm_head.weight".to_string())),
        _ => {}
    }

    let rest = gguf_name.strip_prefix("blk.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    let layer: usize = layer_str.parse().ok()?;
    let lp = format!("{HF_PREFIX}.layers.{layer}");

    // Stacked MoE experts.
    let proj = match suffix {
        "ffn_gate_exps.weight" => Some("gate"),
        "ffn_up_exps.weight" => Some("up"),
        "ffn_down_exps.weight" => Some("down"),
        _ => None,
    };
    if let Some(proj) = proj {
        return Some(GgufName::ExpertStack { layer, proj });
    }

    let mapped: String = match suffix {
        // Block norms.
        "attn_norm.weight" => format!("{lp}.attn_norm.weight"),
        "ffn_norm.weight" => format!("{lp}.ffn_norm.weight"),

        // MLA attention: low-rank q/o pairs, a single latent kv, per-head sinks.
        "attn_q_a.weight" => format!("{lp}.attn.wq_a.weight"),
        "attn_q_b.weight" => format!("{lp}.attn.wq_b.weight"),
        "attn_q_a_norm.weight" => format!("{lp}.attn.q_norm.weight"),
        "attn_kv.weight" => format!("{lp}.attn.wkv.weight"),
        "attn_kv_a_norm.weight" => format!("{lp}.attn.kv_norm.weight"),
        "attn_output_a.weight" => format!("{lp}.attn.wo_a.weight"),
        "attn_output_b.weight" => format!("{lp}.attn.wo_b.weight"),
        "attn_sinks.weight" => format!("{lp}.attn.attn_sink"),

        // Shared compressor (only on `kv_source_layer_ids`).
        "attn_compressor_kv.weight" => format!("{lp}.compressor.wkv.weight"),
        "attn_compressor_gate.weight" => format!("{lp}.compressor.wgate.weight"),
        "attn_compressor_norm.weight" => format!("{lp}.compressor.norm.weight"),

        // Sparse-attention indexer (only on `index_source_layer_ids`).
        "indexer.attn_q_b.weight" => format!("{lp}.indexer.wq_b.weight"),
        "indexer.attn_k.weight" => format!("{lp}.indexer.wk.weight"),
        "indexer.k_norm.weight" => format!("{lp}.indexer.k_norm.weight"),
        "indexer.proj.weight" => format!("{lp}.indexer.proj.weight"),

        // MoE router and shared expert.
        "ffn_gate_inp.weight" => format!("{lp}.ffn.gate.weight"),
        "exp_probs_b.bias" => format!("{lp}.ffn.gate.e_score_correction_bias"),
        "exp_probs_b_vl.bias" => format!("{lp}.ffn.gate.e_score_correction_bias_vl"),
        "ffn_gate_shexp.weight" => format!("{lp}.ffn.shared_experts.w1"),
        "ffn_down_shexp.weight" => format!("{lp}.ffn.shared_experts.w2"),
        "ffn_up_shexp.weight" => format!("{lp}.ffn.shared_experts.w3"),

        // Hyper-connections (mHC), two sites x three tensors.
        "hc_attn_base.weight" => format!("{lp}.hc_attn_base"),
        "hc_attn_fn.weight" => format!("{lp}.hc_attn_fn"),
        "hc_attn_scale.weight" => format!("{lp}.hc_attn_scale"),
        "hc_ffn_base.weight" => format!("{lp}.hc_ffn_base"),
        "hc_ffn_fn.weight" => format!("{lp}.hc_ffn_fn"),
        "hc_ffn_scale.weight" => format!("{lp}.hc_ffn_scale"),

        // Engram (layers 1 and 14 only). `engram_embd` is the ~30 GiB hash
        // table; it is named here but MUST NOT be resident — S3 streams it
        // through a row cache.
        "engram_embd.weight" => format!("{lp}.engram.embd"),
        "engram_q.weight" => format!("{lp}.engram.wq"),
        "engram_k.weight" => format!("{lp}.engram.wk"),
        "engram_wkv.weight" => format!("{lp}.engram.wkv"),

        _ => return None,
    };
    Some(GgufName::Direct(mapped))
}
