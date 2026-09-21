// SPDX-License-Identifier: AGPL-3.0-only

//! Build a [`ModelConfig`] from GGUF file metadata.
//!
//! GGUF carries its model config inline as metadata key/values
//! (`llama.block_count`, `qwen3.attention.head_count`, …) rather than a
//! sibling `config.json`. This module reads those keys through the
//! [`GgufMeta`] accessor (implemented by the GGUF parser in spark-runtime, so
//! avarok-core keeps no GGUF dependency) and produces a validated
//! [`ModelConfig`] for the llama / qwen2 / qwen3 / gemma decoder families.
//!
//! Strategy: synthesize an HF-config-shaped JSON object from the GGUF keys and
//! deserialize it into `ModelConfig` (serde `#[serde(default)]` fills the many
//! fields GGUF has no analog for), then set the architecture flags
//! (`attn_gated`, `weight_prefix`, gemma `embed_scale` /
//! `final_logit_softcapping`) explicitly, then run the shared
//! [`super::finalize_config`]. No silent production defaults: every value GGUF
//! omits is either derived by an explicit documented rule or is an error.

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use super::{ModelConfig, finalize_config};

/// Typed read access to GGUF metadata. Implemented by the spark-runtime GGUF
/// parser over its parsed key/value table. All getters return `None` when the
/// key is absent or holds a different value type — the builder decides whether
/// absence is fatal or has a derivation rule.
pub trait GgufMeta {
    /// Any unsigned/signed integer metadata value, widened to u64.
    fn get_u64(&self, key: &str) -> Option<u64>;
    /// Any float metadata value (f32/f64), widened to f64.
    fn get_f64(&self, key: &str) -> Option<f64>;
    /// A string metadata value.
    fn get_str(&self, key: &str) -> Option<&str>;
    /// Length of an array metadata value (e.g. `tokenizer.ggml.tokens`).
    fn get_arr_len(&self, key: &str) -> Option<usize>;
    /// An integer array metadata value, every element widened to u64.
    ///
    /// DEFAULTS TO `None` so existing implementors keep compiling: a reader
    /// that has not implemented it reports "absent", which every caller here
    /// already handles. DeepSeek-V4.1 is the first arch that needs real array
    /// VALUES rather than a length (`attention.compress_ratios`,
    /// `engram.layer_ids`, `engram.{multipliers,primes,offsets}`), so a reader
    /// that returns `None` here simply cannot build a V4.1 config.
    fn get_u64_arr(&self, _key: &str) -> Option<Vec<u64>> {
        None
    }
    /// A float array metadata value, every element widened to f64. Defaults to
    /// `None` for the same reason as [`GgufMeta::get_u64_arr`].
    /// DeepSeek-V4.1 needs it for `swiglu_clamp_exp` / `swiglu_clamp_shexp`.
    fn get_f64_arr(&self, _key: &str) -> Option<Vec<f64>> {
        None
    }
}

/// Inputs to [`config_from_gguf`]: the metadata accessor plus two facts the
/// builder needs from the tensor section (not the metadata KV block).
pub struct GgufConfigInputs<'a> {
    pub meta: &'a dyn GgufMeta,
    /// Rows of `token_embd.weight` — the authoritative vocab size when the
    /// `{arch}.vocab_size` key is absent. `None` if the loader could not read
    /// the tensor shape before building the config.
    pub token_embd_vocab: Option<usize>,
    /// Whether the file contains an `output.weight` tensor. Its presence means
    /// an untied LM head; its absence means the LM head ties to the input
    /// embeddings. GGUF has no explicit `tie_word_embeddings` key, so this is
    /// the only reliable signal.
    pub has_output_weight: bool,
}

/// Map a GGUF `general.architecture` string to an Atlas `model_type` (must be
/// a supported loader string) and whether attention Q is gated.
///
/// Plain-decoder GGUFs (llama/qwen2) have no dedicated Atlas arch loader; the
/// closest dense GQA path is the Mistral loader. qwen3 dense maps to `qwen3_5`
/// with `num_experts == 0` (dense qwen3.5 loader). Returns an error for
/// unmapped architectures rather than guessing.
fn arch_to_model_type(arch: &str) -> Result<(&'static str, bool)> {
    // (model_type, attn_gated)
    Ok(match arch {
        "llama" => ("mistral", false),
        // qwen2 ships QKV biases; the Mistral GQA path is the closest dense
        // loader. (Bias handling is a known caveat — see module notes.)
        "qwen2" => ("mistral", false),
        // qwen3 dense: q_norm/k_norm, ungated Q. num_experts==0 → dense loader.
        "qwen3" => ("qwen3_5", false),
        "qwen3moe" => ("qwen3_5_moe", false),
        // gemma family: GeGLU, ungated Q, embedding scale + logit softcap.
        "gemma" | "gemma2" | "gemma3" | "gemma4" => ("gemma4", false),
        // DeepSeek-V4.1 Flash. MLA + 384-expert MoE + mHC + shared compressed
        // attention + engram. Its GGUF ships no dense `feed_forward_length`
        // (every FFN is MoE or the shared expert), so the required-key handling
        // below treats that key as optional for MoE arches.
        "deepseek41" => ("deepseek_v41", false),
        other => bail!(
            "GGUF general.architecture '{other}' has no Atlas model_type mapping. \
             Supported GGUF arches: llama, qwen2, qwen3, qwen3moe, gemma/gemma2/gemma3/gemma4, deepseek41, kimi-k3."
        ),
    })
}

/// Build a validated [`ModelConfig`] from GGUF metadata.
pub fn config_from_gguf(inputs: &GgufConfigInputs) -> Result<ModelConfig> {
    let meta = inputs.meta;

    let arch = meta
        .get_str("general.architecture")
        .context("GGUF metadata missing required key 'general.architecture'")?
        .to_string();
    let arch_l = arch.to_ascii_lowercase();
    if matches!(arch_l.as_str(), "kimi-k3" | "kimi_k3" | "kimik3") {
        return kimi_k3::config_from_kimi_k3_gguf(meta);
    }
    let (model_type, attn_gated) = arch_to_model_type(&arch)?;

    // Namespaced key helper: `{arch}.<suffix>`.
    let k = |suffix: &str| format!("{arch}.{suffix}");
    let req_u64 = |suffix: &str| -> Result<u64> {
        meta.get_u64(&k(suffix))
            .with_context(|| format!("GGUF metadata missing required key '{arch}.{suffix}'"))
    };

    // ── Core dimensions (required) ──
    let hidden_size = req_u64("embedding_length")? as usize;
    let num_hidden_layers = req_u64("block_count")? as usize;
    // `feed_forward_length` is the DENSE FFN width. A pure-MoE arch may not ship
    // it at all (DeepSeek-V4.1 has only `expert_feed_forward_length`), so it is
    // required only when the file declares no experts. Resolving to 0 for a MoE
    // file is correct: nothing dispatches a dense FFN there.
    let has_experts = meta.get_u64(&k("expert_count")).unwrap_or(0) > 0;
    let intermediate_size = match meta.get_u64(&k("feed_forward_length")) {
        Some(v) => v as usize,
        None if has_experts => 0,
        None => bail!("GGUF metadata missing required key '{arch}.feed_forward_length'"),
    };
    let num_attention_heads = req_u64("attention.head_count")? as usize;

    // GQA: kv head count defaults to full MHA (== attention heads) when the key
    // is absent, which is the ggml convention.
    let num_key_value_heads = meta
        .get_u64(&k("attention.head_count_kv"))
        .map(|v| v as usize)
        .unwrap_or(num_attention_heads);
    if num_attention_heads > 0
        && (num_key_value_heads == 0 || !num_attention_heads.is_multiple_of(num_key_value_heads))
    {
        bail!(
            "GGUF metadata key '{}.attention.head_count_kv' ({num_key_value_heads}) must be a non-zero divisor of attention.head_count ({num_attention_heads})",
            arch
        );
    }

    // head_dim: explicit key_length if present, else hidden_size / head_count.
    // Erroring on a non-divisible fallback avoids a silently-wrong head_dim.
    let head_dim = match meta.get_u64(&k("attention.key_length")) {
        Some(0) => bail!(
            "GGUF metadata key '{}.attention.key_length' must be greater than zero",
            arch
        ),
        Some(v) => v as usize,
        None => {
            if num_attention_heads == 0 || !hidden_size.is_multiple_of(num_attention_heads) {
                bail!(
                    "GGUF: cannot derive head_dim — '{arch}.attention.key_length' absent and \
                     hidden_size ({hidden_size}) not divisible by head_count ({num_attention_heads})"
                );
            }
            hidden_size / num_attention_heads
        }
    };

    // vocab_size: explicit key → token_embd rows → tokenizer token list length.
    let metadata_vocab = meta.get_u64(&k("vocab_size")).map(|v| v as usize);
    if let (Some(metadata_vocab), Some(tensor_vocab)) = (metadata_vocab, inputs.token_embd_vocab)
        && metadata_vocab != tensor_vocab
    {
        bail!(
            "GGUF: '{arch}.vocab_size' ({metadata_vocab}) does not match token_embd.weight rows \
             ({tensor_vocab})"
        );
    }
    let vocab_size = metadata_vocab
        .or(inputs.token_embd_vocab)
        .or_else(|| meta.get_arr_len("tokenizer.ggml.tokens"))
        .context(
            "GGUF: could not determine vocab_size (no '{arch}.vocab_size', no token_embd rows, \
             no 'tokenizer.ggml.tokens')",
        )?;
    if vocab_size == 0 {
        bail!("GGUF: vocab_size must be non-zero");
    }

    // ── Normalization / RoPE / context (documented explicit defaults) ──
    // rms_norm_eps: ggml default is 1e-5 when the key is absent (differs from
    // Atlas's 1e-6 default — we set it explicitly rather than inherit).
    let rms_norm_eps = meta
        .get_f64(&k("attention.layer_norm_rms_epsilon"))
        .unwrap_or(1e-5);
    // rope_theta: ggml default 10000.0.
    let rope_theta = meta.get_f64(&k("rope.freq_base")).unwrap_or(10_000.0);
    // context_length is required for a usable KV cache upper bound.
    let max_position_embeddings = req_u64("context_length")? as usize;

    // Tokenizer special tokens (0 when unset is acceptable).
    let bos_token_id = meta.get_u64("tokenizer.ggml.bos_token_id").unwrap_or(0);
    let eos_token_id = meta.get_u64("tokenizer.ggml.eos_token_id").unwrap_or(0);

    // Tied embeddings: no `output.weight` tensor ⇒ tied.
    let tie_word_embeddings = !inputs.has_output_weight;

    // ── MoE (only for MoE arches) ──
    let num_experts = if arch == "qwen3moe" {
        req_u64("expert_count")? as usize
    } else {
        meta.get_u64(&k("expert_count"))
            .map(|v| v as usize)
            .unwrap_or(0)
    };
    if arch == "qwen3moe" && num_experts == 0 {
        bail!("GGUF metadata key '{arch}.expert_count' must be greater than zero");
    }

    let mut body: Map<String, Value> = json!({
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "rms_norm_eps": rms_norm_eps,
        "rope_theta": rope_theta,
        "max_position_embeddings": max_position_embeddings,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "tie_word_embeddings": tie_word_embeddings,
        "model_type": model_type,
    })
    .as_object()
    .expect("json! object literal")
    .clone();

    if num_experts > 0 {
        let experts_per_tok = req_u64("expert_used_count").with_context(|| {
            format!("GGUF: MoE arch '{arch}' has expert_count>0 but no '{arch}.expert_used_count'")
        })? as usize;
        let moe_ffn = req_u64("expert_feed_forward_length").with_context(|| {
            format!("GGUF: MoE arch '{arch}' missing '{arch}.expert_feed_forward_length'")
        })? as usize;
        if experts_per_tok == 0 || experts_per_tok > num_experts {
            bail!(
                "GGUF metadata key '{arch}.expert_used_count' must be in 1..={num_experts}, \
                 found {experts_per_tok}"
            );
        }
        if moe_ffn == 0 {
            bail!(
                "GGUF metadata key '{arch}.expert_feed_forward_length' must be greater than zero"
            );
        }
        body.insert("num_experts".into(), json!(num_experts));
        body.insert("num_experts_per_tok".into(), json!(experts_per_tok));
        body.insert("moe_intermediate_size".into(), json!(moe_ffn));
    }

    // sliding_window (gemma hybrid attention); 0/absent ⇒ full attention.
    if let Some(sw) = meta.get_u64(&k("attention.sliding_window")) {
        body.insert("sliding_window".into(), json!(sw));
    }

    // ── Deserialize numeric fields, then set arch fields explicitly ──
    let raw = Value::Object(body);
    let json_str = serde_json::to_string(&raw).context("serialize synthesized GGUF config")?;
    let mut config: ModelConfig =
        serde_json::from_str(&json_str).context("deserialize synthesized GGUF config")?;

    config.model_type = model_type.to_string();
    config.attn_gated = attn_gated;
    // The GGUF name map emits HF names under the `model.` prefix
    // (`model.embed_tokens.weight`, `model.layers.N.*`, `model.norm.weight`).
    // `layer_prefix()` yields `model.layers.N` for both "" and "model", but the
    // embed/norm/lm_head lookups use the raw prefix — so it must be "model", not
    // "" (else they resolve to `.embed_tokens.weight` and fail).
    config.weight_prefix = "model".to_string();

    // Gemma-specific post-parse fixups.
    if model_type == "gemma4" {
        config.embed_scale = (hidden_size as f32).sqrt();
        // Logit softcap: honor the GGUF key if present (gemma2), else 0.0
        // (disabled). gemma3+ dropped softcapping.
        config.final_logit_softcapping = match meta.get_f64(&k("final_logit_softcapping")) {
            Some(v) if v >= 0.0 && v <= f32::MAX as f64 => v as f32,
            Some(v) => bail!(
                "GGUF metadata key '{}.final_logit_softcapping' must be non-negative and representable as a finite f32 (got {v})",
                arch
            ),
            None => 0.0,
        };
    }

    // ── DeepSeek-V4.1 post-parse fixups ──
    //
    // Everything below comes from the `deepseek41.*` metadata namespace. None
    // of it is derivable from the generic decoder fields above, and a V4.1
    // graph built without it is silently wrong rather than loudly broken, so
    // every key here is REQUIRED and absence is an error.
    if model_type == "deepseek_v41" {
        let req = |suffix: &str| -> Result<u64> {
            meta.get_u64(&k(suffix))
                .with_context(|| format!("DeepSeek-V4.1 GGUF missing '{arch}.{suffix}'"))
        };
        let req_arr = |suffix: &str| -> Result<Vec<u64>> {
            meta.get_u64_arr(&k(suffix))
                .with_context(|| format!("DeepSeek-V4.1 GGUF missing array '{arch}.{suffix}'"))
        };

        // MLA geometry. `attention.key_length` already became head_dim above.
        config.q_lora_rank = req("attention.q_lora_rank")? as usize;
        config.o_lora_rank = req("attention.output_lora_rank")? as usize;
        config.o_groups = req("attention.output_group_count")? as usize;
        config.kv_lora_rank = req("attention.value_length")? as usize;
        // Partial RoPE: only `rope.dimension_count` of head_dim is rotated.
        config.rotary_dim = req("rope.dimension_count")? as usize;

        // Sparse attention: the indexer, and the per-layer compression ladder.
        config.index_n_heads = req("attention.indexer.head_count")? as usize;
        config.index_head_dim = req("attention.indexer.key_length")? as usize;
        config.index_topk = req("attention.indexer.top_k")? as usize;
        config.compress_ratios = req_arr("attention.compress_ratios")?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        // 🪤 NOT a per-block array: V4.1 ships 43 entries for 40 blocks (the
        // three DSpark stages carry their own ratio). Do not assume
        // `compress_ratios.len() == num_hidden_layers`.
        if config.compress_ratios.len() < num_hidden_layers {
            bail!(
                "DeepSeek-V4.1: '{arch}.attention.compress_ratios' has {} entries, fewer than \
                 block_count ({num_hidden_layers})",
                config.compress_ratios.len()
            );
        }
        config.compress_rope_theta = meta
            .get_f64(&k("attention.compress_rope_freq_base"))
            .context("DeepSeek-V4.1 GGUF missing 'attention.compress_rope_freq_base'")?
            as f32;

        // Hyper-connections (mHC).
        config.hc_mult = req("hyper_connection.count")? as usize;
        config.hc_sinkhorn_iters = req("hyper_connection.sinkhorn_iterations")? as usize;
        config.hc_eps = meta
            .get_f64(&k("hyper_connection.epsilon"))
            .context("DeepSeek-V4.1 GGUF missing 'hyper_connection.epsilon'")?
            as f32;

        // MoE routing. `expert_gating_func` is ggml's enum; V4.1 uses
        // sqrt-softplus. Refuse an unknown value rather than defaulting to
        // softmax, which would route to the wrong experts and still generate
        // fluent-looking text.
        config.n_routed_experts = num_experts;
        config.norm_topk_prob = req("expert_weights_norm")? != 0;
        config.routed_scaling_factor = meta
            .get_f64(&k("expert_weights_scale"))
            .context("DeepSeek-V4.1 GGUF missing 'expert_weights_scale'")?;
        config.scoring_func = match req("expert_gating_func")? {
            4 => "sqrtsoftplus".to_string(),
            other => bail!(
                "DeepSeek-V4.1: unsupported '{arch}.expert_gating_func' ({other}); \
                 only 4 (sqrt-softplus) is implemented"
            ),
        };
        config.num_hash_layers = meta.get_u64(&k("hash_layer_count")).unwrap_or(0) as usize;

        // SwiGLU clamp. Shipped per-layer; Atlas carries one scalar, so require
        // the array to be uniform rather than silently taking element 0.
        if let Some(clamps) = meta.get_f64_arr(&k("swiglu_clamp_exp")) {
            let first = clamps.first().copied().unwrap_or(0.0);
            if clamps.iter().any(|v| (v - first).abs() > f64::EPSILON) {
                bail!(
                    "DeepSeek-V4.1: '{arch}.swiglu_clamp_exp' is not uniform across layers; \
                     ModelConfig carries a single `swiglu_limit`"
                );
            }
            config.swiglu_limit = first as f32;
        }

        // Engram. Absent `num_embeddings` in the GGUF (unlike the HF config):
        // the table geometry is carried by `offsets`, and row counts are read
        // from the engram tensor shapes at load time.
        config.engram_layer_ids = req_arr("engram.layer_ids")?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        if !config.engram_layer_ids.is_empty() {
            config.engram_max_ngram_size = req("engram.max_ngram_size")? as usize;
            config.engram_n_heads = req("engram.head_count")? as usize;
            config.engram_head_dim = req("engram.key_length")? as usize;
            config.engram_pad_token_id = req("engram.pad_id")? as u32;
            config.engram_multipliers = req_arr("engram.multipliers")?;
            config.engram_primes = req_arr("engram.primes")?;
            config.engram_offsets = req_arr("engram.offsets")?;
            // `engram.token_map` is 129,280 entries and is read directly by the
            // engram module at load time, not carried in ModelConfig.
            if meta.get_arr_len(&k("engram.token_map")).unwrap_or(0) != vocab_size {
                bail!(
                    "DeepSeek-V4.1: '{arch}.engram.token_map' length does not match vocab_size \
                     ({vocab_size})"
                );
            }
        }
    }

    // Reuse the shared quantization-config + validation pass.
    finalize_config(&mut config, &raw)?;
    Ok(config)
}

// ── Fields GGUF does NOT provide, and how they are set (explicit, no silent
//    prod defaults) ──
//   * partial_rotary_factor / rotary_dim: left at struct default 1.0 (full
//     RoPE). `{arch}.rope.dimension_count` could refine this for partial-rotary
//     models; deliberately NOT auto-applied here until a target arch needs it.
//   * layer_types / hybrid fields: left empty (homogeneous decoder).
//   * All SSM/MLA/DeepSeek/MiniMax/vision fields: 0 / empty — not applicable to
//     the llama/qwen/gemma decoder families this builder targets.
//   * ep_rank/ep_world_size/tp_*: set at runtime by the caller, not here.

mod kimi_k3;

#[cfg(test)]
mod tests;
