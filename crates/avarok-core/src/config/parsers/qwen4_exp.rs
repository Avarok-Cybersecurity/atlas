// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3.8-Flash-Next` (`model_type: qwen4_exp`) configuration parser.
//!
//! `Qwen4ExpForConditionalGeneration` — a ~180B hybrid multimodal MoE. Four
//! features put it outside the qwen3_5/qwen3_6 nested-config path, which is
//! why it gets its own parser rather than another arm of `dispatch.rs`:
//!
//!   1. **Hyper-connections.** The residual is `hc_count` (4) parallel
//!      streams, mixed by a LOW-RANK pair of rank `hc_lowrank` (320). Atlas
//!      already carries the stream-major `[T, hc_mult, H]` plumbing for
//!      DeepSeek-V4, whose mixer is Sinkhorn-normalized instead — so
//!      `hc_mult` maps straight across and `hc_lowrank` selects the variant.
//!   2. **A QSA indexer** on the full-attention layers, which is
//!      DeepSeek-V4's semantic indexer under different key names
//!      (`indexer_*` -> `index_*`).
//!   3. **PLE n-gram injection** at ONE decoder layer, rather than LongCat's
//!      fusion into the token embedding.
//!   4. **`layer_types`** interleaving GDN and full attention 3:1.
//!
//! Everything else — 512-expert MoE with a shared expert, mRoPE, the ViT
//! tower, gated attention — lands on fields Atlas already has.
//!
//! Reference: `transformers` 5.8.0.dev0 `modeling_qwen4_exp.py`. The HF
//! repos ship no `.py`, so the modeling code is the transformers tree.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::ModelConfig;
use super::vision::parse_vision_config;

/// Rows-per-head that `ngram_vocab_size_base` is rounded UP to a multiple of
/// (`make_ngram_vocab_size_divisible_by`). Only used to sanity-check the
/// config against itself; the loader reads the real sizes off the checkpoint.
const NGRAM_VOCAB_ALIGN: usize = 128;

pub(crate) fn parse_qwen4_exp(raw: &Value) -> Result<ModelConfig> {
    let text = raw
        .get("text_config")
        .context("qwen4_exp config.json missing text_config")?;

    let mut config: ModelConfig =
        serde_json::from_value(text.clone()).context("Failed to parse qwen4_exp text_config")?;

    // text_config's own model_type is "qwen4_exp_text"; the engine keys
    // kernel-target resolution off the TOP-level name.
    config.model_type = "qwen4_exp".to_string();
    config.nested_config = true;
    // The reference `Qwen4ExpTextTopKRouter` renormalizes the top-k probs
    // (`Qwen4ExpTextConfig.norm_topk_prob` defaults TRUE and the checkpoint
    // json OMITS the key). Serde's bool default is false, so without this the
    // MoE weighted sum runs on RAW top-10 softmax probs — each token's expert
    // contribution scaled by its top-10 mass (measured 0.56-0.93, mean 0.84
    // on the 55-token bisect fixture), compounding over 48 layers into a ~5x
    // highway deficit and temperature-1.0 sampling salad while argmax stays
    // coherent. Same trap minimax.rs documents; found via
    // bench/qwen4_exp/{forward_ref,slice_ref}.py: GDN sublayer cos 0.99999,
    // MoE-delta cos 0.93 ratio 0.84.
    config.norm_topk_prob = true;
    // No `model.norm.weight` in the checkpoint; the reference forward feeds
    // the collapsed highway straight into lm_head. See ModelConfig docs.
    config.final_norm_identity = true;
    // Multimodal wrapper: every decoder tensor is under
    // `model.language_model.`, and the ViT under `model.visual.`. Setting the
    // prefix here means `config.layer_prefix(i)` yields the real key, so the
    // shared loader helpers need no qwen4_exp-specific naming.
    config.weight_prefix = "model.language_model".to_string();

    ensure!(
        config.hidden_size > 0,
        "qwen4_exp hidden_size must be non-zero"
    );
    ensure!(
        config.num_hidden_layers > 0,
        "qwen4_exp num_hidden_layers must be non-zero"
    );
    ensure!(
        config.vocab_size > 0,
        "qwen4_exp vocab_size must be non-zero"
    );

    // eos_token_id may be an array; take the primary.
    if config.eos_token_id == 0
        && let Some(eos) = text.get("eos_token_id")
    {
        let primary = match eos {
            Value::Number(n) => n.as_u64(),
            Value::Array(ids) => ids.first().and_then(Value::as_u64),
            _ => None,
        };
        config.eos_token_id = primary.unwrap_or(0) as u32;
    }

    parse_rope(text, &mut config);
    parse_hyper_connections(text, &mut config)?;
    parse_indexer(text, &mut config)?;
    parse_ngram_ple(text, &mut config)?;

    // `output_gate_type: "sigmoid"` — q_proj emits [q | gate], so its row
    // count is 2x n_heads*head_dim. Same shape Qwen3-Next uses.
    config.attn_gated = text
        .get("output_gate_type")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty() && s != "none");

    // ...and the SAME field drives the GDN gated-norm's activation: the
    // reference passes `output_gate_type or hidden_act` into `RMSNormGated`.
    // SiLU is the family default Atlas's kernels hardcode; "sigmoid" here
    // selects the sigmoid twins at layer init.
    config.gdn_norm_sigmoid = text
        .get("output_gate_type")
        .and_then(Value::as_str)
        .is_some_and(|s| s == "sigmoid");

    // The ViT tower lives at the TOP level, not inside text_config.
    if raw.get("vision_config").is_some() {
        config.vision = parse_vision_config(raw);
    }

    ensure!(
        !config.layer_types.is_empty(),
        "qwen4_exp requires an explicit layer_types array (48 entries \
         interleaving linear_attention and full_attention); deriving it from \
         full_attention_interval would be a guess"
    );
    ensure!(
        config.layer_types.len() == config.num_hidden_layers,
        "qwen4_exp layer_types has {} entries but num_hidden_layers is {}",
        config.layer_types.len(),
        config.num_hidden_layers,
    );

    Ok(config)
}

/// mRoPE + partial rotary live under `rope_parameters`, not at text_config
/// top level, so serde's derive does not reach them.
fn parse_rope(text: &Value, config: &mut ModelConfig) {
    let Some(rp) = text.get("rope_parameters") else {
        return;
    };
    if let Some(theta) = rp.get("rope_theta").and_then(Value::as_f64) {
        config.rope_theta = theta;
    }
    if let Some(prf) = rp.get("partial_rotary_factor").and_then(Value::as_f64) {
        config.partial_rotary_factor = prf;
    }
    config.mrope_interleaved = rp
        .get("mrope_interleaved")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(sec) = rp.get("mrope_section").and_then(Value::as_array) {
        for (i, v) in sec.iter().take(3).enumerate() {
            config.mrope_section[i] = v.as_u64().unwrap_or(0) as usize;
        }
    }
}

/// `hc_count` / `hc_lowrank` -> `hc_mult` / `hc_lowrank`.
fn parse_hyper_connections(text: &Value, config: &mut ModelConfig) -> Result<()> {
    config.hc_mult = text.get("hc_count").and_then(Value::as_u64).unwrap_or(0) as usize;
    config.hc_lowrank = text.get("hc_lowrank").and_then(Value::as_u64).unwrap_or(0) as usize;
    // Both or neither. A model with streams but no mixer rank would silently
    // fall into DeepSeek-V4's Sinkhorn path and mix with the wrong math.
    ensure!(
        (config.hc_mult == 0) == (config.hc_lowrank == 0),
        "qwen4_exp hyper-connection config is partial (hc_count={}, \
         hc_lowrank={}) — both must be present together",
        config.hc_mult,
        config.hc_lowrank,
    );
    Ok(())
}

/// `indexer_*` -> the DeepSeek-V4 `index_*` fields, and a per-layer
/// `compress_ratios` vector keyed off `layer_types`.
fn parse_indexer(text: &Value, config: &mut ModelConfig) -> Result<()> {
    let g = |k: &str| text.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
    config.index_n_heads = g("indexer_n_heads");
    config.index_head_dim = g("indexer_head_dim");
    config.index_topk = g("indexer_budget");
    let ratio = g("indexer_compress_ratio");

    let present = [
        config.index_n_heads,
        config.index_head_dim,
        config.index_topk,
    ]
    .iter()
    .filter(|&&v| v > 0)
    .count();
    ensure!(
        present == 0 || present == 3,
        "qwen4_exp indexer config is partial (n_heads={}, head_dim={}, \
         budget={}) — all three must be present together",
        config.index_n_heads,
        config.index_head_dim,
        config.index_topk,
    );

    // `compress_ratios` is deliberately LEFT EMPTY while the QSA indexer is
    // unwired, because a non-empty value turns on `probes.compressed_attn`
    // and dispatches DeepSeek-V4's compressor — a different mechanism.
    //
    // Below the budget that costs nothing, and not as an approximation we are
    // choosing to tolerate: the reference selects
    // `topk(min(block_topk, num_complete_blocks))` with
    // `block_topk = indexer_budget / compress_ratio` (512 here). At
    // `seq_len <= indexer_budget` there are at most `block_topk` complete
    // blocks, so `min()` takes ALL of them and every token stays visible.
    // Dense attention is EXACT in that range.
    //
    // ABOVE it the indexer becomes load-bearing, so the ratio is recorded on
    // `index_compress_ratio` for the loader to refuse against rather than
    // quietly attending densely. See bench/qwen4_exp/ARCHITECTURE.md §3.
    config.index_compress_ratio = ratio;
    Ok(())
}

/// n-gram / PLE geometry.
///
/// LongCat and Qwen4-Exp describe the same mechanism with different keys:
/// `emb_neighbor_num`/`emb_split_num` here come from `ngram_size` and
/// `heads_per_ngram`, and the head count is `heads_per_ngram * (ngram_size - 1)`
/// exactly as LongCat's is `emb_split_num * (emb_neighbor_num - 1)`.
fn parse_ngram_ple(text: &Value, config: &mut ModelConfig) -> Result<()> {
    let g = |k: &str| text.get(k).and_then(Value::as_u64).unwrap_or(0) as usize;
    let ngram_size = g("ngram_size");
    let heads_per_ngram = g("heads_per_ngram");
    config.ngram_vocab_size_base = g("ngram_vocab_size_base");
    config.ngram_split_parts = g("split_ngram_parts");
    config.ple_conv_kernel_size = g("ple_conv_kernel_size");
    if let Some(ids) = text.get("ple_layer_ids").and_then(Value::as_array) {
        config.ple_layer_ids = ids
            .iter()
            .filter_map(Value::as_u64)
            .map(|v| v as usize)
            .collect();
    }

    if ngram_size == 0 && heads_per_ngram == 0 && config.ple_layer_ids.is_empty() {
        return Ok(()); // not an n-gram checkpoint
    }

    ensure!(
        ngram_size >= 2,
        "qwen4_exp ngram_size must be >= 2, got {ngram_size}"
    );
    ensure!(
        heads_per_ngram > 0,
        "qwen4_exp heads_per_ngram must be non-zero when ngram_size is set"
    );
    config.emb_neighbor_num = ngram_size;
    config.emb_split_num = heads_per_ngram;

    // 16 heads x 160 dims = 2560 = hidden_size. Unlike LongCat — which
    // PROJECTS each 256-dim table row up to hidden and sums — Qwen4-Exp
    // CONCATENATES the per-head slices, so the head count must divide
    // hidden_size exactly or the concat cannot reconstruct a hidden vector.
    let heads = heads_per_ngram * (ngram_size - 1);
    ensure!(
        config.hidden_size.is_multiple_of(heads),
        "qwen4_exp hidden_size {} must divide evenly by the {} n-gram heads \
         (heads_per_ngram {} x (ngram_size {} - 1)) — the per-head slices are \
         concatenated, not projected",
        config.hidden_size,
        heads,
        heads_per_ngram,
        ngram_size,
    );

    ensure!(
        !config.ple_layer_ids.is_empty(),
        "qwen4_exp declares n-gram tables but no ple_layer_ids — nothing \
         would consume them"
    );
    for &l in &config.ple_layer_ids {
        ensure!(
            l < config.num_hidden_layers,
            "qwen4_exp ple_layer_ids contains layer {l} but the model has \
             only {} layers",
            config.num_hidden_layers,
        );
    }

    if config.ngram_vocab_size_base > 0 {
        let align = text
            .get("make_ngram_vocab_size_divisible_by")
            .and_then(Value::as_u64)
            .unwrap_or(NGRAM_VOCAB_ALIGN as u64) as usize;
        ensure!(
            align > 0 && config.ngram_vocab_size_base.is_multiple_of(align),
            "qwen4_exp ngram_vocab_size_base {} is not a multiple of \
             make_ngram_vocab_size_divisible_by {}",
            config.ngram_vocab_size_base,
            align,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "qwen4_exp_tests.rs"]
mod qwen4_exp_tests;
