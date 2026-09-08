// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash (`glm5_next`) config parser.
//!
//! Reference checkpoint: `LibertAIDAI/GLM-5.3-Flash-NVFP4` snapshot
//! `9e0d74e3cef17f634e84fb8e2223707e02616290`. All constants asserted in the
//! tests below were read from that checkpoint's `config.json` and from a scan
//! of all 120 shard headers. Claims are scoped to that checkpoint.
//!
//! Two things make this model unlike anything already parsed here:
//!
//! 1. **NoPE MLA.** `qk_rope_head_dim` is `0`. The DeepSeek-V4 parser derives
//!    `qk_nope_head_dim = head_dim - qk_rope_head_dim` and only does so when
//!    `qk_rope_head_dim > 0`, i.e. it encodes "there is a rope section" as an
//!    invariant. We must NOT reuse it: a zero here is a fact to preserve, not a
//!    missing value to repair. Upstream vLLM's equivalent assumption is exactly
//!    what made its SM120 sparse-MLA backend structurally unusable for this
//!    model (`pe_dim must be 64 for fp8_ds_mla`).
//!
//! 2. **Hybrid layer stack.** 45 text layers alternate KDA linear attention
//!    with DeepSeek-style sparse attention, and the checkpoint states the split
//!    explicitly in `linear_attn_config.{kda_layers,full_attn_layers}`. We trust
//!    that list over any modular arithmetic, and cross-check it against the
//!    `layer_types` array when present.

use anyhow::{Context, Result, bail};

use super::super::{Glm5NextRouterMode, LayerType, ModelConfig, finalize_config};

/// `num_hidden_layers` counts text layers only; the MTP layer sits at index
/// `num_hidden_layers` (45) and is NOT included in that count.
///
/// 🪤 GLM-5.3 does **not** use DeepSeek's `mtp.0.*` naming — the MTP weights
/// live under `model.language_model.layers.45.*`. A `grep mtp` over this
/// checkpoint returns zero hits. Verified over all 120 shard headers.
pub fn glm5_next_mtp_layer_index(config: &ModelConfig) -> usize {
    config.num_hidden_layers
}

fn text_config(raw: &serde_json::Value) -> &serde_json::Value {
    raw.get("text_config").unwrap_or(raw)
}

/// GLM's own name for its sparse-MLA mixer.
///
/// Since Slice 8 this maps to [`LayerType::SparseAttention`], a real variant, and the
/// array **round-trips**: `layer_types[i].hf_name()` reproduces the checkpoint string.
/// It used to be flattened onto `FullAttention` "for scheduling purposes" — which was
/// only true while nothing scheduled on it. A sparse layer needs indexer state, an
/// indexer weight family and a per-query top-k step, so cache sizing and weight binding
/// have to be able to tell the two apart.
pub const GLM5NEXT_SPARSE_ATTN: &str = "deepseek_sparse_attention";

pub fn parse_glm5_next(json: &str) -> Result<ModelConfig> {
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON in GLM-5.3 (glm5_next) config.json")?;

    // GLM-5.3 nests everything under `text_config` (the top level carries the
    // multimodal wrapper + `quantization_config`). Parse the inner object, but
    // keep the outer one for quantization and for the architecture string.
    let text = text_config(&raw).clone();

    // `deepseek_sparse_attention` now deserializes onto `LayerType::SparseAttention` via a
    // serde alias, so the strip below is no longer required for parsing to succeed. It stays
    // because `build_layer_types` derives the array from `linear_attn_config`'s index lists
    // (the authoritative source) and then CROSS-CHECKS it against the textual array; letting
    // serde populate the field first would make that cross-check compare a value against
    // itself.
    let mut text_for_struct = text.clone();
    if let Some(obj) = text_for_struct.as_object_mut() {
        obj.remove("layer_types");
        // 🔴 GLM-5.3-Flash declares THREE stop tokens — `eos_token_id` is an ARRAY:
        // 154820 `<|endoftext|>`, 154827 `<|user|>`, 154829 `<|observation|>`. Before Slice 9
        // that made this checkpoint's config.json fail to parse outright ("invalid type:
        // sequence, expected u32"); the parser had only ever seen a hand-written fixture with
        // no `eos_token_id` at all. Handled centrally now — `eos_token_id_field` takes element
        // 0 as the primary (`tokenizer_config.json` names `<|endoftext|>` as THE `eos_token`)
        // and `parse_config` fills `eos_token_ids` with the complete set.
        // Read stop tokens through `ModelConfig::eos_ids()` / `is_eos()`, never off the scalar.
    }
    let text_json =
        serde_json::to_string(&text_for_struct).context("re-serialize glm5_next text_config")?;
    let mut config: ModelConfig =
        serde_json::from_str(&text_json).context("Failed to parse glm5_next text_config")?;
    let text = &text;

    // Canonical model_type. The inner object says `glm5_next_text`; Atlas keys
    // dispatch off the outer family name.
    config.model_type = "glm5_next".to_string();

    // ---- MLA geometry -----------------------------------------------------
    // NoPE: qk_rope_head_dim is legitimately 0 and must survive untouched.
    // Do not "repair" it, and do not derive qk_nope from it.
    let qk_rope = text
        .get("qk_rope_head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    match qk_rope {
        Some(v) => config.qk_rope_head_dim = v,
        // Absent (not zero) would mean an unfamiliar variant: refuse rather
        // than silently assume NoPE.
        None => bail!(
            "glm5_next config.json has no qk_rope_head_dim; refusing to guess \
             whether this checkpoint is NoPE"
        ),
    }
    if let Some(v) = text.get("qk_nope_head_dim").and_then(|v| v.as_u64()) {
        config.qk_nope_head_dim = v as usize;
    }
    if let Some(v) = text.get("v_head_dim").and_then(|v| v.as_u64()) {
        config.v_head_dim = v as usize;
    }
    // `head_dim` is 0 in the checkpoint. For MLA the meaningful per-head width
    // is qk_head_dim (256), NOT hidden_size / num_attention_heads (which would
    // give 64 and silently corrupt every attention shape) — the same trap the
    // DeepSeek-V4 parser documents for its own head_dim.
    if config.head_dim == 0 {
        config.head_dim = text
            .get("qk_head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(config.qk_nope_head_dim + config.qk_rope_head_dim);
    }
    // partial_rotary_factor is rope_dim / head_dim; NoPE makes it exactly 0.
    config.partial_rotary_factor = if config.head_dim > 0 {
        config.qk_rope_head_dim as f64 / config.head_dim as f64
    } else {
        0.0
    };

    // ---- MoE --------------------------------------------------------------
    if config.num_experts == 0 && config.n_routed_experts > 0 {
        config.num_experts = config.n_routed_experts;
    }
    let n_shared = text
        .get("n_shared_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if config.shared_expert_intermediate_size == 0 && n_shared > 0 {
        config.shared_expert_intermediate_size = n_shared * config.moe_intermediate_size;
    }

    // ---- KDA linear attention --------------------------------------------
    // The checkpoint carries these under `linear_attn_config`, not as the
    // flat `linear_*` keys Atlas uses for Qwen GDN.
    let lac = text.get("linear_attn_config");
    if let Some(lac) = lac {
        let g = |k: &str| lac.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
        if let Some(v) = g("num_heads") {
            config.linear_num_key_heads = v;
            config.linear_num_value_heads = v;
        }
        if let Some(v) = g("head_dim") {
            config.linear_key_head_dim = v;
            config.linear_value_head_dim = v;
        }
        if let Some(v) = g("short_conv_kernel_size") {
            config.linear_conv_kernel_dim = v;
        }
        // Bounds the log-decay `kda_gate` emits. Absent means an unfamiliar KDA variant, not
        // "no bound" — 0.0 would clamp every decay to <= 0 and is a different model.
        match lac.get("gate_lower_bound").and_then(|v| v.as_f64()) {
            Some(v) => config.linear_gate_lower_bound = v as f32,
            None => bail!(
                "glm5_next: linear_attn_config has no gate_lower_bound; refusing to guess the \
                 KDA decay bound (GLM-5.3-Flash declares -5.0)"
            ),
        }
    }

    // ---- DSA indexer ------------------------------------------------------
    // Default only if truly absent. GLM's value is 2048 and it matters: it is
    // part of what a sparse-MLA backend gates on.
    if config.index_topk == 0
        && let Some(v) = text.get("index_topk").and_then(|v| v.as_u64())
    {
        config.index_topk = v as usize;
    }
    // `index_kpool` sets the pool budget (`index_topk / index_kpool`), so a missing
    // value is not a cosmetic default — it would silently change how many candidates
    // the top-k ranks. Read it, never guess it.
    if let Some(v) = text.get("index_kpool").and_then(|v| v.as_u64()) {
        config.index_kpool = v as usize;
    }
    if let Some(v) = text
        .get("index_kpool_always_select_tail")
        .and_then(|v| v.as_bool())
    {
        config.index_kpool_always_select_tail = v;
    }

    // ---- Layer types ------------------------------------------------------
    config.layer_types = build_layer_types(text, config.num_hidden_layers)?;

    // MTP / NextN layers sit PAST the text stack. GLM-5.3-Flash declares
    // `num_nextn_predict_layers = 1`, so layer 45 exists as a real decoder layer while
    // `layer_types` legitimately covers only 0..=44. Give it its own slot rather than
    // appending it, so every "iterate the text stack" loop keeps meaning what it says.
    //
    // The config does not state the MTP block's mixer kind. Derived here as "the same
    // non-linear mixer this model uses", which the Slice-8 checkpoint audit confirms:
    // layer 45's `self_attn` tensor set is name/dtype/shape IDENTICAL to the 11
    // `deepseek_sparse_attention` text layers. Weight binding classifies from tensor
    // names anyway and does not trust this field.
    let n_mtp = text
        .get("num_nextn_predict_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if n_mtp > 0 {
        let kind = if config.layer_types.contains(&LayerType::SparseAttention) {
            LayerType::SparseAttention
        } else {
            LayerType::FullAttention
        };
        config.mtp_layer_types = vec![kind; n_mtp];
    }

    // ---- Dense-vs-routed MLP split ---------------------------------------
    // `first_k_dense_replace = 3`: layers 0..=2 carry a dense MLP, every later text layer
    // routes to experts. Nothing in Atlas read this before, so `mlp_only_layers` came out
    // EMPTY and the whole stack looked routed — a dense layer bound as MoE looks for
    // `mlp.experts.*` that do not exist. Cross-checked against the textual
    // `mlp_layer_types` array when the checkpoint carries one, the same way
    // `build_layer_types` cross-checks the mixer map.
    config.mlp_only_layers = build_mlp_only_layers(text, config.num_hidden_layers)?;

    // ---- SwiGLU clamp -----------------------------------------------------
    // 🔴 GLM clamps its SwiGLU and the clamp is ASYMMETRIC (`gate` upper-bounded only, `up`
    // both ways). Nothing in Atlas read this before, so the value would have had to be
    // hardcoded at a call site or defaulted to "no clamp" — and an absent clamp is INVISIBLE
    // on well-scaled activations, firing only on the tails. Read it, never guess it.
    config.swiglu_limit = match text.get("swiglu_limit").and_then(|v| v.as_f64()) {
        Some(v) if v > 0.0 => v as f32,
        Some(v) => bail!("glm5_next: swiglu_limit is {v}, which cannot bound anything"),
        None => bail!(
            "glm5_next config.json has no swiglu_limit; refusing to guess whether this \
             checkpoint clamps its SwiGLU (GLM-5.3-Flash declares 10.0)"
        ),
    };

    // ---- Grouped expert routing -------------------------------------------
    // `n_group`/`topk_group` are 1 here, which makes the group mask all-ones and grouped
    // routing a NO-OP. `glm5next_router_topk` implements plain top-k and takes `n_group`
    // only to REFUSE a checkpoint where the machinery would matter. Refuse at parse time
    // too, so the failure names the config rather than surfacing as a silent kernel return.
    for key in ["n_group", "topk_group"] {
        if let Some(v) = text.get(key).and_then(|v| v.as_u64())
            && v != 1
        {
            bail!(
                "glm5_next: {key} = {v}. Grouped expert routing is not implemented — \
                 glm5next_router_topk ranks every expert in one group."
            );
        }
    }

    // ---- Router dtype ladder ---------------------------------------------
    // 🔴 SEMANTIC, not precision. Default = HF 5.16.1's explicit fp32 router. The checkpoint may
    // override with `moe_router_dtype` (the same field vLLM reads); an UNRECOGNISED value is a
    // hard error, never a silent fallback onto the other semantics.
    config.glm5next_router_mode = match text.get("moe_router_dtype").and_then(|v| v.as_str()) {
        None => Glm5NextRouterMode::HfFp32,
        Some(s) => Glm5NextRouterMode::from_config_str(s).ok_or_else(|| {
            anyhow::anyhow!(
                "glm5_next: unknown moe_router_dtype {s:?}; expected float32 or bfloat16"
            )
        })?,
    };

    finalize_config(&mut config, &raw).context("glm5_next: finalize_config")?;
    refuse_shared_indexer(text, &config).context("glm5_next: indexer_types")?;
    validate_glm5_next(&config)?;
    Ok(config)
}

/// Layers whose MLP is dense rather than routed.
///
/// `first_k_dense_replace` is the authoritative knob; the textual `mlp_layer_types` array is
/// used to CROSS-CHECK it, never as a silent substitute. A disagreement is a hard error: the
/// two answers differing means the checkpoint is not the one this parser was written for.
fn build_mlp_only_layers(text: &serde_json::Value, n_layers: usize) -> Result<Vec<usize>> {
    let first_k = text
        .get("first_k_dense_replace")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let textual: Option<Vec<usize>> =
        text.get("mlp_layer_types")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .enumerate()
                    .filter(|(_, v)| v.as_str() != Some("sparse"))
                    .map(|(i, _)| i)
                    .collect()
            });

    match (first_k, textual) {
        (Some(k), t) => {
            if k > n_layers {
                bail!("glm5_next: first_k_dense_replace {k} exceeds num_hidden_layers {n_layers}");
            }
            let derived: Vec<usize> = (0..k).collect();
            if let Some(t) = t
                && t != derived
            {
                bail!(
                    "glm5_next: first_k_dense_replace={k} implies dense layers \
                     {derived:?}, but mlp_layer_types says {t:?}"
                );
            }
            Ok(derived)
        }
        // No `first_k_dense_replace`: the textual array is then the only statement of the
        // split, and it must be present and correctly sized.
        (None, Some(t)) => Ok(t),
        (None, None) => bail!(
            "glm5_next: neither first_k_dense_replace nor mlp_layer_types present; \
             refusing to guess which layers are dense"
        ),
    }
}

/// Build the per-layer mixer map.
///
/// Priority: the explicit `linear_attn_config.kda_layers` / `full_attn_layers`
/// lists, cross-checked against `layer_types` when both are present. We do not
/// fall back to `layer % 4 == 3` arithmetic — that pattern happens to hold for
/// this checkpoint but is not stated anywhere as a contract.
fn build_layer_types(text: &serde_json::Value, n_layers: usize) -> Result<Vec<LayerType>> {
    let idx_list = |key: &str| -> Option<Vec<usize>> {
        text.get("linear_attn_config")?
            .get(key)?
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            })
    };

    let kda = idx_list("kda_layers");
    let full = idx_list("full_attn_layers");

    let mut types = vec![LayerType::FullAttention; n_layers];
    match (kda, full) {
        (Some(kda), Some(full)) => {
            if kda.len() + full.len() != n_layers {
                bail!(
                    "glm5_next: kda_layers ({}) + full_attn_layers ({}) != num_hidden_layers ({})",
                    kda.len(),
                    full.len(),
                    n_layers
                );
            }
            for i in &kda {
                if *i >= n_layers {
                    bail!("glm5_next: kda_layers index {i} out of range for {n_layers} layers");
                }
                types[*i] = LayerType::LinearAttention;
            }
            // `full_attn_layers` is GLM's name for "not KDA". On GLM-5.3-Flash those layers
            // are `deepseek_sparse_attention`, so the textual array decides the variant —
            // the index list alone cannot tell sparse from dense full attention.
            let textual = text.get("layer_types").and_then(|v| v.as_array());
            for i in &full {
                if *i >= n_layers {
                    bail!("glm5_next: full_attn_layers index {i} out of range");
                }
                if types[*i] == LayerType::LinearAttention {
                    bail!("glm5_next: layer {i} listed as BOTH kda and full attention");
                }
                types[*i] = match textual.and_then(|a| a.get(*i)).and_then(|v| v.as_str()) {
                    Some(GLM5NEXT_SPARSE_ATTN) => LayerType::SparseAttention,
                    _ => LayerType::FullAttention,
                };
            }
        }
        _ => {
            // Fall back to the textual `layer_types` array.
            let arr = text
                .get("layer_types")
                .and_then(|v| v.as_array())
                .context("glm5_next: neither linear_attn_config lists nor layer_types present")?;
            if arr.len() != n_layers {
                bail!(
                    "glm5_next: layer_types has {} entries, expected {n_layers}",
                    arr.len()
                );
            }
            for (i, v) in arr.iter().enumerate() {
                types[i] = match v.as_str().unwrap_or("") {
                    "linear_attention" => LayerType::LinearAttention,
                    GLM5NEXT_SPARSE_ATTN => LayerType::SparseAttention,
                    "full_attention" => LayerType::FullAttention,
                    other => bail!("glm5_next: unknown layer_type {other:?} at layer {i}"),
                };
            }
        }
    }

    // Cross-check against layer_types when we used the index lists.
    if let Some(arr) = text.get("layer_types").and_then(|v| v.as_array())
        && arr.len() == n_layers
    {
        for (i, v) in arr.iter().enumerate() {
            let want = match v.as_str().unwrap_or("") {
                "linear_attention" => LayerType::LinearAttention,
                GLM5NEXT_SPARSE_ATTN => LayerType::SparseAttention,
                _ => LayerType::FullAttention,
            };
            if types[i] != want {
                bail!(
                    "glm5_next: layer {i} disagrees — index lists say {:?}, layer_types says {want:?}",
                    types[i]
                );
            }
        }
    }
    Ok(types)
}

/// Refuse a checkpoint whose DSA layers use **shared** indexing.
///
/// `indexer_types[i]` is `"full"` (this layer runs its own DSA indexer) or `"shared"`
/// (it reuses the previous full layer's top-k selection, which HF propagates as
/// `prev_topk_indices`). A shared layer must attend to the **upstream** layer's token
/// set; running its own indexer instead yields a plausible wrong answer with no crash
/// and no shape error, so this is refused at load rather than discovered in output.
///
/// `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3` carries an explicit 45-entry array that is
/// entirely `"full"` — its 11 DSA layers (3, 7, … 43) each run their own indexer — so
/// propagation is deliberately NOT implemented. Verified against the checkpoint config
/// 2026-08-27.
///
/// Nothing is stored: the value of this check is the refusal. Adding a config field no
/// runtime path reads would be dead surface.
///
/// When the array is absent, HF derives it, and so must we — an absent array does not
/// mean all-full. Source: `transformers` 5.16.1 `configuration_glm5_next.py`:
/// `index_topk_pattern` (an `"FSSF…"` string) wins, else
/// `"full" if (max(i - offset + 1, 0) % freq) == 0` with `freq = max(index_topk_freq, 1)`
/// and `offset = index_skip_topk_offset` (default 2).
fn refuse_shared_indexer(text: &serde_json::Value, config: &ModelConfig) -> Result<()> {
    let n = config.num_hidden_layers;
    let modes: Vec<String> = if let Some(arr) = text.get("indexer_types").and_then(|v| v.as_array())
    {
        if arr.len() != n {
            bail!(
                "indexer_types has {} entries for {n} layers; a length mismatch would \
                 silently misalign every layer's indexer mode",
                arr.len()
            );
        }
        arr.iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .context("indexer_types entry is not a string")
            })
            .collect::<Result<_>>()?
    } else if let Some(pat) = text.get("index_topk_pattern").and_then(|v| v.as_str()) {
        if pat.chars().count() != n {
            bail!(
                "index_topk_pattern has {} chars for {n} layers",
                pat.chars().count()
            );
        }
        pat.chars()
            .map(|c| match c {
                'F' => Ok("full".to_string()),
                'S' => Ok("shared".to_string()),
                other => bail!("index_topk_pattern: unknown char {other:?}, expected F or S"),
            })
            .collect::<Result<_>>()?
    } else {
        let freq = text
            .get("index_topk_freq")
            .and_then(|v| v.as_u64())
            .unwrap_or(1)
            .max(1) as i64;
        let offset = text
            .get("index_skip_topk_offset")
            .and_then(|v| v.as_i64())
            .unwrap_or(2);
        (0..n)
            .map(|i| {
                let shifted = ((i as i64) - offset + 1).max(0);
                if shifted % freq == 0 {
                    "full"
                } else {
                    "shared"
                }
                .to_string()
            })
            .collect()
    };

    for (i, m) in modes.iter().enumerate() {
        if m != "full" && m != "shared" {
            bail!("layer {i}: unknown indexer mode {m:?}, expected \"full\" or \"shared\"");
        }
    }

    // Only DSA layers have an indexer at all; a "shared" entry on a KDA layer is inert.
    let shared: Vec<usize> = config
        .layer_types
        .iter()
        .enumerate()
        .filter(|(i, t)| {
            **t != LayerType::LinearAttention && modes.get(*i).is_some_and(|m| m == "shared")
        })
        .map(|(i, _)| i)
        .collect();
    if !shared.is_empty() {
        bail!(
            "DSA layer(s) {shared:?} use SHARED indexing (reuse the previous full layer's \
             top-k). Atlas runs a per-layer indexer and does not propagate selections, so \
             these layers would attend to the wrong token set — a wrong answer, not a \
             crash. GLM-5.3-Flash-NVFP4 is entirely \"full\"; implement prev_topk_indices \
             propagation before serving a checkpoint that is not."
        );
    }
    Ok(())
}

fn validate_glm5_next(config: &ModelConfig) -> Result<()> {
    if config.qk_rope_head_dim != 0 {
        bail!(
            "glm5_next: expected NoPE (qk_rope_head_dim == 0), got {}. \
             A non-zero value means this is not the GLM-5.3 geometry we support.",
            config.qk_rope_head_dim
        );
    }
    if config.head_dim == 0 {
        bail!("glm5_next: head_dim resolved to 0");
    }
    let linear = config
        .layer_types
        .iter()
        .filter(|t| **t == LayerType::LinearAttention)
        .count();
    let full = config.layer_types.len() - linear;
    if linear == 0 || full == 0 {
        bail!("glm5_next: degenerate layer map — {linear} linear / {full} full");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from the real checkpoint config.json (LibertAI NVFP4 @ 9e0d74e3).
    /// Values are verbatim; only unrelated keys were dropped for size.
    fn glm53_config_json() -> String {
        // KDA layers are every index NOT congruent to 3 (mod 4); the real file
        // enumerates them explicitly and we mirror that here.
        let kda: Vec<String> = (0..45)
            .filter(|i| i % 4 != 3)
            .map(|i| i.to_string())
            .collect();
        let full: Vec<String> = (0..45)
            .filter(|i| i % 4 == 3)
            .map(|i| i.to_string())
            .collect();
        let layer_types: Vec<String> = (0..45)
            .map(|i| {
                if i % 4 == 3 {
                    "\"deepseek_sparse_attention\"".to_string()
                } else {
                    "\"linear_attention\"".to_string()
                }
            })
            .collect();
        format!(
            r#"{{
  "architectures": ["Glm5NextForConditionalGeneration"],
  "model_type": "glm5_next",
  "text_config": {{
    "model_type": "glm5_next_text",
    "num_hidden_layers": 45,
    "num_nextn_predict_layers": 1,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "head_dim": 0,
    "qk_head_dim": 256,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "mla_use_nope": true,
    "index_topk": 2048,
    "index_kpool": 4,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "mhc": true,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "first_k_dense_replace": 3,
    "swiglu_limit": 10.0,
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": true,
    "n_group": 1,
    "topk_group": 1,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "rms_norm_eps": 1e-05,
    "vocab_size": 154880,
    "max_position_embeddings": 1048576,
    "linear_attn_config": {{
      "num_heads": 64,
      "head_dim": 128,
      "short_conv_kernel_size": 4,
      "gate_lower_bound": -5.0,
      "kda_layers": [{kda}],
      "full_attn_layers": [{full}]
    }},
    "layer_types": [{lt}]
  }}
}}"#,
            kda = kda.join(","),
            full = full.join(","),
            lt = layer_types.join(",")
        )
    }

    /// Inject a key into `text_config` and re-serialise, so a test can vary one
    /// checkpoint field without restating the whole fixture.
    fn with_text_key(key: &str, value: serde_json::Value) -> String {
        let mut raw: serde_json::Value =
            serde_json::from_str(&glm53_config_json()).expect("fixture json");
        raw["text_config"][key] = value;
        raw.to_string()
    }

    /// The real checkpoint's array: 45 entries, every one `"full"`.
    #[test]
    fn an_all_full_indexer_array_is_accepted() {
        let all_full = serde_json::Value::from(vec!["full"; 45]);
        let c = parse_glm5_next(&with_text_key("indexer_types", all_full)).expect("parse");
        // The 11 DSA layers of GLM-5.3-Flash.
        let dsa: Vec<usize> = c
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| **t != LayerType::LinearAttention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(dsa, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]);
    }

    /// 🔴 A shared DSA layer must attend to the UPSTREAM layer's token set. Atlas runs a
    /// per-layer indexer and does not propagate, so this is a wrong answer with no crash
    /// — refused at load.
    #[test]
    fn a_shared_dsa_layer_is_refused() {
        let mut modes = vec!["full"; 45];
        modes[7] = "shared"; // a real DSA layer
        let e = parse_glm5_next(&with_text_key("indexer_types", modes.into()))
            .expect_err("shared DSA indexing must be refused");
        let msg = e.to_string() + &e.root_cause().to_string();
        assert!(msg.contains('7'), "the error must name the layer: {msg}");
    }

    /// A `"shared"` entry on a KDA layer is inert — those layers have no indexer at all.
    /// Refusing it would reject a legal checkpoint.
    #[test]
    fn a_shared_entry_on_a_linear_layer_is_inert() {
        let mut modes = vec!["full"; 45];
        modes[0] = "shared"; // layer 0 is KDA
        assert!(parse_glm5_next(&with_text_key("indexer_types", modes.into())).is_ok());
    }

    /// A length mismatch would silently misalign every layer's mode.
    #[test]
    fn a_wrong_length_indexer_array_is_refused() {
        let short = serde_json::Value::from(vec!["full"; 44]);
        assert!(parse_glm5_next(&with_text_key("indexer_types", short)).is_err());
    }

    /// An ABSENT array does not mean all-full: HF derives it. With the default
    /// `freq = 1` every layer is full, which is why the bare fixture parses — but a
    /// `freq = 4` schedule genuinely produces shared DSA layers and must be refused.
    #[test]
    fn an_absent_array_is_derived_not_assumed_full() {
        // freq=1 (the default) → all full → parses.
        assert!(parse_glm5_next(&glm53_config_json()).is_ok());
        // freq=4, offset=2 → full only where (max(i-1,0) % 4 == 0); DSA layer 7 is shared.
        let e = parse_glm5_next(&with_text_key("index_topk_freq", 4.into()))
            .expect_err("a freq schedule that shares DSA layers must be refused");
        assert!(e.root_cause().to_string().contains("SHARED"), "{e}");
    }

    /// The `"FSSF…"` pattern string overrides the freq schedule, as in HF.
    #[test]
    fn an_index_topk_pattern_is_honoured() {
        // All-F pattern parses; one S on a DSA layer does not.
        let ok: String = "F".repeat(45);
        assert!(parse_glm5_next(&with_text_key("index_topk_pattern", ok.into())).is_ok());
        let mut bad: Vec<char> = "F".repeat(45).chars().collect();
        bad[43] = 'S'; // the last DSA layer
        let bad: String = bad.into_iter().collect();
        assert!(parse_glm5_next(&with_text_key("index_topk_pattern", bad.into())).is_err());
    }

    #[test]
    fn parses_glm5_next() {
        let c = parse_glm5_next(&glm53_config_json()).expect("parse");
        assert_eq!(c.model_type, "glm5_next");
        assert_eq!(c.num_hidden_layers, 45);
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.n_routed_experts, 288);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.moe_intermediate_size, 2048);
    }

    /// The acceptance criterion: a legitimate zero must round-trip untouched.
    #[test]
    fn nope_rope_dim_zero_survives_exactly() {
        let c = parse_glm5_next(&glm53_config_json()).expect("parse");
        assert_eq!(c.qk_rope_head_dim, 0, "NoPE zero must not be 'repaired'");
        assert_eq!(c.qk_nope_head_dim, 256, "nope dim must come from the file");
        assert_eq!(c.v_head_dim, 256);
        assert_eq!(c.partial_rotary_factor, 0.0);
    }

    /// 🪤 GLM-5.3 is **MLA AND NoPE at the same time** — the exact combination
    /// that broke Atlas's MLA decode dispatch.
    ///
    /// `qwen3_attention/decode/run_paged_decode.rs` used to select the MLA
    /// compressed-cache decode with `mla.rope > 0`, treating "has a RoPE
    /// section" as a proxy for "is MLA". That holds for DeepSeek-V4-Flash
    /// (rope=64) and fails here: GLM has a 512-dim latent cache and rope=0, so
    /// it satisfied "is MLA" while failing the proxy, and would have fallen
    /// through to the generic *non-MLA* paged decode carrying an MLA-shaped
    /// cache — a wrong answer with no crash.
    ///
    /// The predicate is now `mla.is_some()`, and only the genuine RoPE
    /// operations are guarded on `rope > 0`. This test pins the config-level
    /// fact that makes the distinction necessary; if it ever fails, re-read that
    /// dispatch before touching anything else.
    #[test]
    fn is_mla_and_nope_simultaneously() {
        let c = parse_glm5_next(&glm53_config_json()).expect("parse");
        assert!(
            c.kv_lora_rank > 0,
            "GLM-5.3 is MLA: a latent KV cache of {} dims",
            c.kv_lora_rank
        );
        assert_eq!(c.kv_lora_rank, 512);
        assert_eq!(
            c.qk_rope_head_dim, 0,
            "...and simultaneously NoPE. `rope > 0` must never stand in for `is MLA`."
        );
    }

    /// head_dim=0 in-file must resolve to qk_head_dim (256), never to
    /// hidden_size / num_attention_heads (64).
    #[test]
    fn head_dim_resolves_to_mla_width_not_hidden_over_heads() {
        let c = parse_glm5_next(&glm53_config_json()).expect("parse");
        assert_eq!(c.head_dim, 256);
        assert_ne!(c.head_dim, 4096 / 64);
    }

    /// Reconciled census, TEXT LAYERS ONLY (0..44): 34 KDA + 11 DSA.
    /// See .planning/ATLAS-GLM5NEXT-SKILL-RECONCILIATION-20260826.md — the
    /// whole-checkpoint totals differ because layer 45 (MTP) is DSA-shaped.
    #[test]
    fn layer_census_matches_reconciled_counts() {
        let c = parse_glm5_next(&glm53_config_json()).expect("parse");
        let kda = c
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::LinearAttention)
            .count();
        // Since Slice 8 the DSA layers are `SparseAttention`, not `FullAttention` — and
        // there must be ZERO plain full-attention layers, or something was flattened.
        let dsa = c
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::SparseAttention)
            .count();
        let plain_full = c
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count();
        assert_eq!(c.layer_types.len(), 45);
        assert_eq!(kda, 34, "KDA layers over text layers 0..44");
        assert_eq!(dsa, 11, "DSA layers over text layers 0..44");
        assert_eq!(plain_full, 0, "GLM-5.3 has no plain full-attention layer");
        // Spot-check the actual indices, not just the totals.
        assert_eq!(c.layer_types[0], LayerType::LinearAttention);
        assert_eq!(c.layer_types[3], LayerType::SparseAttention);
        assert_eq!(c.layer_types[43], LayerType::SparseAttention);
        assert_eq!(c.layer_types[44], LayerType::LinearAttention);
    }

    /// `layer_types` must round-trip back to the checkpoint's own vocabulary. This is what
    /// "flattened onto FullAttention" used to break: the parse succeeded and the array
    /// silently said `full_attention` where the checkpoint said `deepseek_sparse_attention`.
    #[test]
    fn layer_types_round_trip_to_the_checkpoint_strings() {
        let c = parse_glm5_next(&glm53_config_json()).expect("parse");
        let raw: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
        let want = raw["text_config"]["layer_types"]
            .as_array()
            .expect("layer_types");
        assert_eq!(want.len(), c.layer_types.len());
        for (i, w) in want.iter().enumerate() {
            assert_eq!(
                c.layer_types[i].hf_name(),
                w.as_str().unwrap(),
                "layer {i} does not round-trip"
            );
        }
    }

    /// Layer 45 (MTP) is a real decoder layer that is NOT part of the text stack.
    /// It must be representable without being appended to `layer_types`.
    #[test]
    fn mtp_layer_is_represented_outside_the_text_stack() {
        let c = parse_glm5_next(&glm53_config_json()).expect("parse");
        assert_eq!(c.num_hidden_layers, 45);
        assert_eq!(c.layer_types.len(), 45, "text stack stays 0..=44");
        assert_eq!(c.mtp_layer_types, vec![LayerType::SparseAttention]);
        // Index 45 resolves, and it resolves through the MTP list, not the text stack.
        assert_eq!(c.layer_type_at(45), Some(LayerType::SparseAttention));
        assert_eq!(c.layer_type_at(46), None);
        assert!(c.has_sparse_attention());
        assert_eq!(c.sparse_attention_layers().len(), 11, "text stack only");
    }

    #[test]
    fn kda_geometry_from_linear_attn_config() {
        let c = parse_glm5_next(&glm53_config_json()).expect("parse");
        assert_eq!(c.linear_num_key_heads, 64);
        assert_eq!(c.linear_key_head_dim, 128);
        assert_eq!(c.linear_conv_kernel_dim, 4);
    }

    #[test]
    fn indexer_topk_is_2048_not_the_deepseek_default() {
        let c = parse_glm5_next(&glm53_config_json()).expect("parse");
        assert_eq!(c.index_topk, 2048);
    }

    #[test]
    fn missing_rope_key_is_refused_not_guessed() {
        let mut v: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
        v["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("qk_rope_head_dim");
        let err = parse_glm5_next(&v.to_string()).unwrap_err();
        assert!(
            err.to_string().contains("refusing to guess"),
            "unexpected error: {err}"
        );
    }

    /// GLM's SwiGLU clamp is asymmetric and fires only on the activation tails, so a
    /// checkpoint that declares it and a parser that defaults it disagree SILENTLY on
    /// well-scaled inputs. The parser must refuse rather than fall back to "no clamp".
    #[test]
    fn a_missing_swiglu_limit_is_refused_not_defaulted() {
        let mut v: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
        v["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("swiglu_limit");
        let err = parse_glm5_next(&v.to_string()).unwrap_err();
        assert!(
            err.to_string().contains("swiglu_limit"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn the_swiglu_limit_is_read_verbatim() {
        let c = parse_glm5_next(&glm53_config_json()).unwrap();
        assert_eq!(c.swiglu_limit, 10.0);
    }

    /// `n_group`/`topk_group` > 1 would make the group mask load-bearing, and
    /// `glm5next_router_topk` returns without writing rather than implementing it. Refuse at
    /// parse time so the failure names the config, not an empty selection row.
    #[test]
    fn grouped_expert_routing_is_refused() {
        for key in ["n_group", "topk_group"] {
            let mut v: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
            v["text_config"][key] = serde_json::json!(8);
            let err = parse_glm5_next(&v.to_string()).unwrap_err();
            assert!(
                err.to_string().contains("Grouped expert routing"),
                "{key}: unexpected error: {err}"
            );
        }
    }

    #[test]
    fn contradictory_layer_maps_are_rejected() {
        let mut v: serde_json::Value = serde_json::from_str(&glm53_config_json()).unwrap();
        // Claim layer 0 is full attention in layer_types while the index list
        // says KDA — must not be silently resolved.
        v["text_config"]["layer_types"][0] =
            serde_json::Value::String("deepseek_sparse_attention".into());
        let err = parse_glm5_next(&v.to_string()).unwrap_err();
        assert!(err.to_string().contains("disagrees"), "unexpected: {err}");
    }

    // ───────────────────────────────────────────── GLM router dtype ladder (Slice 10)

    /// 🔴 The production default is HF's fp32 router. An absent `moe_router_dtype` means "the
    /// checkpoint did not say", and the reference implementation's answer for that is fp32 — the
    /// OPPOSITE of vLLM's fallthrough to bf16.
    #[test]
    fn glm_router_defaults_to_hf_fp32_when_the_config_is_silent() {
        let cfg = parse_glm5_next(&glm53_config_json()).unwrap();
        assert_eq!(cfg.glm5next_router_mode, Glm5NextRouterMode::HfFp32);
        assert!(cfg.glm5next_router_mode.is_fp32());
    }

    /// The compatibility mode must be reachable EXPLICITLY, from its own field — never inferred
    /// from quantization settings or from the weight-storage precision schedule.
    #[test]
    fn glm_router_bf16_compat_mode_is_explicit_and_never_inferred() {
        let base = glm53_config_json();
        let with = base.replace(
            r#""hc_mult": 4,"#,
            r#""hc_mult": 4, "moe_router_dtype": "bfloat16","#,
        );
        assert_ne!(with, base, "fixture anchor moved");
        let cfg = parse_glm5_next(&with).unwrap();
        assert_eq!(cfg.glm5next_router_mode, Glm5NextRouterMode::VllmBf16);
        assert!(!cfg.glm5next_router_mode.is_fp32());

        let fp32 = base.replace(
            r#""hc_mult": 4,"#,
            r#""hc_mult": 4, "moe_router_dtype": "float32","#,
        );
        assert_eq!(
            parse_glm5_next(&fp32).unwrap().glm5next_router_mode,
            Glm5NextRouterMode::HfFp32
        );
    }

    /// An unrecognised value must be a hard error. Silently falling back would pick one of two
    /// semantics at random, which is the whole defect this switch exists to prevent.
    #[test]
    fn an_unknown_router_dtype_is_refused_not_defaulted() {
        let bad = glm53_config_json().replace(
            r#""hc_mult": 4,"#,
            r#""hc_mult": 4, "moe_router_dtype": "fp8","#,
        );
        assert!(parse_glm5_next(&bad).is_err());
    }
}
