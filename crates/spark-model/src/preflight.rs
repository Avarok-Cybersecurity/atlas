// SPDX-License-Identifier: AGPL-3.0-only
//
//! Pre-flight weight-store / config consistency checks.
//!
//! Runs **before** NCCL init and model construction so obvious checkpoint
//! mismatches (wrong expert count, missing `lm_head`, MiniMax checkpoint
//! shipped with MTP tensors that the loader can't consume, etc.) fail fast
//! with a readable error instead of surfacing later as:
//!
//!   * an `ncclCommInitRank` hang (when only one rank bails before
//!     reaching collective init),
//!   * a cryptic `build_model` error ~10 minutes into startup,
//!   * an opaque "Received NCCL unique ID from master" log trail with
//!     no explanation — the failure mode Discord users have been posting.
//!
//! The checks are intentionally cheap: they only consult tensor NAMES
//! already in the `WeightStore` (loaded lazily by the safetensors index
//! pass), never touch GPU memory, and never issue collectives. Safe to
//! call on every rank.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

/// Run all model-agnostic + model-type-specific pre-flight checks.
///
/// Called by `spark-server/src/main.rs` immediately after the
/// `WeightStore` is populated and before `spark_comm::NcclBackend::new`
/// runs, so a bad checkpoint aborts rank 0 before rank 1 even connects.
///
/// `use_speculative` is the final resolved flag (user `--speculative`
/// OR model default). It gates MiniMax-specific MTP-presence diagnostics:
/// if the user didn't ask for speculative decoding, MTP tensors in the
/// checkpoint are harmless dead weight and do not warrant a bail.
pub fn preflight(store: &WeightStore, config: &ModelConfig, use_speculative: bool) -> Result<()> {
    // Model-agnostic checks — driven purely by `store.names()` and
    // `config` (which already carries the parsed `config.json` values).
    check_quant_method(config)?;
    check_embedding_and_head(store)?;
    let max_layer_idx = check_layer_count(store, config)?;
    check_expert_count(store, config)?;
    check_correction_bias_shape(store, config)?;

    // If the checkpoint has layers beyond `config.num_hidden_layers` AND
    // the user asked for speculative decoding, warn/error about MTP
    // consumability. The only model family where Atlas currently bails
    // rather than ignoring extra MTP layers is MiniMax — but the check
    // itself is discovery-based, not name-based.
    if use_speculative && max_layer_idx + 1 > config.num_hidden_layers {
        check_mtp_consumability(config)?;
    }

    tracing::info!("Pre-flight checks passed");
    Ok(())
}

/// Fail fast when the checkpoint declares a `quant_method` Atlas doesn't
/// understand. Discovery-based fallback at load time would then either
/// silently mis-detect the format (the Discord 2026-04-17 bug) or die
/// with a cryptic dtype error. A clear error here beats either.
fn check_quant_method(config: &ModelConfig) -> Result<()> {
    let Some(qc) = &config.quantization_config else {
        return Ok(());
    };
    // Empty method strings are fine — the config-parser already skips
    // fully-empty blocks; a non-empty ignore list with empty method
    // just means "use heuristic detection on the tensor names".
    if qc.quant_method.is_empty() {
        return Ok(());
    }
    const KNOWN_METHODS: &[&str] = &["compressed-tensors", "modelopt", "fp8"];
    if !KNOWN_METHODS.contains(&qc.quant_method.as_str()) {
        bail!(
            "Pre-flight: checkpoint declares quant_method={:?} which Atlas doesn't \
             recognize. Supported schemes: {:?}. If this is a new NVIDIA/HF format, \
             add an impl of `QuantFormat` in `crates/spark-model/src/quant_format/` \
             and extend `detect_quant_format`. See `docs/EP2-TROUBLESHOOTING.md`.",
            qc.quant_method,
            KNOWN_METHODS,
        );
    }
    Ok(())
}

fn check_embedding_and_head(store: &WeightStore) -> Result<()> {
    // Three canonical embedding-tensor naming schemes across the
    // families Atlas supports:
    //   `*.embed_tokens.weight`  — HF standard (Qwen, Gemma, MiniMax)
    //   `*.embeddings.weight`    — Nemotron-H backbone prefix
    //   `tok_embeddings.weight`  — Mistral consolidated checkpoints
    // Scan discovery-based so future families that adopt yet-another
    // spelling only need to appear as a new suffix here — no enumerated
    // prefix list to maintain.
    const EMBED_SUFFIXES: &[&str] = &[".embed_tokens.weight", ".embeddings.weight"];
    let has_embed = store
        .names()
        .any(|n| n == "tok_embeddings.weight" || EMBED_SUFFIXES.iter().any(|s| n.ends_with(s)));
    if !has_embed {
        bail!(
            "Pre-flight: no embedding tensor found (checked suffixes: \
             {EMBED_SUFFIXES:?} and bare `tok_embeddings.weight`). \
             Is this a language-model checkpoint?"
        );
    }
    // LM head is optional (tied embeddings skip it). Scan suffixes:
    //   `lm_head.weight`       — HF / Qwen / Gemma / MiniMax
    //   `output.weight`        — Mistral consolidated
    let has_head = store
        .names()
        .any(|n| n.ends_with("lm_head.weight") || n == "output.weight");
    if !has_head {
        tracing::info!("Pre-flight: no dedicated LM head tensor; assuming tied embeddings.");
    }
    Ok(())
}

/// Detect the highest per-layer index present in the store by scanning
/// any tensor matching `*.layers.{N}.*`. Returns the observed max index
/// so the caller can decide whether extra MTP layers were shipped.
///
/// We intentionally do NOT key off a specific sub-path like
/// `input_layernorm`: Nemotron-H's Mamba2 layers ship `norm.weight`
/// instead, and an earlier anchor-specific check broke preflight for
/// every Nemotron checkpoint (Discord 2026-04-19). Any tensor under
/// `.layers.N.` counts as evidence that layer N exists.
fn check_layer_count(store: &WeightStore, config: &ModelConfig) -> Result<usize> {
    let mut observed: Vec<usize> = Vec::new();
    for name in store.names() {
        // Accept `*.layers.N.*` (HF-style with any prefix) AND
        // `layers.N.*` (Mistral consolidated checkpoints have no
        // leading `model.` / `backbone.` prefix).
        let tail = if let Some(pos) = name.find(".layers.") {
            &name[pos + ".layers.".len()..]
        } else if let Some(rest) = name.strip_prefix("layers.") {
            rest
        } else {
            continue;
        };
        let Some(end) = tail.find('.') else { continue };
        let Ok(idx) = tail[..end].parse::<usize>() else {
            continue;
        };
        observed.push(idx);
    }
    if observed.is_empty() {
        bail!(
            "Pre-flight: no `*.layers.N.*` tensors found. \
             Checkpoint is empty or uses a naming convention Atlas \
             doesn't recognize (expected {} layers).",
            config.num_hidden_layers,
        );
    }
    observed.sort_unstable();
    observed.dedup();
    let max_idx = *observed.last().unwrap();
    let expected = config.num_hidden_layers;
    if max_idx + 1 < expected {
        bail!(
            "Pre-flight: checkpoint has layers 0..{} but config.num_hidden_layers = {}. \
             Wrong variant, or the index pass dropped tensors.",
            max_idx + 1,
            expected,
        );
    }
    if max_idx + 1 > expected {
        let extras = (max_idx + 1) - expected;
        tracing::info!(
            "Pre-flight: checkpoint has {extras} layer(s) beyond num_hidden_layers={expected} \
             (max index {max_idx}). Treating extras as MTP/draft modules."
        );
    }
    Ok(max_idx)
}

/// Detect the highest expert index across every layer and compare with
/// `config.num_experts`. Catches the "community re-quant shipped a
/// different base model" class of error cheaply.
fn check_expert_count(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    if config.num_experts == 0 {
        return Ok(());
    }
    let mut max_expert: Option<usize> = None;
    for name in store.names() {
        // Patterns seen across supported MoE models:
        //   <prefix>.layers.{L}.block_sparse_moe.experts.{E}.w?.weight
        //   <prefix>.layers.{L}.mlp.experts.{E}.{up|down|gate}_proj.weight
        //   <prefix>.layers.{L}.feed_forward.experts.{E}.weight
        let Some(idx) = extract_expert_idx(name) else {
            continue;
        };
        max_expert = Some(max_expert.map_or(idx, |m| m.max(idx)));
    }
    let Some(max_idx) = max_expert else {
        // Config says we're MoE but no expert tensors exist at all —
        // EP=2 may have sharded them all onto another rank, which is
        // legitimate. Warn rather than fail.
        tracing::warn!(
            "Pre-flight: config.num_experts={} but no expert tensors found locally. \
             Normal under EP>1 when the local rank owns zero experts; otherwise check \
             your checkpoint.",
            config.num_experts,
        );
        return Ok(());
    };
    let expected = config.num_experts;
    if max_idx + 1 > expected {
        bail!(
            "Pre-flight: checkpoint has experts 0..{} but config.num_experts = {}. \
             Likely a different base-model variant (e.g. {}-expert re-quant shipped with \
             a {}-expert config).",
            max_idx + 1,
            expected,
            max_idx + 1,
            expected,
        );
    }
    Ok(())
}

/// Parse the expert index `E` out of a tensor key, handling the three
/// common naming conventions (`block_sparse_moe.experts.{E}`,
/// `mlp.experts.{E}`, `feed_forward.experts.{E}`).
fn extract_expert_idx(name: &str) -> Option<usize> {
    for marker in [
        ".block_sparse_moe.experts.",
        ".mlp.experts.",
        ".feed_forward.experts.",
        ".experts.",
    ] {
        if let Some(tail) = name.split(marker).nth(1)
            && let Some(end) = tail.find('.')
            && let Ok(idx) = tail[..end].parse::<usize>()
        {
            return Some(idx);
        }
    }
    None
}

/// Check whether the loader for the declared `model_type` can actually
/// consume the extra layers the checkpoint ships. Today only MiniMax
/// ships per-module MTP layers that Atlas's loader doesn't handle yet
/// (see `weight_loader/minimax.rs:load_mtp_weights_multi`). Every other
/// family either embeds MTP differently (Qwen3.5 / Qwen3-Next ship
/// a dedicated `mtp.safetensors` shard with its own prefix — not extra
/// transformer layers) or doesn't ship it at all.
///
/// This function stays discovery-based: when new families grow MTP
/// support, add an entry in `MTP_SUPPORTED_MODEL_TYPES` below and it
/// works without further preflight surgery.
fn check_mtp_consumability(config: &ModelConfig) -> Result<()> {
    const MTP_SUPPORTED_MODEL_TYPES: &[&str] = &[
        "qwen3_next",
        "qwen3_5_moe",
        "qwen3_6_moe",
        "qwen3_vl_moe",
        "qwen3_coder_next",
    ];
    if MTP_SUPPORTED_MODEL_TYPES.contains(&config.model_type.as_str()) {
        return Ok(());
    }
    bail!(
        "Pre-flight: `--speculative` requested, but the checkpoint for model_type='{}' \
         ships MTP module layers that Atlas's loader doesn't consume yet. \
         Either retry without `--speculative`, or pick a checkpoint variant that \
         omits the MTP layers. Supported MTP model_types: {:?}.",
        config.model_type,
        MTP_SUPPORTED_MODEL_TYPES,
    );
}

/// Generic MoE correction_bias shape check (DeepSeek V3 / MiniMax M2 /
/// any future family that uses the loss-free-balancing bias). The
/// bias tensor name is `<moe-prefix>.e_score_correction_bias` and its
/// shape must match `config.num_experts`. Discovery-based: we scan
/// `store.names()` for ANY tensor ending in that suffix and validate
/// each one, so we don't need to know the MoE prefix
/// (`block_sparse_moe` on MiniMax, `mlp` on DeepSeek V3, etc.).
fn check_correction_bias_shape(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    if config.num_experts == 0 {
        return Ok(());
    }
    for name in store.names() {
        if !name.ends_with(".e_score_correction_bias") && !name.ends_with(".correction_bias") {
            continue;
        }
        let t = store.get(name)?;
        let n_elems = t.num_elements();
        if n_elems != config.num_experts {
            bail!(
                "Pre-flight: '{name}' has {n_elems} elements but config.num_experts = {}. \
                 The checkpoint is for a different expert count; EP sharding math would \
                 be wrong and the MoE router would route to non-existent experts.",
                config.num_experts,
            );
        }
    }
    Ok(())
}
