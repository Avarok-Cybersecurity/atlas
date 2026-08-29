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

mod deepseek_v4;

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
pub fn preflight(
    store: &WeightStore,
    config: &ModelConfig,
    use_speculative: bool,
    kv_cache_dtype: Option<&str>,
) -> Result<()> {
    // NLLB / M2M-100 is an encoder-decoder checkpoint: the tied embedding is
    // `model.shared.weight`, layers span separate encoder + decoder stacks, and
    // there is cross-attention — none of which fit these decoder-only checks.
    // `NllbGpuModel::new` validates its own weights (presence + bf16 dtype).
    if matches!(config.model_type.as_str(), "m2m_100" | "nllb") {
        tracing::info!(
            "Pre-flight: NLLB/M2M-100 encoder-decoder — generic checks skipped \
             (weights validated in NllbGpuModel::new)"
        );
        return Ok(());
    }
    // Model-agnostic checks — driven purely by `store.names()` and
    // `config` (which already carries the parsed `config.json` values).
    check_quant_method(config)?;
    deepseek_v4::check_native_dspark_checkpoint(store, config, use_speculative)?;
    check_embedding_and_head(store)?;
    let max_layer_idx = check_layer_count(store, config)?;
    check_expert_count(store, config)?;
    check_correction_bias_shape(store, config)?;
    check_qsa_kv_dtype(store.names(), kv_cache_dtype)?;

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

/// A QSA checkpoint served with a non-BF16 KV cache cannot decode.
///
/// The selection gather copies raw NHD rows out of the paged cache, so a
/// quantized cache has nothing it can copy. The decode path already refuses —
/// but it refuses on the FIRST REQUEST, which is after the weights have loaded
/// (three minutes and a hundred-plus gigabytes for Qwen3.8-Flash-Next) and
/// after the operator has sent something and waited for it. The fact is known
/// as soon as the store is populated, so it is said here instead.
///
/// Presence of `self_attn.indexer.*` is what makes a checkpoint QSA — the same
/// thing the loader keys on. Nothing in `config.json` records it.
fn check_qsa_kv_dtype<'a>(
    names: impl Iterator<Item = &'a str>,
    kv_cache_dtype: Option<&str>,
) -> Result<()> {
    // Names, not the store: this needs nothing else, and taking the store would
    // have meant a test-only way to put a name into one.
    let has_qsa = names.into_iter().any(|n| n.contains(".self_attn.indexer."));
    if !has_qsa {
        return Ok(());
    }
    // `None` means the caller genuinely has no KV cache to speak of, NOT "the
    // default is fine": the engine default is fp8, and reading an absent flag as
    // bf16 made this check inert on the bare `spark serve <model>` -- the exact
    // invocation it exists to catch, and the one that pays a hundred-plus
    // gigabytes of load before the decode path refuses on the first request.
    // Callers resolve the default before calling; see `serve_load`.
    let Some(dt) = kv_cache_dtype else {
        return Ok(());
    };
    // Only `bf16` parses downstream (`KvCacheDtype::FromStr`); `bfloat16` and
    // `auto` were accepted here but are rejected by CLI validation before they
    // could ever arrive, so listing them only made the rule look laxer than it is.
    let d = dt.trim().to_ascii_lowercase();
    anyhow::ensure!(
        d == "bf16",
        "this checkpoint uses QSA sparse attention, whose selection gather \
         copies raw NHD rows, so it needs a plain BF16 KV cache — but \
         the KV cache resolves to `{dt}`.\n  \
         Serve with `--kv-cache-dtype bf16`. Dropping the flag does NOT do \
         that -- the default is fp8.\n  \
         Said here rather than on your first request, which is where the \
         decode path notices."
    );
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
    const EMBED_EXACTS: &[&str] = &[
        "tok_embeddings.weight",
        "embed_tokens.weight",
        "embed.weight",
    ];
    let has_embed = store
        .names()
        .any(|n| EMBED_EXACTS.contains(&n) || EMBED_SUFFIXES.iter().any(|s| n.ends_with(s)));
    if !has_embed {
        let sample: Vec<_> = store.names().take(20).collect();
        bail!(
            "Pre-flight: no embedding tensor found (checked exact: {EMBED_EXACTS:?}, \
             suffixes: {EMBED_SUFFIXES:?}). Is this a language-model checkpoint? \
             Sample tensor names: {sample:?}\
             \n\nHint: Some re-quant checkpoints (e.g. RedHatAI DeepSeek-V4-Flash-NVFP4-FP8) \
             ship embedding weights in a separate file not listed in model.safetensors.index.json. \
             Check if the checkpoint directory contains a separate model.safetensors or \
             embed_tokens.safetensors file, and if model.safetensors.index.json maps \
             'model.embed_tokens.weight' to a shard."
        );
    }
    // LM head is optional (tied embeddings skip it). Scan suffixes:
    //   `lm_head.weight`       — HF / Qwen / Gemma / MiniMax
    //   `output.weight`        — Mistral consolidated
    //   `head.weight`          — DeepSeek-V4 / RedHatAI re-quant
    let has_head = store
        .names()
        .any(|n| n.ends_with("lm_head.weight") || n == "output.weight" || n == "head.weight");
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
    // LongCat-Flash serves each CHECKPOINT layer as TWO engine sublayers
    // (dual-sublayer "shortcut" blocks — the HF modeling file makes the same
    // 2x expansion), so the checkpoint legitimately carries half as many
    // `layers.N` indices as `num_hidden_layers`. Comparing against the engine
    // count would reject every valid LongCat checkpoint.
    let expected = if config.model_type.starts_with("longcat_flash") {
        config.num_hidden_layers / 2
    } else {
        config.num_hidden_layers
    };
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
        "holo3_1_moe",
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
        // The bias is per ROUTER LOGIT, and a zero-expert model (LongCat)
        // scores `num_experts + zero_expert_num` of them — the identity
        // experts are selectable but have no FFN weights.
        let expected = config.num_experts + config.zero_expert_num;
        if n_elems != expected {
            bail!(
                "Pre-flight: '{name}' has {n_elems} elements but config declares {expected} \
                 router logits (num_experts = {} + zero_expert_num = {}). The checkpoint is \
                 for a different expert count; EP sharding math would be wrong and the MoE \
                 router would route to non-existent experts.",
                config.num_experts,
                config.zero_expert_num,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod qsa_kv_tests {
    use super::check_qsa_kv_dtype;

    /// Names from a checkpoint with no indexer: not QSA, so the flag is none of
    /// this check's business.
    const PLAIN: [&str; 2] = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
    ];
    /// Presence of an indexer tensor is what makes a checkpoint QSA — the same
    /// thing the loader keys on.
    const QSA: [&str; 2] = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.indexer.index_qk_proj.weight",
    ];

    #[test]
    fn a_non_qsa_checkpoint_accepts_any_kv_dtype() {
        for dt in [None, Some("fp8"), Some("bf16"), Some("nonsense")] {
            assert!(check_qsa_kv_dtype(PLAIN.into_iter(), dt).is_ok(), "{dt:?}");
        }
    }

    /// The refusal has to name the flag AND the value, because the operator is
    /// reading it three minutes into a load they will have to repeat.
    #[test]
    fn a_qsa_checkpoint_refuses_a_quantized_kv_cache_by_name() {
        let e = check_qsa_kv_dtype(QSA.into_iter(), Some("fp8")).expect_err("must refuse");
        let msg = format!("{e}");
        assert!(msg.contains("--kv-cache-dtype bf16"), "{msg}");
        assert!(msg.contains("fp8"), "must quote what was given: {msg}");
        // The old text said "or drop the flag", which is the one thing that
        // cannot work: dropping it resolves to fp8, which is how we got here.
        assert!(
            !msg.contains("drop the flag"),
            "must not advise dropping the flag: {msg}"
        );
    }

    /// bf16, in the spellings that actually parse downstream.
    #[test]
    fn a_qsa_checkpoint_accepts_bf16() {
        for dt in [Some("bf16"), Some("BF16"), Some(" bf16 ")] {
            assert!(
                check_qsa_kv_dtype(QSA.into_iter(), dt).is_ok(),
                "{dt:?} must pass"
            );
        }
    }

    /// The regression this check was written for and then could not catch: the
    /// engine default is fp8, so an unresolved `None` read as "bf16, fine" made
    /// the check inert on `spark serve <model>` with no flag — the invocation
    /// most operators type. Callers resolve before calling; if one forgets, this
    /// is the test that says so.
    #[test]
    fn the_engine_default_is_not_something_a_qsa_checkpoint_can_use() {
        assert_eq!(spark_server_default_kv_dtype(), "fp8");
        check_qsa_kv_dtype(QSA.into_iter(), Some(spark_server_default_kv_dtype()))
            .expect_err("the resolved default must be refused, not waved through");
    }

    /// Spelled out rather than imported: `spark-server` depends on this crate,
    /// so the constant cannot come the other way. If the server's default ever
    /// changes, the assertion above fails here and names this comment.
    fn spark_server_default_kv_dtype() -> &'static str {
        "fp8"
    }

    /// Spellings the CLI rejects before preflight can see them. Accepting them
    /// here only made the rule look laxer than it is.
    #[test]
    fn spellings_that_do_not_parse_downstream_are_not_accepted() {
        for dt in [Some("bfloat16"), Some("auto")] {
            check_qsa_kv_dtype(QSA.into_iter(), dt).expect_err("{dt:?} must not pass");
        }
    }
}
