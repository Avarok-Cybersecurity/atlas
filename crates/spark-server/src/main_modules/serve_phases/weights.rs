// SPDX-License-Identifier: AGPL-3.0-only

//! Weight-store loading: main checkpoint, prefix auto-detect, DFlash drafter.

use std::path::Path;

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn quant_multiplier(config: &ModelConfig) -> Option<f64> {
    if config.model_type == "minimax_m2" || config.model_type == "step3p7" {
        Some(1.02)
    } else if config
        .quantization_config
        .as_ref()
        .is_some_and(|qc| qc.quant_method == "fp8")
    {
        Some(1.05)
    } else {
        None
    }
}

pub(crate) fn load_weight_store(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    ep_rank: usize,
    ep_size: usize,
    oom_reserve_bytes: usize,
) -> Result<spark_runtime::weights::WeightStore> {
    use spark_runtime::weights::WeightLoader;
    let mult = quant_multiplier(config);

    // GGUF checkpoints are dequantized to BF16 by a dedicated loader; take that
    // path whenever a .gguf file is present (fast/safetensors loaders can't read it).
    if spark_runtime::weights::find_gguf(model_dir).is_some() {
        tracing::info!("Detected GGUF weights; using GgufLoader (GPU dequant → BF16)");
        let mut loader = if ep_size > 1 {
            spark_runtime::weights::GgufLoader::with_ep(ep_rank, ep_size, config.num_experts)
        } else {
            spark_runtime::weights::GgufLoader::new()
        };
        loader.peak_memory_multiplier = mult;
        let store = loader
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load model weights (GGUF loader)")?;
        tracing::info!("Loaded {} weight tensors (GGUF)", store.len());
        return Ok(store);
    }

    let use_fast_load =
        !args.no_fast_load && std::env::var("ATLAS_FAST_LOAD").ok().as_deref() != Some("0");
    let store = if use_fast_load {
        #[cfg(unix)]
        {
            tracing::info!("Using fast weight loader (O_DIRECT + pipelined read/copy)");
            let mut loader = if ep_size > 1 {
                spark_runtime::fast_weights::FastSafetensorsLoader::with_ep(
                    ep_rank,
                    ep_size,
                    config.num_experts,
                )
            } else {
                spark_runtime::fast_weights::FastSafetensorsLoader::new()
            };
            loader.peak_memory_multiplier = mult;
            loader.skip_activation_scales = skip_activation_scales(config);
            loader.skip_mtp = skip_mtp(config, want_mtp(args));
            loader.prefetch_shards = args.fast_load_prefetch_shards
                || std::env::var("ATLAS_FAST_LOAD_PREFETCH_SHARDS")
                    .ok()
                    .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
            if loader.prefetch_shards {
                tracing::info!("Fast weight loader shard prefetch/readahead enabled");
            }
            loader
                .load(model_dir, gpu, oom_reserve_bytes)
                .context("Failed to load model weights (fast loader)")?
        }
        #[cfg(not(unix))]
        {
            anyhow::bail!("--fast-load requires a Unix host (needs O_DIRECT / posix_fadvise)");
        }
    } else {
        let mut loader = if ep_size > 1 {
            spark_runtime::weights::SafetensorsLoader::with_ep(ep_rank, ep_size, config.num_experts)
        } else {
            spark_runtime::weights::SafetensorsLoader::new()
        };
        loader.peak_memory_multiplier = mult;
        loader.skip_activation_scales = skip_activation_scales(config);
        loader.skip_mtp = skip_mtp(config, want_mtp(args));
        loader
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load model weights")?
    };
    tracing::info!("Loaded {} weight tensors", store.len());
    Ok(store)
}

pub(crate) fn load_dflash_drafter(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<
    Option<(
        spark_runtime::weights::WeightStore,
        spark_model::weight_loader::DflashConfig,
    )>,
> {
    use spark_runtime::weights::WeightLoader;
    if !args.dflash {
        return Ok(None);
    }
    let drafter_id = args
        .draft_model
        .clone()
        .or_else(|| ptx_set.dflash.as_ref().map(|d| d.draft_model.to_string()))
        .context(
            "--dflash set but no drafter HF id provided: pass --draft-model <ID> \
             or use a target whose MODEL.toml has a [dflash] section",
        )?;
    tracing::info!("DFlash: resolving drafter '{drafter_id}'");
    let drafter_dir =
        crate::model_resolver::resolve_model_dir(&drafter_id, args.cache_dir.as_deref())
            .context("Failed to resolve DFlash drafter checkpoint")?;
    let drafter_config_json = std::fs::read_to_string(drafter_dir.join("config.json"))
        .with_context(|| {
            format!(
                "Failed to read drafter config.json at {}",
                drafter_dir.display()
            )
        })?;
    let drafter_config =
        spark_model::weight_loader::dflash_loader::parse_dflash_config(&drafter_config_json)?;
    let mut loader = spark_runtime::weights::SafetensorsLoader::new();
    loader.peak_memory_multiplier = None;
    let drafter_store = loader
        .load(&drafter_dir, gpu, 0)
        .context("Failed to load DFlash drafter weights")?;
    tracing::info!(
        "DFlash drafter store: {} tensors, {} bytes",
        drafter_store.len(),
        drafter_store.total_bytes()
    );
    Ok(Some((drafter_store, drafter_config)))
}

/// Startup-loaded LoRA adapter: its own WeightStore + parsed PEFT config.
/// One `LoraAdapterState` per repeated `--lora-adapter NAME=PATH`; each becomes
/// one resident pool slot. A single adapter is byte-identical to the v0 path.
pub(crate) struct LoraAdapterState {
    pub name: String,
    pub peft_config: atlas_core::config::PeftAdapterConfig,
    pub store: spark_runtime::weights::WeightStore,
}

/// Resolve + load every `--lora-adapter` into its own on-device `WeightStore`
/// (slot 0..N-1). Empty when no adapter is requested. Rejects >`--max-loras`
/// adapters and duplicate names up front.
pub(crate) fn load_lora_adapters(
    args: &cli::ServeArgs,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<Vec<LoraAdapterState>> {
    if args.lora_adapter.is_empty() {
        return Ok(Vec::new());
    }
    if args.lora_adapter.len() > args.max_loras {
        anyhow::bail!(
            "--lora-adapter given {} times but --max-loras={} (pool has {} slots); \
             raise --max-loras or stage the extras on an $ATLAS_LORA_PEER",
            args.lora_adapter.len(),
            args.max_loras,
            args.max_loras,
        );
    }
    let mut states: Vec<LoraAdapterState> = Vec::with_capacity(args.lora_adapter.len());
    for (name, spec) in &args.lora_adapter {
        if states.iter().any(|s| &s.name == name) {
            anyhow::bail!("--lora-adapter name '{name}' given twice (names must be unique)");
        }
        tracing::info!("LoRA: resolving adapter '{name}' from '{spec}'");
        let adapter_dir =
            crate::model_resolver::resolve_adapter_dir(spec, args.cache_dir.as_deref())
                .context("Failed to resolve LoRA adapter")?;
        let cfg_path = adapter_dir.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("Failed to read {}", cfg_path.display()))?;
        // Hard-error parser (atlas-core config/parsers/lora.rs) — scaling is read
        // per adapter (alpha/r, alpha/sqrt(r) under use_rslora), NEVER defaulted.
        let peft_config = atlas_core::config::parse_peft_adapter_config(&raw)
            .with_context(|| format!("Failed to parse {}", cfg_path.display()))?;
        if peft_config.r > args.max_lora_rank {
            anyhow::bail!(
                "LoRA adapter '{}' has r={} > --max-lora-rank {} — raise the flag \
                 (slot pool is rank-padded to it) or use a smaller adapter",
                name,
                peft_config.r,
                args.max_lora_rank,
            );
        }
        let store = spark_runtime::weights::adapter::load_adapter_safetensors(&adapter_dir, gpu, 0)
            .context("Failed to load LoRA adapter weights")?;
        tracing::info!(
            "LoRA adapter '{}': {} tensors, {} bytes loaded; r={}, alpha={}, \
             use_rslora={}, scaling={:.6}, target_modules={:?}",
            name,
            store.len(),
            store.total_bytes(),
            peft_config.r,
            peft_config.lora_alpha,
            peft_config.use_rslora,
            peft_config.scaling(),
            peft_config.target_modules,
        );
        states.push(LoraAdapterState {
            name: name.clone(),
            peft_config,
            store,
        });
    }
    Ok(states)
}

/// Whether this model's loader can skip the W4A4 `*.input_scale` activation
/// scales.
///
/// ModelOpt NVFP4 ships one 0-dim F32 scalar per quantized projection. On
/// Qwen3.8-Flash-Next that is ~74k four-byte allocations (48 layers x 512
/// experts x 3 projections), each taking a full allocation granule — GBs of
/// padding for values the w4a16 path never reads. The NVFP4 loader already
/// treats the key as optional (`if store.contains(..) else NULL`), so not
/// uploading them is identical to loading a checkpoint that never had them.
///
/// Deliberately an ALLOW-LIST, not a blanket skip: `step3p7` reads
/// `input_scale` on its own loader path, and silently withholding a tensor a
/// loader DOES read is exactly the class of bug that stays invisible until
/// the output is subtly wrong.
fn skip_activation_scales(config: &ModelConfig) -> bool {
    matches!(config.model_type.as_str(), "qwen4_exp")
}

/// Whether `mtp.*` can be left on disk for this model.
///
/// The reason is no longer that the loader builds no MTP head — it does now
/// (`weight_loader::qwen4_exp::load_qwen4_exp_mtp_module`). It is MEMORY: the
/// qwen4_exp MTP block is a second model, ~1.5 GB resident on the per-expert
/// NVFP4 snapshot and a ~5 GB BF16 transient on the fused one, on a model that
/// already sits at ~94.6 GB with ~2.7 GB of headroom. That is memory the KV
/// cache needs unless MTP is actually armed.
///
/// So it is skipped by DEFAULT and uploaded only when asked: `--speculative`,
/// or `ATLAS_QWEN4EXP_MTP=1` to load and audit the block without arming
/// speculation. No other `model_type` ever returns true here, in either arm.
fn skip_mtp(config: &ModelConfig, want_mtp: bool) -> bool {
    matches!(config.model_type.as_str(), "qwen4_exp") && !want_mtp
}

/// True when the operator asked for the `mtp.*` tensors to be uploaded.
fn want_mtp(args: &cli::ServeArgs) -> bool {
    args.speculative || std::env::var("ATLAS_QWEN4EXP_MTP").as_deref() == Ok("1")
}

#[cfg(test)]
mod skip_mtp_tests {
    use super::*;

    fn cfg(model_type: &str) -> ModelConfig {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.model_type = model_type.to_string();
        c
    }

    /// Default serving must stay byte-identical: `mtp.*` still never uploaded.
    #[test]
    fn qwen4_exp_skips_by_default_and_uploads_when_asked() {
        assert!(skip_mtp(&cfg("qwen4_exp"), false));
        assert!(!skip_mtp(&cfg("qwen4_exp"), true));
    }

    /// The new parameter cannot regress another family: `skip_mtp` is false
    /// for every other `model_type` in BOTH arms, so nothing else changes
    /// behaviour when MTP is armed.
    #[test]
    fn other_model_types_are_unaffected_in_both_arms() {
        for mt in ["qwen3_next", "qwen3_5_moe", "deepseek_v4", "holo3_1_moe"] {
            assert!(!skip_mtp(&cfg(mt), false), "{mt}");
            assert!(!skip_mtp(&cfg(mt), true), "{mt}");
        }
    }
}
