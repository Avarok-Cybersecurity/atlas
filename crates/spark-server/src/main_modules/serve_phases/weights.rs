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
        loader
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load model weights")?
    };
    tracing::info!("Loaded {} weight tensors", store.len());
    Ok(store)
}

pub(crate) fn auto_detect_weight_prefix(
    store: &spark_runtime::weights::WeightStore,
    config: &mut ModelConfig,
) {
    if config.weight_prefix.is_empty() && config.nested_config {
        config.weight_prefix = if store.contains("language_model.model.embed_tokens.weight") {
            "language_model.model".to_string()
        } else if store.contains("model.language_model.embed_tokens.weight") {
            "model.language_model".to_string()
        } else {
            let scanned = store
                .names()
                .find(|k| k.contains(".layers.0."))
                .and_then(|k| k.split(".layers.0.").next())
                .map(|s| s.to_string());
            if let Some(ref prefix) = scanned {
                tracing::info!("Auto-detected weight prefix: '{prefix}'");
            }
            scanned.unwrap_or_else(|| "model".to_string())
        };
    }
    if !config.weight_prefix.is_empty() {
        tracing::info!("Weight prefix: {}", config.weight_prefix);
    }
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
