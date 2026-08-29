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
            loader.skip_mtp = skip_mtp(config);
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
        loader.skip_mtp = skip_mtp(config);
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
    // ── DFlash footprint pre-flight (2026-08-19) ────────────────────────
    // Every byte below lands OUTSIDE the KV planner's view until it is
    // already allocated, and on GB10's unified LPDDR5X an over-commit is
    // not an OOM error — it is the HOST swapping (measured: 1.8 GB/s to
    // disk before the OOM-killer fired, with the peak-memory guard on this
    // very load explicitly disabled). Estimate the drafter's WHOLE
    // footprint from the checkpoint metadata BEFORE allocating anything,
    // and refuse while memory is still sane:
    //   * drafter weights: safetensors bytes on disk (BF16 ~= device bytes)
    //   * head fixed costs: scratch (~250 MB), fused_kv, drafter KV cache
    //     (max_seq_len x layers x 2 x kv_dim x BF16), DFlash2 selector host
    //     copies (~2 x vocab x rank BF16 — unified memory, so host counts)
    //   * ATLAS_DFLASH_DRAFTER_FP8: FP8 mirrors of the dense weights
    //     (~0.5x store) + the lm_head mirror (vocab x hidden FP8) + an
    //     equal transient for the quantize staging
    let store_bytes: u64 = std::fs::read_dir(&drafter_dir)
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
        // std::fs::metadata FOLLOWS symlinks; DirEntry::metadata does not,
        // and HF snapshot dirs are all symlinks into blobs/ — the first
        // live run of this gate reported 'weights 0.00 GB' for a 3.85 GB
        // drafter because it measured the link, not the blob.
        .filter_map(|e| std::fs::metadata(e.path()).ok().map(|m| m.len()))
        .sum();
    let c = &drafter_config;
    let kv_dim = c.num_key_value_heads * c.head_dim;
    let drafter_kv =
        (args.max_seq_len as u64) * (c.num_hidden_layers as u64) * 2 * (kv_dim as u64) * 2;
    let fused_kv = (c.num_hidden_layers as u64) * 2 * (kv_dim as u64) * (c.hidden_size as u64) * 2;
    let selector_host = c
        .dflash_config
        .as_ref()
        .map(|d| d.selector_rank)
        .filter(|r| *r > 0)
        .map(|r| {
            2 * (c.vocab_size as u64) * (r as u64) * 2 + (r as u64) * (c.hidden_size as u64) * 2
        })
        .unwrap_or(0);
    // Same predicate as the gate that actually allocates them
    // (from_weights: `!= Some("0")`), NOT `.is_some()`. FP8 drafter
    // weights are DEFAULT-ON, so testing "is the variable set" counted
    // the mirrors as zero on exactly the default path — the pre-flight
    // printed `fp8-mirrors 0.00` while the mirrors were resident.
    let fp8_mirrors = if std::env::var("ATLAS_DFLASH_DRAFTER_FP8").ok().as_deref() != Some("0") {
        let lm_head = (c.vocab_size as u64) * (c.hidden_size as u64);
        store_bytes / 2 + 2 * lm_head
    } else {
        0
    };
    let scratch_est: u64 = 300 << 20;
    let estimate = store_bytes + drafter_kv + fused_kv + selector_host + fp8_mirrors + scratch_est;

    let free = gpu.free_memory().unwrap_or(0) as u64;
    let total = gpu.total_memory().unwrap_or(0) as u64;
    // The non-budget headroom (total x (1 - util)) must SURVIVE the drafter:
    // it is the co-tenant/system slack the util flag promises to leave.
    let headroom = (total as f64 * (1.0 - args.gpu_memory_utilization)) as u64;
    if free < estimate + headroom {
        anyhow::bail!(
            "DFlash drafter would over-commit unified memory: estimated footprint {:.2} GB              (weights {:.2} + drafter-KV {:.2} + fused_kv {:.2} + selector-host {:.2} +              fp8-mirrors {:.2} + scratch {:.2}) but only {:.2} GB free with {:.2} GB              headroom pledged by --gpu-memory-utilization {:.2}. On GB10 this would SWAP              the host, not error. Lower --max-seq-len, lower --gpu-memory-utilization              pressure elsewhere, or drop ATLAS_DFLASH_DRAFTER_FP8.",
            estimate as f64 / 1e9,
            store_bytes as f64 / 1e9,
            drafter_kv as f64 / 1e9,
            fused_kv as f64 / 1e9,
            selector_host as f64 / 1e9,
            fp8_mirrors as f64 / 1e9,
            scratch_est as f64 / 1e9,
            free as f64 / 1e9,
            headroom as f64 / 1e9,
            args.gpu_memory_utilization,
        );
    }
    tracing::info!(
        "DFlash footprint pre-flight: estimate {:.2} GB (weights {:.2}, drafter-KV {:.2},          fp8-mirrors {:.2}, selector-host {:.2}) vs {:.2} GB free, {:.2} GB headroom — OK",
        estimate as f64 / 1e9,
        store_bytes as f64 / 1e9,
        drafter_kv as f64 / 1e9,
        fp8_mirrors as f64 / 1e9,
        selector_host as f64 / 1e9,
        free as f64 / 1e9,
        headroom as f64 / 1e9,
    );

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

/// Best-effort: does the TARGET checkpoint ship `lm_head.weight` natively as
/// FP8 E4M3? Reads only the safetensors JSON header of the shard that holds
/// the tensor (8-byte length prefix + header), never the weights. Any failure
/// returns `false`, which keeps the pre-flight estimate conservative.
fn target_ships_native_fp8_lm_head(args: &cli::ServeArgs) -> bool {
    fn inner(args: &cli::ServeArgs) -> Option<bool> {
        let dir = if let Some(p) = &args.model_from_path {
            p.clone()
        } else {
            crate::model_resolver::resolve_model_dir(
                args.model.as_deref()?,
                args.cache_dir.as_deref(),
            )
            .ok()?
        };
        const KEYS: [&str; 3] = [
            "lm_head.weight",
            "language_model.lm_head.weight",
            "model.lm_head.weight",
        ];
        // Multi-shard: index.json names the shard; single-file fallback.
        let shard = if let Ok(idx) = std::fs::read(dir.join("model.safetensors.index.json")) {
            let idx: serde_json::Value = serde_json::from_slice(&idx).ok()?;
            let map = idx.get("weight_map")?;
            KEYS.iter()
                .find_map(|k| map.get(*k).and_then(|v| v.as_str()))
                .map(|s| dir.join(s))?
        } else {
            dir.join("model.safetensors")
        };
        use std::io::Read as _;
        let mut f = std::fs::File::open(shard).ok()?;
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8).ok()?;
        let hlen = u64::from_le_bytes(len8);
        if hlen > 64 << 20 {
            return None; // implausible header — refuse to slurp
        }
        let mut hdr = vec![0u8; hlen as usize];
        f.read_exact(&mut hdr).ok()?;
        let hdr: serde_json::Value = serde_json::from_slice(&hdr).ok()?;
        let dtype = KEYS
            .iter()
            .find_map(|k| hdr.get(*k))
            .and_then(|t| t.get("dtype"))
            .and_then(|d| d.as_str())?;
        Some(dtype == "F8_E4M3")
    }
    inner(args).unwrap_or(false)
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
        let rank_ceiling = args.max_lora_rank.unwrap_or(64);
        if peft_config.r > rank_ceiling {
            anyhow::bail!(
                "LoRA adapter '{}' has r={} > --max-lora-rank {} — raise the flag \
                 (slot pool is rank-padded to it) or use a smaller adapter",
                name,
                peft_config.r,
                rank_ceiling,
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

/// Whether this model's loader builds no MTP head, so `mtp.*` need not be
/// uploaded at all.
///
/// `Qwen4ExpWeightLoader::load_mtp_weights` returns `None` (#753 item I: the
/// MTP block is effectively a second model — its own 512-expert MoE, its own
/// hyper-connection mixer, its own indexer). Uploading ~1.5 GB of weights
/// that are then discarded is memory the KV cache needs.
fn skip_mtp(config: &ModelConfig) -> bool {
    matches!(config.model_type.as_str(), "qwen4_exp")
}
