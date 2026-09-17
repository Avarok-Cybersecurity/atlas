// SPDX-License-Identifier: AGPL-3.0-only

//! Loading the DFlash drafter, split out of `weights.rs` to keep it under
//! the 500-line cap.

use super::*;

pub(crate) fn load_dflash_drafter(
    args: &cli::ServeArgs,
    ptx_set: &avarok_kernels::TargetPtxSet,
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
    //   * AVAROK_DFLASH_DRAFTER_FP8: FP8 mirrors of the dense weights
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
    let fp8_mirrors = if std::env::var("AVAROK_DFLASH_DRAFTER_FP8").ok().as_deref() != Some("0") {
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
            "DFlash drafter would over-commit unified memory: estimated footprint {:.2} GB              (weights {:.2} + drafter-KV {:.2} + fused_kv {:.2} + selector-host {:.2} +              fp8-mirrors {:.2} + scratch {:.2}) but only {:.2} GB free with {:.2} GB              headroom pledged by --gpu-memory-utilization {:.2}. On GB10 this would SWAP              the host, not error. Lower --max-seq-len, lower --gpu-memory-utilization              pressure elsewhere, or drop AVAROK_DFLASH_DRAFTER_FP8.",
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
