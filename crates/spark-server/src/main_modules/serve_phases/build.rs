// SPDX-License-Identifier: AGPL-3.0-only

//! Model factory call, prefix-cache + high-speed-swap setup, and the
//! rank > 0 EP worker entry point.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn build_prefix_cache(
    args: &cli::ServeArgs,
) -> Box<dyn spark_runtime::prefix_cache::PrefixCache> {
    if args.enable_prefix_caching {
        if args.high_speed_swap {
            tracing::info!(
                "Prefix caching: ENABLED (radix tree, with --high-speed-swap disk-side refcounts)"
            );
        } else {
            tracing::info!("Prefix caching: ENABLED (radix tree)");
        }
        Box::new(spark_runtime::radix_tree::RadixTree::new())
    } else {
        tracing::info!("Prefix caching: disabled");
        Box::new(spark_runtime::prefix_cache::NoPrefixCaching)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_model(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    store: &spark_runtime::weights::WeightStore,
    gpu: Box<dyn spark_runtime::gpu::GpuBackend>,
    max_batch_tokens: usize,
    kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    inference_reserve: usize,
    layer_dtypes: Vec<spark_runtime::kv_cache::KvCacheDtype>,
    hss_cache_blocks_per_seq: Option<u32>,
    prefix_cache: Box<dyn spark_runtime::prefix_cache::PrefixCache>,
    comm: Option<std::sync::Arc<dyn spark_comm::CommBackend>>,
    dflash_args: Option<spark_model::factory::DflashBuildArgs<'_>>,
) -> Result<Box<dyn spark_model::traits::Model>> {
    let mtp_quant: spark_model::layers::MtpQuantization = args
        .mtp_quantization
        .parse()
        .context("Invalid --mtp-quantization value")?;
    spark_model::factory::build_model(
        config.clone(),
        store,
        gpu,
        max_batch_tokens,
        args.block_size,
        args.max_seq_len,
        args.max_batch_size,
        mtp_quant,
        args.speculative || args.dflash,
        prefix_cache,
        args.mtp_vocab,
        comm,
        args.self_speculative || args.ngram_speculative,
        if args.dflash {
            args.dflash_gamma.saturating_sub(1).max(1)
        } else {
            args.num_drafts
        },
        kv_dtype,
        inference_reserve,
        args.gpu_memory_utilization,
        args.ssm_cache_slots,
        layer_dtypes,
        args.ssm_checkpoint_interval,
        hss_cache_blocks_per_seq,
        dflash_args,
    )
    .context("Failed to build model")
}

pub(crate) fn build_high_speed_swap_config(
    args: &cli::ServeArgs,
) -> Result<Option<spark_storage::HighSpeedSwapConfig>> {
    if !args.high_speed_swap {
        return Ok(None);
    }
    let dir = args
        .high_speed_swap_dir
        .clone()
        .unwrap_or_else(|| std::path::PathBuf::from("/var/tmp/atlas-hsw"));
    let bytes_gb = args.high_speed_swap_gb.unwrap_or(64);
    let resident_blocks = args.high_speed_swap_resident_blocks.unwrap_or(8192);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        anyhow::bail!(
            "--high-speed-swap: failed to create dir {}: {e}",
            dir.display()
        );
    }
    let cfg = spark_storage::HighSpeedSwapConfig {
        dir,
        bytes: bytes_gb * (1 << 30),
        resident_blocks,
        rank: args.high_speed_swap_rank,
        qd: args.high_speed_swap_qd,
        graph: args.high_speed_swap_graph.unwrap_or(true),
        projection_seed: 0xCAFE_F00D,
    };
    cfg.validate()?;
    Ok(Some(cfg))
}

pub(crate) fn validate_head_high_speed_swap(
    args: &cli::ServeArgs,
    early_high_speed_swap_cfg: &Option<spark_storage::HighSpeedSwapConfig>,
    swap_space_gb: usize,
) -> Result<Option<spark_storage::HighSpeedSwapConfig>> {
    let Some(cfg) = early_high_speed_swap_cfg.as_ref() else {
        return Ok(None);
    };
    if swap_space_gb > 0
        && cfg.dir.canonicalize().ok().as_deref()
            == std::path::Path::new("/tmp/atlas-swap")
                .canonicalize()
                .ok()
                .as_deref()
    {
        let _ = args;
        anyhow::bail!(
            "--high-speed-swap-dir must not be /tmp/atlas-swap (already used \
             by --swap-space-gb sequence-level fallback)"
        );
    }
    tracing::info!(
        "--high-speed-swap enabled: dir={}, budget={} GiB, scratch={} blocks, \
         rank={}, qd={}, graph={}",
        cfg.dir.display(),
        cfg.bytes / (1 << 30),
        cfg.resident_blocks,
        cfg.rank,
        cfg.qd,
        cfg.graph,
    );
    Ok(Some(cfg.clone()))
}

pub(crate) fn maybe_run_ep_worker(
    args: &cli::ServeArgs,
    model: &mut Option<Box<dyn spark_model::traits::Model>>,
    early_high_speed_swap_cfg: &Option<spark_storage::HighSpeedSwapConfig>,
) -> Result<bool> {
    if args.rank == 0 {
        return Ok(false);
    }
    let rank = args.rank;
    let model_owned = model.take().expect("EP worker requires owned model");
    let model_has_proposer = model_owned.has_proposer();
    if !args.speculative && !args.self_speculative && !args.ngram_speculative && model_has_proposer
    {
        let override_set = matches!(
            std::env::var("ATLAS_ALLOW_SPEC_MISMATCH").as_deref(),
            Ok("1") | Ok("true")
        );
        if !override_set {
            anyhow::bail!(
                "EP worker (rank {rank}) started WITHOUT any --speculative flag, \
                 but this checkpoint has MTP weights and the head will likely use them. \
                 Mirror the head's --speculative / --mtp-quantization / --num-drafts \
                 flags here, or set ATLAS_ALLOW_SPEC_MISMATCH=1 if the head is also \
                 non-speculative."
            );
        }
        tracing::warn!(
            "EP worker (rank {rank}) running WITHOUT speculative flags but \
             ATLAS_ALLOW_SPEC_MISMATCH=1 — head must NOT issue MTP commands."
        );
    } else if !model_has_proposer
        && !args.speculative
        && !args.self_speculative
        && !args.ngram_speculative
    {
        tracing::info!(
            "EP worker (rank {rank}): checkpoint has no MTP weights; \
             spec-mismatch guard auto-skipped (head can't use MTP either)."
        );
    }
    let worker_hss_cfg = early_high_speed_swap_cfg.clone();
    // Copy primitives out of `args` so the worker thread (which is
    // `'static`) doesn't capture the function-scoped `&ServeArgs` ref.
    let max_batch_size = args.max_batch_size;
    let handle = std::thread::spawn(move || {
        model_owned
            .bind_gpu_to_thread()
            .expect("Failed to bind GPU to EP worker thread");
        if let Some(cfg) = worker_hss_cfg {
            match model_owned.high_speed_swap_dims() {
                Some(dims) => {
                    if let Err(e) = spark_storage::install_local(rank as u64, cfg, dims) {
                        tracing::error!(
                            "EP worker (rank {rank}): --high-speed-swap install failed: {e:#}"
                        );
                    } else {
                        tracing::info!(
                            "EP worker (rank {rank}): --high-speed-swap orchestrator installed"
                        );
                    }
                }
                None => {
                    tracing::warn!(
                        "EP worker (rank {rank}): --high-speed-swap requested but model \
                         does not expose high_speed_swap_dims; skipping install"
                    );
                }
            }
        }
        // Slots vec sized to match the head's scheduler `max_batch_size`.
        // Pre-allocate every slot. The head only emits `0xFFFFFFF1`
        // (free+realloc) on lifecycle events — sequence finish/error —
        // not on first use, so a fresh `prefill_a_step` for slot N
        // arrives as `0xFFFFFFF0` with no prior alloc broadcast. Under v1
        // (max_batch_size=1) this is just slot 0, matching the legacy
        // behavior. Under v2 (max_batch_size>1) every slot must be
        // populated up front for the same reason.
        //
        // Both ranks' SSM pools start with the same free-list ordering
        // (see ssm_pool.rs: `(0..max_slots).rev().collect()` + `pop()`),
        // so pre-allocating in `0..max_batch_size` order on the worker
        // means `slots[i].slot_idx == i` — matching the slot ids the
        // head's `alloc_sequence` returns for its Nth claim.
        let mut slots: Vec<Option<spark_model::traits::SequenceState>> =
            (0..max_batch_size).map(|_| None).collect();
        for slot in slots.iter_mut() {
            *slot = Some(
                model_owned
                    .alloc_sequence()
                    .expect("Failed to allocate EP worker sequence"),
            );
        }
        tracing::info!(
            "EP worker ready (rank {rank}, {} slots), waiting for commands",
            slots.len()
        );
        loop {
            match model_owned.ep_worker_step(&mut slots) {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => {
                    tracing::error!("EP worker error: {e:#}");
                    break;
                }
            }
        }
        for slot in slots.iter_mut() {
            if let Some(seq) = slot.as_mut() {
                let _ = model_owned.free_sequence(seq);
            }
        }
        tracing::info!("EP worker stopped (rank {rank})");
    });
    handle.join().expect("EP worker thread panicked");
    Ok(true)
}
