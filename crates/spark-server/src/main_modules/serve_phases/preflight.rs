// SPDX-License-Identifier: AGPL-3.0-only

//! GPU init + pre-load reserve preflight + post-load OOM check.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

mod ssm_h_fp16;
use ssm_h_fp16::ssm_h_fp16_preconditions;

pub(crate) struct ReservePreflight {
    pub(crate) inference_reserve: usize,
    pub(crate) buffer_arena_bytes: usize,
    pub(crate) gdn_two_phase_bytes: usize,
    pub(crate) ssm_prefill_chunk: usize,
    pub(crate) max_batch_tokens_pre: usize,
}

pub(crate) fn preflight_reserve(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    free_mem: usize,
) -> Result<ReservePreflight> {
    let h_state_bytes = config.ssm_h_state_bytes();
    let conv_state_bytes = config.ssm_conv_state_bytes();
    // `args.dflash` belongs here: `TransformerModel::new` forces the verify
    // pools ON whenever DFlash capture layers exist (`has_mtp |= dflash`),
    // so a DFlash serve allocates the full K=γ+1 intermediate/checkpoint
    // pools. Omitting it reserved only the base per-seq blobs and left the
    // entire verify-pool family OUTSIDE the util pledge — 13.7 GB tracked vs
    // a 1.3 GB reserve on the 27B at bs=8/γ=8 (2026-08-22 boot ledger, the
    // measured bulk of the ~12 GB pledge overshoot).
    let spec_on_pool =
        args.speculative || args.self_speculative || args.ngram_speculative || args.dflash;
    ssm_h_fp16_preconditions(args, config)?;
    // SSM state pool = per-seq live state (max_batch blobs) + MTP verify
    // state (intermediates + checkpoint) for the slots spec dispatch can
    // actually reach. SSOT: `ssm_reserve::mtp_state_slots` — the SAME
    // number `SsmStatePool::new` allocates and the scheduler's spec
    // dispatch guard enforces. At bs<=32 this reproduces the historical
    // `max_batch × blob × (1 + (num_drafts+1) + 1)` byte-for-byte; above
    // 32 it stops reserving verify blobs for slots that can never verify
    // (25.4 GB at bs=64/K=4 on the 27B — the bs=64 preflight refusal).
    // Kill switch: ATLAS_MTP_POOL_FULL_WIDTH (presence) restores
    // full-width sizing on BOTH sides.
    let mtp_state_slots = spark_model::ssm_reserve::mtp_state_slots(args.max_batch_size);
    // Tiered verify slots (2026-08-16): the H-intermediate term is per-slot
    // (`verify_slot_h_intermediates`); DFlash pools are γ-sized and do not
    // follow the MTP ladder, so they reserve uniform full width — mirroring
    // `SsmStatePool::new`'s `num_intermediates != num_drafts + 1` condition.
    // Stage-3 f16-SIZED pool: the FP32 prefill staging arena, ONE blob per
    // slot (shared across layers — see `ssm_h_prefill_stage_bytes`). A
    // separate term for the same reason the replay ring is: it is sized by a
    // SINGLE layer's h blob, not by the across-layers per-seq total every
    // other term here uses. Zero on an FP32-sized pool. `max_batch_size`, not
    // `+1`: this preflight has never counted the pools' dummy slot.
    let ssm_h_stage_bytes = spark_model::ssm_reserve::ssm_h_prefill_stage_bytes(
        args.max_batch_size,
        h_state_bytes,
        spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
    );
    // DFlash pool width: the verify pools are γ-sized (uniform K = γ+1 on
    // EVERY slot — `SsmStatePool::new`'s `uniform_h`), and γ comes from the
    // DRAFTER's checkpoint, which loads long after this preflight. Peek the
    // drafter's config.json for `dflash_config.block_size`; a missing or
    // remote checkpoint falls back to `resolved_dflash_gamma`'s 16 ceiling —
    // the same unknown-γ fallback the pool allocation uses (`γ+1 = 17`), so
    // the miss direction is over-reserve, never under.
    let pool_num_drafts = if args.dflash {
        peek_dflash_block_size(args.draft_model.as_deref())
            .map(spark_model::layers::qwen3_ssm::default_dflash_gamma)
            .unwrap_or_else(|| args.resolved_dflash_gamma(None))
    } else {
        args.resolved_num_drafts()
    };
    let ssm_pool_bytes = spark_model::ssm_reserve::ssm_pool_reserve_bytes(
        args.max_batch_size,
        config.num_ssm_layers() * h_state_bytes,
        config.num_ssm_layers() * conv_state_bytes,
        spec_on_pool,
        pool_num_drafts,
        mtp_state_slots,
        args.dflash,
        // Stage-3 f16-SIZED pool: mirrors `SsmStatePool::new`'s narrowing.
        // Unreachable today (ssm_h_fp16_preconditions refuses the mode
        // above), wired so preflight and allocator cannot diverge when the
        // refusal lifts.
        spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
        // `--ssm-rollback-mode` (published by serve_flags before this runs).
        // Replay drops every per-token verify intermediate; its input ring
        // is the separate term below.
        spark_model::ssm_reserve::ssm_rollback_mode(),
    );
    // Replay-mode verify-window input ring (EXPERIMENTAL scaffold): sized by
    // the SAME SSOT `SsmStatePool::new` allocates through. K ceiling is the
    // MTP `num_drafts + 1` — matching this preflight's existing convention
    // for the conv term (the DFlash γ=17 widening and the pools' dummy slot
    // have never been preflight-counted; the CUDA headroom absorbs them).
    let ssm_replay_ring = if spec_on_pool
        && spark_model::ssm_reserve::ssm_rollback_mode()
            == spark_model::ssm_reserve::SsmRollbackMode::Replay
    {
        spark_model::ssm_reserve::ssm_replay_ring_bytes(
            config.num_ssm_layers(),
            spark_model::ssm_reserve::ssm_replay_row_bytes(
                config.ssm_qkvz_size(),
                config.linear_num_value_heads,
            ),
            pool_num_drafts + 1,
            mtp_state_slots,
        )
    } else {
        0
    };
    let spec_tokens_pre = spec_reserve_tokens(args);
    // B4 (chunked-prefill BF16 KV cliff): the prior `.min(8192)` cap forced
    // every prompt > 8 k to chunk, which compounds K-side BF16 rounding noise
    // at chunk boundaries (per the 4-agent audit 2026-05-27). When the user
    // explicitly passes `--max-prefill-tokens N` (anything other than the
    // default 8192), respect it — no hard cap. Otherwise default to 8192 to
    // bound GDN persistent-buffer reservation for unbounded `max_seq_len`.
    let ssm_prefill_chunk: usize = if config.num_ssm_layers() > 0 {
        if args.max_prefill_tokens != 8192 && args.max_prefill_tokens > 0 {
            args.max_seq_len.min(args.max_prefill_tokens)
        } else {
            args.max_seq_len.min(8192)
        }
    } else {
        0
    };
    let user_set_prefill_pre = args.max_prefill_tokens != 8192;
    let prefill_budget_pre = if user_set_prefill_pre && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    // Issue #15 auto-clamp removed (2026-07-02): snapshot reachability is
    // handled by the tail-checkpoint split in `prefill_chunk_dispatch`, so
    // the budget (and this arena-sizing mirror) stays at full chunk size.
    let max_batch_tokens_pre = prefill_budget_pre
        .max(spec_tokens_pre)
        .max(args.max_batch_size);
    let buffer_arena_bytes = spark_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens_pre,
        args.max_seq_len,
        args.block_size,
        args.max_batch_size,
    )
    .total_bytes();
    // SSM snapshot pool = Marconi prefix-cache region + Phase-C
    // decode-rollback ring. The decode ring is sized per active
    // sequence (ring slots × `max_batch_size`) and only allocated for SSM
    // models. SSOT: `spark_model::ssm_reserve::decode_rollback_ring_slots`
    // makes the SAME decision (same env vars, same constant) the runtime
    // allocation in `TransformerModel::new` makes — including the skip under
    // `--speculative`/`--dflash` (the ring's save/rollback path only runs on
    // plain decode; the spec path rolls back through the verify snapshot).
    // Reserving the ring unconditionally while the runtime skipped it
    // stranded ~38 GB at bs32 on the 27B (75.2 GB SSM reserve vs an 85.2 GB
    // budget at util 0.70) and capped the native batch at ~20.
    // `use_speculative` here MUST mirror what `build_model` passes:
    // `args.speculative || args.dflash`.
    // Kill switch: `ATLAS_SSM_RESERVE_RING_FULL` present ⇒ restore the old
    // unconditional reservation (accounting-only, safe over-reserve;
    // presence-style — `=0` is NOT "off").
    let decode_ring_slots = if std::env::var("ATLAS_SSM_RESERVE_RING_FULL").is_ok() {
        if config.num_ssm_layers() > 0 {
            atlas_kernels::DECODE_ROLLBACK_RING_SLOTS
        } else {
            0
        }
    } else {
        spark_model::ssm_reserve::decode_rollback_ring_slots(
            config.num_ssm_layers(),
            args.speculative || args.dflash,
        )
        .slots
    };
    let ssm_snapshot_bytes = (args.ssm_cache_slots + decode_ring_slots * args.max_batch_size)
        * config.num_ssm_layers()
        * (h_state_bytes + conv_state_bytes);
    // Same predicate as the pool term: DFlash IS a speculative serve and
    // pays the same graph/JIT/scratch overheads the 4 GB headroom exists for.
    let cuda_headroom: usize = if spec_on_pool {
        4 * 1024 * 1024 * 1024
    } else {
        512 * 1024 * 1024
    };
    let gdn_two_phase_bytes: usize = {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        if conv_dim > 0 && config.num_ssm_layers() > 0 {
            let sl = max_batch_tokens_pre;
            sl * conv_dim * 2 + sl * nv * 2 * 4 + sl * value_dim * 2 + sl * value_dim * 2
        } else {
            0
        }
    };
    let inference_reserve: usize = ssm_pool_bytes
        + ssm_h_stage_bytes
        + ssm_replay_ring
        + ssm_snapshot_bytes
        + gdn_two_phase_bytes
        + cuda_headroom;
    let total_reserve = inference_reserve + buffer_arena_bytes;
    if total_reserve > free_mem {
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gb = free_mem as f64 / (1024.0 * 1024.0 * 1024.0);
        let fixed = ssm_pool_bytes + ssm_h_stage_bytes + ssm_snapshot_bytes + cuda_headroom;
        let budget_for_seq_term = free_mem.saturating_sub(fixed) / 2;
        let per_tok_bytes = {
            let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
            let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
            let nv = config.linear_num_value_heads;
            let conv_dim = key_dim * 2 + value_dim;
            if conv_dim > 0 && config.num_ssm_layers() > 0 {
                (conv_dim * 2) + (nv * 2 * 4) + (value_dim * 2) + (value_dim * 2)
            } else {
                0
            }
        };
        let suggested = budget_for_seq_term
            .checked_div(per_tok_bytes)
            .map(|q| q.max(2048))
            .unwrap_or(0);
        let hint = if suggested > 0 && suggested < args.max_seq_len {
            format!(
                " Try --max-seq-len {} (or lower --max-batch-size / --num-drafts).",
                suggested
            )
        } else if args.max_batch_size > 1 {
            " Reduce --max-batch-size.".to_string()
        } else {
            " Use a smaller model or a GPU with more memory.".to_string()
        };
        anyhow::bail!(
            "Preflight failed: inference buffers alone need {:.2} GB but only {:.2} GB is free on the GPU \
             (before weights load). SSM pool + GDN chunked prefill scales with --max-seq-len={} × --max-batch-size={}.{}",
            need_gb,
            free_gb,
            args.max_seq_len,
            args.max_batch_size,
            hint,
        );
    }
    tracing::info!(
        "Preflight reserve: inference={} MB, buffer_arena={} MB (pre-load free: {:.1} GB)",
        inference_reserve / (1024 * 1024),
        buffer_arena_bytes / (1024 * 1024),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    // Q09: per-component breakdown so future MTP/spec-decode reserve
    // jumps are diagnosable from the log alone. Each line is dropped at
    // debug to avoid noise on hot startup paths; flip to info if you
    // need to trace a specific deployment's reserve.
    let spec_on = spec_on_pool;
    tracing::debug!(
        "Preflight reserve breakdown: \
         ssm_pool={} MB ({} max_batch blobs + {} MTP-covered slots × {} verify blobs, \
         {} ssm_layers × (h+conv)), \
         ssm_snapshot={} MB ({} slots), \
         gdn_two_phase={} MB ({} tokens), \
         cuda_headroom={} MB ({}), \
         spec_on={}, num_drafts={}",
        ssm_pool_bytes / (1024 * 1024),
        args.max_batch_size,
        if spec_on_pool { mtp_state_slots } else { 0 },
        if spec_on_pool {
            args.resolved_num_drafts() + 2
        } else {
            0
        },
        config.num_ssm_layers(),
        ssm_snapshot_bytes / (1024 * 1024),
        args.ssm_cache_slots,
        gdn_two_phase_bytes / (1024 * 1024),
        max_batch_tokens_pre,
        cuda_headroom / (1024 * 1024),
        if spec_on { "spec/MTP on" } else { "no spec" },
        spec_on,
        if spec_on {
            args.resolved_num_drafts() as i64
        } else {
            -1
        },
    );
    Ok(ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
    })
}

/// Initialize the GPU backend for the active feature.
///
/// Compile-time dispatch:
/// - `cuda` feature → `AtlasCudaBackend` loading PTX modules from `ptx_set`.
/// - `metal` feature → `MetalGpuBackend` loading metallib modules from
///   `ptx_set` as well. Both arms register the RESOLVED target's modules;
///   `metallib_modules()` is a plain alias of target 0, so registering from
///   it served another model's kernels in a multi-target build.
#[cfg(feature = "cuda")]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    // BEFORE the backend, because constructing it loads every PTX module and
    // a driver-side arch rejection names neither the arch nor the GPU. The
    // compiled arch is only knowable here: `AtlasCudaBackend::new` takes module
    // blobs. WHICH arch string to hand over is `preflight_arch`'s decision and
    // not this call site's — `ptx_set.target.arch` is the base SM with the
    // feature suffix stripped, and judging that would wave `sm_90a` kernels
    // onto a CC 10.0 device.
    super::kernel_gate::gate_arch_preflight(
        args.check_kernels,
        ptx_set,
        spark_runtime::cuda_backend::arch_preflight::preflight_device_arch(
            args.gpu_ordinal,
            spark_runtime::cuda_backend::arch_preflight::preflight_arch(ptx_set),
        ),
    )?;

    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize CUDA backend")?;

    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(backend);
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    // Baseline for self-relative KV budgeting: free memory now (post context +
    // PTX modules, pre weights) minus free-at-build = this process's own
    // footprint, co-tenants excluded. See gpu::baseline_free_bytes.
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "GPU {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    // The RESOLVED target's modules, exactly like the CUDA arm above.
    // `metallib_modules()` is an alias of `ptx_modules()`, which build-codegen
    // emits as a plain alias of TARGET 0 in a multi-target build — so this
    // registered another model's kernels and every lookup for the model
    // actually being served failed.
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::metal_backend::MetalGpuBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize Metal backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "Metal device {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn post_load_memory_audit(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    weight_bytes: usize,
    free_mem: usize,
    inference_reserve: usize,
    total_reserve: usize,
    gdn_two_phase_bytes: usize,
    max_batch_tokens_pre: usize,
) -> Result<()> {
    let estimated_free = free_mem.saturating_sub(weight_bytes);
    let actual_free = gpu.free_memory().unwrap_or(estimated_free);
    let available_free = if actual_free > 0 {
        actual_free
    } else {
        estimated_free
    };
    if available_free < total_reserve {
        let avail_gb = available_free as f64 / (1024.0 * 1024.0 * 1024.0);
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let hint = if args.max_batch_size > 1 {
            format!(
                " Reduce --max-batch-size (currently {}) or --max-seq-len (currently {}).",
                args.max_batch_size, args.max_seq_len
            )
        } else {
            format!(
                " Reduce --max-seq-len (currently {}) or use a smaller model.",
                args.max_seq_len
            )
        };
        anyhow::bail!(
            "Insufficient GPU memory for inference buffers. \
             After loading {:.2} GB of weights, only {:.2} GB remains \
             but {:.2} GB is needed for SSM state pool ({} slots × {} layers) + scratch buffers.{}",
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            avail_gb,
            need_gb,
            args.max_batch_size,
            config.num_ssm_layers(),
            hint,
        );
    }
    if gdn_two_phase_bytes > 0 {
        tracing::info!(
            "GDN chunked prefill reserve: {} MB (chunk_size={}, max_seq_len={})",
            gdn_two_phase_bytes / (1024 * 1024),
            max_batch_tokens_pre,
            args.max_seq_len,
        );
    }
    tracing::info!(
        "Weights: {:.2} GB, estimated free: {:.1} GB, actual free: {:.1} GB (reserve: {} MB)",
        weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        estimated_free as f64 / (1024.0 * 1024.0 * 1024.0),
        actual_free as f64 / (1024.0 * 1024.0 * 1024.0),
        inference_reserve / (1024 * 1024),
    );
    Ok(())
}

/// Peek the DFlash drafter's trained block size (γ) from its config.json
/// without loading the checkpoint — the preflight reserve needs the verify
/// pools' K = γ+1 long before the drafter loads. `None` (missing path, remote
/// HF id, absent field) falls back to the caller's 16 ceiling: the same
/// unknown-γ width `SsmStatePool` allocates, so a failed peek over-reserves
/// rather than re-opening the pledge hole.
fn peek_dflash_block_size(draft_model: Option<&str>) -> Option<usize> {
    let dir = std::path::Path::new(draft_model?);
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let g = v.get("dflash_config")?.get("block_size")?.as_u64()? as usize;
    (g > 0).then_some(g)
}

/// SSOT for "how many rows can one sequence's speculative step occupy" —
/// the term the batch-token floors (`max_batch_tokens_pre` here,
/// `resolve_prefill_budget` in `kv_cache.rs`) take a max against.
///
/// It was the same three-flag expression copy-pasted in both files, and
/// both copies omitted `--dflash` (returning 1 for a serve whose verify
/// step is γ+1 rows wide). Inert today only because the prefill budget's
/// 8192 floor dominates the max — this exists so the two sites cannot
/// drift and so the DFlash width is stated, not defaulted.
///
/// MTP ladder: `num_drafts + 2` (the K = drafts+1 verify rows plus the
/// bonus row — the historical constant, unchanged). DFlash: γ + 1 verify
/// rows (`[last_token, draft_0..γ-1]`), which is the same arithmetic at
/// the effective `num_drafts = γ - 1` the scheduler runs with; γ comes
/// from the drafter's checkpoint via the same peek the pool reserve uses.
pub(crate) fn spec_reserve_tokens(args: &cli::ServeArgs) -> usize {
    if args.dflash {
        let gamma = args.dflash_gamma.unwrap_or_else(|| {
            peek_dflash_block_size(args.draft_model.as_deref())
                .map(spark_model::layers::qwen3_ssm::default_dflash_gamma)
                .unwrap_or_else(|| args.resolved_dflash_gamma(None))
        });
        gamma + 1
    } else if args.speculative || args.self_speculative || args.ngram_speculative {
        args.resolved_num_drafts() + 2
    } else {
        1
    }
}
