// SPDX-License-Identifier: AGPL-3.0-only

//! GPU init + pre-load reserve preflight + post-load OOM check.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

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
    let spec_on_pool = args.speculative || args.self_speculative || args.ngram_speculative;
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
    let ssm_blob_bytes = config.num_ssm_layers() * (h_state_bytes + conv_state_bytes);
    let mtp_state_slots = spark_model::ssm_reserve::mtp_state_slots(args.max_batch_size);
    let ssm_pool_bytes = spark_model::ssm_reserve::ssm_pool_reserve_bytes(
        args.max_batch_size,
        ssm_blob_bytes,
        spec_on_pool,
        args.num_drafts,
        mtp_state_slots,
    );
    let spec_tokens_pre = if args.speculative || args.self_speculative || args.ngram_speculative {
        args.num_drafts + 2
    } else {
        1
    };
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
    let cuda_headroom: usize =
        if args.speculative || args.self_speculative || args.ngram_speculative {
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
    let inference_reserve: usize =
        ssm_pool_bytes + ssm_snapshot_bytes + gdn_two_phase_bytes + cuda_headroom;
    let total_reserve = inference_reserve + buffer_arena_bytes;
    if total_reserve > free_mem {
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let free_gb = free_mem as f64 / (1024.0 * 1024.0 * 1024.0);
        let fixed = ssm_pool_bytes + ssm_snapshot_bytes + cuda_headroom;
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
        if spec_on_pool { args.num_drafts + 2 } else { 0 },
        config.num_ssm_layers(),
        ssm_snapshot_bytes / (1024 * 1024),
        args.ssm_cache_slots,
        gdn_two_phase_bytes / (1024 * 1024),
        max_batch_tokens_pre,
        cuda_headroom / (1024 * 1024),
        if spec_on { "spec/MTP on" } else { "no spec" },
        spec_on,
        if spec_on { args.num_drafts as i64 } else { -1 },
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

/// `ATLAS_SSM_H_FP16` refuses rather than degrades.
///
/// The flag narrows the GDN h-state to FP16. Stage 1 shipped twins of the two
/// NON-speculative decode kernels — `gated_delta_rule_decode_f16_strided_norm_half`
/// (batched) and `..._f16_norm` (per-sequence, taken at n == 1 and whenever
/// pool slots fragment out of slice order). Stage 2 adds twins of the MTP
/// verify path — `gated_delta_rule_wy{2,3,4}_f16` plus the register-resident
/// `wy{2,3}_resident_f16` — so the state stays FP16 through a verify step and
/// the flag composes with `--speculative`. That matters because the ladder's
/// low and middle rungs are all spec-ON, so before stage 2 they could not use
/// the lever at all.
///
/// Every remaining h-state reader is still FP32, and an FP32 kernel pointed at
/// an FP16 pool produces fluent garbage rather than an error — so the
/// unsupported configurations are rejected here, at boot, instead of being
/// discovered in a benchmark.
///
/// Sizing is untouched: the pool stays FP32-sized (prefill still writes FP32)
/// and the FP16 state occupies the first half of the same region, so no
/// reserve arithmetic above depends on the flag.
fn ssm_h_fp16_preconditions(args: &cli::ServeArgs, config: &ModelConfig) -> Result<()> {
    // SSOT: the same resolution the kernels dispatch on — `--ssm-h-dtype`,
    // falling back to `ATLAS_SSM_H_FP16`. This check used to decode the
    // environment independently, which is how a preflight could pass on a
    // reading the kernels did not share.
    if !spark_model::layers::qwen3_ssm::ssm_h_fp16_enabled() || config.num_ssm_layers() == 0 {
        return Ok(());
    }
    // STAGE 2 lifted the blanket refusal on `--speculative`: the MTP verify
    // path's WY kernels now have FP16 h-state twins
    // (`gated_delta_rule_wy{2,3,4}_f16` and the register-resident
    // `wy{2,3}_resident_f16`), so the h-state stays FP16 end-to-end through a
    // verify step. What is still refused is any configuration that can reach a
    // K with NO twin, because the fallback is an FP32 kernel over an FP16 pool
    // — which does not fault, it emits fluent garbage.
    //
    // The reachable K is bounded by the draft count: the ladder's draft count
    // is capped by `--num-drafts`, and K = drafts + 1 rows per sequence. Twins
    // exist for K = 2, 3, 4, so up to 3 drafts is supported. Above that the
    // width lands on the wyN (K=5..8) or wy17 DFlash arms, which are FP32-only.
    if args.self_speculative || args.ngram_speculative {
        anyhow::bail!(
            "--ssm-h-dtype f16 supports --speculative (MTP) only. The self-speculative and \
             ngram-speculative verify paths still write the h-state as FP32, and an FP32 \
             kernel over an FP16 pool produces fluent garbage rather than an error. Run \
             without --self-speculative/--ngram-speculative, or use --ssm-h-dtype f32."
        );
    }
    if args.speculative && args.num_drafts > 3 {
        anyhow::bail!(
            "--ssm-h-dtype f16 supports up to 3 drafts (K <= 4 verify rows); --num-drafts is \
             {}. Wider verify widths dispatch the wyN (K=5..8) / wy17 arms, which have no \
             FP16 h-state twin. Lower --num-drafts to 3, or use --ssm-h-dtype f32.",
            args.num_drafts
        );
    }
    if !spark_model::layers::qwen3_ssm::gdn_fused_norm_enabled() {
        anyhow::bail!(
            "--ssm-h-dtype f16 requires --gdn-fused-norm — the non-fused decode arms \
             (gated_delta_rule_decode, ..._decode_f32_strided) have no FP16 twin in stage 1."
        );
    }
    if std::env::var("ATLAS_GDN_FUSED_CONV").ok().as_deref() == Some("1") {
        anyhow::bail!(
            "--ssm-h-dtype f16 is incompatible with ATLAS_GDN_FUSED_CONV=1 —              gated_delta_rule_decode_f32_conv_norm has no FP16 twin in stage 1."
        );
    }
    if config.linear_key_head_dim != 128 || config.linear_value_head_dim != 128 {
        anyhow::bail!(
            "--ssm-h-dtype f16 needs linear head dims 128/128 (the FP16 twins size their shared              memory for k_dim == 128); this model is {}/{}",
            config.linear_key_head_dim,
            config.linear_value_head_dim
        );
    }
    tracing::info!(
        "--ssm-h-dtype f16: GDN h-state stored FP16 during decode AND MTP verify (pool stays \
         FP32-sized; prefill unchanged). Scan replica at n=128: 183 -> 84 ms/step."
    );
    Ok(())
}
