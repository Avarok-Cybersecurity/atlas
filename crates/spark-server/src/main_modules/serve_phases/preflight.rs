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
    let ssm_multiplier = if args.speculative || args.self_speculative || args.ngram_speculative {
        1 + (args.num_drafts + 1) + 1
    } else {
        1
    };
    let ssm_pool_bytes = args.max_batch_size
        * config.num_ssm_layers()
        * (h_state_bytes + conv_state_bytes)
        * ssm_multiplier;
    let spec_tokens_pre = if args.speculative || args.self_speculative || args.ngram_speculative {
        args.num_drafts + 2
    } else {
        1
    };
    let ssm_prefill_chunk: usize = if config.num_ssm_layers() > 0 {
        args.max_seq_len.min(8192)
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
    let max_batch_tokens_pre = prefill_budget_pre
        .max(spec_tokens_pre)
        .max(args.max_batch_size);
    let buffer_arena_bytes = spark_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens_pre,
        args.max_seq_len,
        args.block_size,
    )
    .total_bytes();
    let ssm_snapshot_bytes =
        args.ssm_cache_slots * config.num_ssm_layers() * (h_state_bytes + conv_state_bytes);
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
        let suggested = if per_tok_bytes > 0 {
            (budget_for_seq_term / per_tok_bytes).max(2048)
        } else {
            0
        };
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
    Ok(ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
    })
}

pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::cuda_backend::AtlasCudaBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize CUDA backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    tracing::info!(
        "GPU {}: {:.1} GB total, {:.1} GB free",
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
