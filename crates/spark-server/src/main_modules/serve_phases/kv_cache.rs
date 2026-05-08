// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-budget + KV-cache dtype resolution.

use anyhow::Result;

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) struct PrefillBudget {
    pub(crate) prefill_budget: usize,
    pub(crate) max_batch_tokens: usize,
    pub(crate) spec_tokens: usize,
}

pub(crate) fn resolve_prefill_budget(
    args: &cli::ServeArgs,
    ssm_prefill_chunk: usize,
) -> PrefillBudget {
    let spec_tokens = if args.speculative || args.self_speculative || args.ngram_speculative {
        args.num_drafts + 2
    } else {
        1
    };
    let user_set_prefill = args.max_prefill_tokens != 8192;
    let prefill_budget_pre_hss = if user_set_prefill && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    // Issue #15: when prefix caching + SSM snapshots are both enabled, the
    // SSM snapshot taken at finalize_last (token_count = full prompt length)
    // is unreachable from future requests because the radix-tree match for
    // a different prompt length will be < tokens.len(). Intermediate
    // checkpoints are saved by `prefill_b_save_checkpoint` only at chunk
    // ends whose end_block is a multiple of `ssm_checkpoint_interval`, so a
    // single-chunk prefill produces zero reachable snapshots. Auto-clamp
    // the prefill budget to a checkpoint-aligned size so chunked prefill
    // actually fires and downstream agentic conversations get cache hits.
    let prefill_budget_pre_hss = if !user_set_prefill
        && args.enable_prefix_caching
        && args.ssm_checkpoint_interval > 0
        && args.ssm_cache_slots > 0
    {
        let target = args.ssm_checkpoint_interval * args.block_size;
        if prefill_budget_pre_hss > target && target > 0 {
            tracing::info!(
                "--enable-prefix-caching with --ssm-checkpoint-interval={} \
                 and --block-size={}: auto-clamping max_prefill_tokens \
                 from {} to {} so chunked prefill fires at SSM-checkpoint \
                 boundaries (issue #15). Override with --max-prefill-tokens \
                 to keep the larger value.",
                args.ssm_checkpoint_interval,
                args.block_size,
                prefill_budget_pre_hss,
                target,
            );
            target
        } else {
            prefill_budget_pre_hss
        }
    } else {
        prefill_budget_pre_hss
    };
    let prefill_budget = if args.high_speed_swap {
        let hss_cap_tokens = args.high_speed_swap_cache_blocks_per_seq as usize * args.block_size;
        let hss_chunk_max = hss_cap_tokens.saturating_sub(args.max_batch_size);
        let clamped = prefill_budget_pre_hss.min(hss_chunk_max);
        if clamped < prefill_budget_pre_hss {
            tracing::info!(
                "--high-speed-swap: clamping max_prefill_tokens from {} to {} \
                 (cap {} × bs {} − max_batch_size {}) to keep chunked prefill \
                 within the rolling HBM window",
                prefill_budget_pre_hss,
                clamped,
                args.high_speed_swap_cache_blocks_per_seq,
                args.block_size,
                args.max_batch_size,
            );
        }
        // Issue #31 was fixed by the cursor-advance-during-slide change in
        // `block_mgmt::ensure_blocks_through_{prefill,decode}` plus the
        // post-prefix-cache-hit cursor advance in
        // `prefill_b/prefix_lookup.rs`. Long prompts with HSS now work; the
        // earlier startup WARN that flagged this combination as broken is
        // obsolete and has been removed. Per-config diagnostics live at
        // INFO level on the next line.
        if args.max_seq_len > hss_cap_tokens {
            tracing::info!(
                "--high-speed-swap engaged: cap={} blocks × bs={} = {} tokens HBM-resident, \
                 --max-seq-len={} tokens total. Prefill slides will advance per-layer offload \
                 cursors as the window moves so older blocks stay reachable on disk.",
                args.high_speed_swap_cache_blocks_per_seq,
                args.block_size,
                hss_cap_tokens,
                args.max_seq_len,
            );
        }
        clamped
    } else {
        prefill_budget_pre_hss
    };
    let max_batch_tokens = (prefill_budget + args.max_batch_size)
        .max(spec_tokens)
        .max(args.max_batch_size);
    tracing::info!(
        "Prefill config: ssm_prefill_chunk={}, args.max_prefill_tokens={}, prefill_budget={}, max_batch_tokens={}",
        ssm_prefill_chunk,
        args.max_prefill_tokens,
        prefill_budget,
        max_batch_tokens,
    );
    if args.max_prefill_tokens == 0 && args.max_seq_len > 32768 {
        tracing::warn!(
            "--max-prefill-tokens=0 with --max-seq-len={} disables chunked prefill. \
             Long agentic sessions may eventually fail with 'CUDA kernel launch failed (status 1)' \
             when an unchunked prefill exceeds device launch grid limits. \
             Consider --max-prefill-tokens=8192 (default) for sessions that grow past 32K tokens.",
            args.max_seq_len,
        );
    }
    PrefillBudget {
        prefill_budget,
        max_batch_tokens,
        spec_tokens,
    }
}

pub(crate) struct KvCacheConfig {
    pub(crate) effective_kv_dtype_str: String,
    pub(crate) kv_dtype: spark_runtime::kv_cache::KvCacheDtype,
    pub(crate) layer_dtypes: Vec<spark_runtime::kv_cache::KvCacheDtype>,
    pub(crate) hss_cache_blocks_per_seq: Option<u32>,
}

pub(crate) fn resolve_kv_cache_config(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    behavior_default_kv_dtype: &str,
) -> Result<KvCacheConfig> {
    // Resolution rules:
    //   1. No MODEL.toml override        → use args.kv_cache_dtype as-is.
    //   2. User matches MODEL.toml       → silent (correct config).
    //   3. User at CLI default ("fp8")   → apply MODEL.toml override + info log.
    //   4. User explicitly mismatches    → respect user, warn loudly.
    // Rule 4 catches the gemma/mistral collapse (NVFP4 KV → `<unused>` /
    // `后汉书` token loop) and the FP8 KV mismatch on bf16-required attention
    // paths. We respect the user's choice so experimentation isn't blocked,
    // but the warning makes the cause traceable when decode goes degenerate.
    let effective_kv_dtype_str: String = if behavior_default_kv_dtype.is_empty()
        || args.kv_cache_dtype == behavior_default_kv_dtype
    {
        args.kv_cache_dtype.clone()
    } else if args.kv_cache_dtype == "fp8" {
        tracing::info!(
            "KV cache dtype: {} (from MODEL.toml default_kv_dtype, override with --kv-cache-dtype)",
            behavior_default_kv_dtype,
        );
        behavior_default_kv_dtype.to_string()
    } else {
        tracing::warn!(
            "KV cache dtype: {} (user override). MODEL.toml recommends '{}' for this \
             model — mismatched KV dtype is a known cause of decode-path corruption \
             (e.g. gemma `<unused>` collapse, mistral character-token loops on NVFP4 KV). \
             Pass --kv-cache-dtype {} to use the recommended value.",
            args.kv_cache_dtype,
            behavior_default_kv_dtype,
            behavior_default_kv_dtype,
        );
        args.kv_cache_dtype.clone()
    };
    let kv_dtype: spark_runtime::kv_cache::KvCacheDtype = effective_kv_dtype_str.parse()?;
    if kv_dtype == spark_runtime::kv_cache::KvCacheDtype::Fp8 {
        if config.fp8_kv_calibration_tokens > 0 {
            tracing::info!(
                "FP8 KV cache with online calibration: tracking max |K|/|V| during \
                 first {} tokens to compute per-tensor scales.{}",
                config.fp8_kv_calibration_tokens,
                if args.fp8_kv_calibration_tokens == 0 {
                    " (auto-enabled from MODEL.toml)"
                } else {
                    ""
                },
            );
        } else {
            tracing::warn!(
                "FP8 KV cache selected. This requires calibrated k_scale/v_scale in the model \
                 checkpoint. Without scales (default=1.0), BF16 values are silently clipped to \
                 E4M3 range [-448, 448], destroying dynamic range. Use --fp8-kv-calibration-tokens 256 \
                 for online calibration, or --kv-cache-dtype nvfp4/bf16 if your model lacks k/v scales."
            );
        }
    }
    let num_attn_layers = config.num_attention_layers();
    let kv_hp_layers: usize = match args.kv_high_precision_layers.to_lowercase().as_str() {
        "max" | "all" => num_attn_layers,
        "auto" => 2,
        s => s.parse().unwrap_or_else(|_| {
            tracing::warn!("Invalid --kv-high-precision-layers '{}', using 0", s);
            0
        }),
    };
    let kv_hp_layers = if kv_hp_layers == 0
        && matches!(
            kv_dtype,
            spark_runtime::kv_cache::KvCacheDtype::Turbo3
                | spark_runtime::kv_cache::KvCacheDtype::Turbo4
                | spark_runtime::kv_cache::KvCacheDtype::Turbo8
        ) {
        let auto_hp = ((num_attn_layers as f32 / 3.0).ceil() as usize).max(2);
        tracing::info!(
            "Auto-enabling --kv-high-precision-layers {} for {} ({}/{} attn layers BF16; \
             scaled with attn-layer count to keep accumulated turbo quant error tractable)",
            auto_hp,
            effective_kv_dtype_str,
            (auto_hp * 2).min(num_attn_layers),
            num_attn_layers,
        );
        auto_hp
    } else {
        kv_hp_layers
    };
    if kv_hp_layers == 0 && kv_dtype != spark_runtime::kv_cache::KvCacheDtype::Bf16 {
        tracing::warn!(
            "⚠ --kv-high-precision-layers is 0: all KV cache layers use {} precision. \
             NVFP4 models may hallucinate or lose coherence at long context. \
             Consider --kv-high-precision-layers max (or 2-5) for better quality.",
            effective_kv_dtype_str,
        );
    }
    let layer_dtypes =
        crate::main_modules::build_layer_kv_dtypes(kv_dtype, num_attn_layers, kv_hp_layers);
    let hss_cache_blocks_per_seq = if args.high_speed_swap {
        Some(args.high_speed_swap_cache_blocks_per_seq)
    } else {
        None
    };
    Ok(KvCacheConfig {
        effective_kv_dtype_str,
        kv_dtype,
        layer_dtypes,
        hss_cache_blocks_per_seq,
    })
}
