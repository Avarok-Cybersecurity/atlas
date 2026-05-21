// SPDX-License-Identifier: AGPL-3.0-only

//! Server initialization and runtime: phases 0-11 of the Atlas startup sequence.
//!
//! Refactor wave-4f extracted the bulk of each phase to `serve_phases.rs`
//! (resolve_topology, preflight_reserve, load_weight_store,
//! resolve_kv_cache_config, resolve_tokenizer_runtime, init_nccl_comm,
//! maybe_run_ep_worker, build_model, etc.) — `serve` now reads as a
//! straight call sequence rather than 1.8 KLOC of inline wiring.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::api::InferenceRequest;
use crate::main_modules::AppState;
use crate::main_modules::serve_phases;
use crate::tokenizer::ChatTokenizer;
use crate::{
    cli, conversation_store, rate_limiter, response_store, scheduler, scheduling_policy,
    session_manager,
};

pub(crate) async fn serve(mut args: cli::ServeArgs) -> Result<()> {
    tracing::info!("Atlas Spark starting...");
    tracing::info!("Licensed under AGPL-3.0-only — see /LICENSE in this container");

    // 0. Resolve model directory from HF ID or path
    let model_dir = serve_phases::resolve_model_dir(&args)?;

    tracing::info!("Port: {}", args.port);

    tracing::info!("SSM decode dtype: f32 (full precision)");

    // 1. Load model config (supports HF config.json and Mistral params.json)
    let (mut config, config_json) = serve_phases::load_model_config(&model_dir)?;

    // ModelOpt-exported checkpoints drop a sibling `hf_quant_config.json`
    // whose TOP LEVEL is already the quantization block.
    serve_phases::merge_sidecar_quant_config(&model_dir, &mut config);

    if let Some(ref qc) = config.quantization_config {
        tracing::info!(
            "Quantization config: method={:?}, algo={:?}, format={:?}, {} module(s) in ignore list",
            qc.quant_method,
            qc.quant_algo,
            qc.format,
            qc.ignore_modules.len(),
        );
    }

    tracing::info!(
        "Model config: {} layers, {} attention, {} SSM, {} experts, rope_theta={}, head_dim={}, rotary_dim={}",
        config.num_hidden_layers,
        config.num_attention_layers(),
        config.num_ssm_layers(),
        config.num_experts,
        config.rope_theta,
        config.head_dim,
        config.rotary_dim(),
    );

    // 2. Select kernel target and initialize GPU backend
    //
    // Each kernel target declares which (model_type, hidden_size) pairs it supports
    // via [[model_types]] in MODEL.toml. Exact hidden_size matches win over wildcards.
    let ptx_set = atlas_kernels::ptx_for_config(&config.model_type, config.hidden_size)
        .with_context(|| {
            format!(
                "No compiled kernel target matches model_type '{}' / hidden_size={}. \
             Available targets: {:?}",
                config.model_type,
                config.hidden_size,
                atlas_kernels::available_targets()
                    .iter()
                    .map(|t| &t.target.model)
                    .collect::<Vec<_>>(),
            )
        })?;
    let sampling_presets = ptx_set.sampling;
    tracing::info!(
        "Selected kernel target: {} ({} modules)",
        ptx_set.target,
        ptx_set.modules.len(),
    );

    // Apply MODEL.toml [behavior].default_num_drafts unless user passed --num-drafts.
    serve_phases::apply_model_default_num_drafts(&mut args, &ptx_set);

    let (gpu, free_mem) = serve_phases::init_gpu_backend(&args, &ptx_set)?;

    // ── Pre-load reserve preflight ──
    let serve_phases::ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
    } = serve_phases::preflight_reserve(&args, &config, free_mem)?;
    let total_reserve = inference_reserve + buffer_arena_bytes;

    // 2a-2. OOM watchdog: background async task that polls GPU memory every 2s.
    // On GB10 unified memory, GPU OOM = system freeze, so we exit(1) early.
    // Threshold: 2 GB (enough to detect runaway allocation before system locks up).
    //
    // CUDA-only: Apple Silicon UMA already exposes `currentAllocatedSize`
    // and the OS handles memory pressure via Metal's working-set policy,
    // so the dedicated watchdog isn't needed.
    #[cfg(feature = "cuda")]
    let _oom_watchdog = spark_runtime::cuda_backend::spawn_oom_watchdog(
        2048, // 2 GB threshold
        std::time::Duration::from_secs(2),
    );
    #[cfg(feature = "cuda")]
    tracing::info!("OOM watchdog started (threshold: 2 GB, interval: 2s)");

    // 2b. Resolve TP / EP topology and set on model config.
    let serve_phases::Topology {
        world_size,
        tp_size: _tp_size,
        ep_size,
        tp_rank: _tp_rank,
        ep_rank,
    } = serve_phases::resolve_topology(&args, &mut config)?;
    // FP8 KV calibration: CLI flag overrides MODEL.toml default.
    config.fp8_kv_calibration_tokens = if args.fp8_kv_calibration_tokens > 0 {
        args.fp8_kv_calibration_tokens
    } else {
        ptx_set.behavior.fp8_kv_calibration_tokens
    };

    // 3. Load model weights
    let oom_reserve_bytes = args.oom_guard_mb * 1024 * 1024;
    tracing::info!("OOM guard reserve: {} MB", args.oom_guard_mb);
    let store = serve_phases::load_weight_store(
        &args,
        &config,
        &model_dir,
        gpu.as_ref(),
        ep_rank,
        ep_size,
        oom_reserve_bytes,
    )?;

    // 3b. Auto-detect weight key prefix for nested models.
    serve_phases::auto_detect_weight_prefix(&store, &mut config);

    // Pre-flight weight-store / config consistency check. Runs before
    // NCCL init so a mis-matched checkpoint (wrong expert count, MiniMax
    // + MTP tensors + `--speculative`, missing embedding, etc.) aborts
    // this rank with a readable error BEFORE rank 1 ever connects or
    // `ncclCommInitRank` is called. Several community re-quants of
    // MiniMax M2.7 hang on NCCL init today because the actual mismatch
    // only surfaces later inside `build_model`; this check surfaces it
    // up-front.
    spark_model::preflight::preflight(&store, &config, args.speculative)
        .context("Checkpoint pre-flight check failed")?;

    // Resolve and log the QuantFormat dispatch decision now so a silent
    // fallback is visible in the server log (and not just in the
    // detection code path mid-load). The returned trait object is
    // currently only consulted via `detect_nvfp4_variant`; explicit
    // use at each load site is a follow-up migration.
    let quant_format = spark_model::quant_format::detect_quant_format(&config, &store);
    tracing::info!(
        "Quantization format: {} (base variant {:?}), ignored globs = {}",
        quant_format.name(),
        quant_format.base_variant(),
        match &config.quantization_config {
            Some(qc) => qc.ignore_modules.len(),
            None => 0,
        },
    );

    // 4. Post-load OOM check + audit log.
    serve_phases::post_load_memory_audit(
        &args,
        &config,
        gpu.as_ref(),
        store.total_bytes(),
        free_mem,
        inference_reserve,
        total_reserve,
        gdn_two_phase_bytes,
        max_batch_tokens_pre,
    )?;

    // 5. Build model via factory.
    let serve_phases::PrefillBudget {
        prefill_budget,
        max_batch_tokens,
        spec_tokens: _spec_tokens,
    } = serve_phases::resolve_prefill_budget(&args, ssm_prefill_chunk);
    if args.dflash && args.enable_prefix_caching {
        tracing::warn!(
            "dflash: --enable-prefix-caching has a community-reported correctness regression on SM12.x with DFlash; outputs may be wrong on multi-turn cache hits. Run a greedy diff-test against a non-DFlash baseline before relying on outputs."
        );
    }
    let prefix_cache = serve_phases::build_prefix_cache(&args);
    let comm = serve_phases::init_nccl_comm(&args, gpu.as_ref(), world_size)?;
    if args.profile {
        // SAFETY: called before any threads are spawned.
        unsafe {
            std::env::set_var("ATLAS_PROFILE", "1");
        }
    }
    serve_phases::cap_vocab_size_to_tokenizer(&model_dir, &mut config);
    let serve_phases::KvCacheConfig {
        effective_kv_dtype_str: _,
        kv_dtype,
        layer_dtypes,
        hss_cache_blocks_per_seq,
    } = serve_phases::resolve_kv_cache_config(&args, &config, ptx_set.behavior.default_kv_dtype)?;
    let dflash_drafter_state = serve_phases::load_dflash_drafter(&args, &ptx_set, gpu.as_ref())?;
    let dflash_args =
        dflash_drafter_state
            .as_ref()
            .map(|(s, c)| spark_model::factory::DflashBuildArgs {
                drafter_store: s,
                drafter_config: c.clone(),
                gamma: Some(args.dflash_gamma),
                window_size: if args.dflash_window_size > 0 {
                    Some(args.dflash_window_size)
                } else {
                    None
                },
            });
    let model = serve_phases::build_model(
        &args,
        &config,
        &store,
        gpu,
        max_batch_tokens,
        kv_dtype,
        inference_reserve,
        layer_dtypes,
        hss_cache_blocks_per_seq,
        prefix_cache,
        comm,
        dflash_args,
    )?;

    // Phase 6.3 — HSS config built early so the EP worker can install it.
    let early_high_speed_swap_cfg = serve_phases::build_high_speed_swap_config(&args)?;

    // EP worker: rank > 0 enters command loop, returns when head exits.
    let mut model_opt = Some(model);
    if serve_phases::maybe_run_ep_worker(&args, &mut model_opt, &early_high_speed_swap_cfg)? {
        return Ok(());
    }
    let model = model_opt.expect("head retains model on rank 0");

    // Build EOS token list from generation_config.json (authoritative) or config.json fallback
    let mut eos_tokens = serve_phases::load_eos_tokens(&model_dir, &config);

    // Read default sampling parameters from generation_config.json.
    let serve_phases::SamplingDefaults {
        temperature: default_temperature,
        top_k: default_top_k,
        top_p: default_top_p,
        top_n_sigma: default_top_n_sigma,
        min_p: default_min_p,
    } = serve_phases::load_sampling_defaults(&model_dir, &args);

    // 6. Load tokenizer
    // Thinking support is derived from model capabilities, not hardcoded model names.
    // Models with SSM layers or Qwen3.5-style architecture support <think> tokens.
    // The --enable-thinking flag controls OPEN-ENDED vs CLOSED thinking.
    let caps = config.capabilities();
    let supports_thinking = caps.supports_thinking;
    let tokenizer = ChatTokenizer::from_model_dir(
        &model_dir,
        eos_tokens[0],
        supports_thinking,
        &config.model_type,
        Some(std::path::Path::new(".")), // repo root for override templates
    )?;

    // Tokenizer-derived runtime: vocab cap, reasoning parser, think tokens,
    // im_start hard-stop, reflection suppression, tool-call open/close tokens,
    // and the XGrammar engine.
    let serve_phases::TokenizerRuntime {
        reasoning_parser_box,
        think_end_token,
        think_start_token,
        reflection_suppress_ids,
        tool_call_start_token,
        tool_call_end_token,
        grammar_engine,
    } = serve_phases::resolve_tokenizer_runtime(
        &args,
        &mut config,
        &tokenizer,
        &mut eos_tokens,
        supports_thinking,
    );

    // 7. Create scheduler channel + spawn scheduler
    let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>(args.max_num_seqs);

    let model_name = serve_phases::resolve_model_name(&args, &config_json, &model_dir);

    let scheduler_model = model;
    let scheduler_eos = eos_tokens;
    // EP: force batch_size=1 (worker protocol is single-sequence).
    // MTP speculative decoding IS supported with EP via verify broadcast protocol.
    let max_batch_size = if world_size > 1 {
        tracing::info!("EP active: forcing max_batch_size=1");
        1
    } else {
        args.max_batch_size
    };
    // `use_speculative` gates the scheduler's `step_mtp` path which already
    // dispatches both MTP and DFlash proposers via the shared `DraftProposer`
    // trait + the `drafts.len() ≥ 4` ladder route to `step_verify_dflash`
    // (scheduler.rs:3013). So `--dflash` enables `use_speculative` too.
    let use_speculative = (args.speculative || args.dflash) && scheduler_model.has_proposer();
    let use_self_spec = args.self_speculative && scheduler_model.has_self_speculative();
    let use_ngram_spec = args.ngram_speculative;
    // For DFlash, force `num_drafts = γ - 1` so the scheduler asks the
    // proposer for γ tokens (DraftProposer::propose semantics: "up to
    // num_drafts" → drafts.len() = γ → routes to step_verify_dflash).
    let num_drafts = if args.dflash {
        args.dflash_gamma.saturating_sub(1).max(1)
    } else {
        args.num_drafts
    };

    if args.dflash {
        tracing::info!(
            "DFlash speculative decoding: ENABLED (γ={}, window={}, drafter installed)",
            args.dflash_gamma,
            if args.dflash_window_size == 0 {
                "full".to_string()
            } else {
                args.dflash_window_size.to_string()
            }
        );
    } else if use_ngram_spec {
        tracing::info!("N-gram speculative decoding: ENABLED (K=2 verify, CPU proposer)");
    } else if use_self_spec {
        tracing::info!(
            "Self-speculative decoding: ENABLED ({num_drafts} drafts/step, layer-skipping)"
        );
    } else if use_speculative {
        tracing::info!("Speculative decoding: ENABLED ({num_drafts} drafts/step)");
    } else if scheduler_model.has_proposer() {
        tracing::info!(
            "MTP proposer available but speculative decoding disabled (use --speculative to enable)"
        );
    }

    let policy: Box<dyn scheduling_policy::SchedulingPolicy> = match args.scheduling_policy.as_str()
    {
        "fifo" => {
            tracing::info!("Scheduling policy: FIFO");
            Box::new(scheduling_policy::FifoPolicy)
        }
        "slai" => {
            tracing::info!(
                "Scheduling policy: SLAI (TBT deadline={}ms)",
                args.tbt_deadline_ms,
            );
            Box::new(scheduling_policy::SlaiPolicy::new(args.tbt_deadline_ms))
        }
        other => anyhow::bail!(
            "Unknown scheduling policy '{}'. Supported: fifo, slai",
            other,
        ),
    };

    // Use prefill_budget (which accounts for SSM no-chunking override) instead of raw CLI arg.
    let max_prefill_tokens = prefill_budget;
    let swap_space_gb = args.swap_space_gb;
    let block_size = args.block_size;

    // ── --high-speed-swap config validation (PCND: required-when-set) ──
    let high_speed_swap_cfg = serve_phases::validate_head_high_speed_swap(
        &args,
        &early_high_speed_swap_cfg,
        swap_space_gb,
    )?;

    let adaptive_sampling = args.adaptive_sampling;
    let session_manager = session_manager::SessionSsmManager::new(600); // 10 min TTL
    // Spontaneous-thinking budget: when the model emits `<think>` without
    // the request having explicitly enabled thinking, this caps how many
    // thinking tokens are allowed before `</think>` is force-emitted. CLI
    // override beats MODEL.toml. Used by the scheduler in place of a
    // previous hard-coded 512 fallback so MODEL.toml can right-size the
    // cap per architecture.
    let scheduler_spontaneous_think_budget = args
        .max_thinking_budget
        .unwrap_or(ptx_set.behavior.max_thinking_budget);
    std::thread::spawn(move || {
        scheduler::run(
            scheduler_model,
            request_rx,
            scheduler_eos,
            max_batch_size,
            use_speculative,
            num_drafts,
            policy,
            max_prefill_tokens,
            max_batch_tokens,
            use_self_spec,
            use_ngram_spec,
            swap_space_gb,
            high_speed_swap_cfg,
            block_size,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            reflection_suppress_ids,
            grammar_engine,
            adaptive_sampling,
            session_manager,
            scheduler_spontaneous_think_budget,
        );
    });

    // Tool call parser resolution: CLI > MODEL.toml > defaults table.
    let tool_call_parser = serve_phases::resolve_tool_call_parser(&args, &ptx_set, &config)?;

    // 8. Build app state
    let model_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let response_store = response_store::ResponseStore::from_env();
    let rate_limiter = rate_limiter::RateLimiter::from_env();
    let conversation_store = conversation_store::ConversationStore::from_env();
    serve_phases::log_response_store_audit(&response_store, &rate_limiter);
    let dump_writer = serve_phases::open_dump_writer(&args);
    let auth = build_auth_config(&args)?;
    let state = Arc::new(AppState {
        tokenizer,
        model_name,
        max_seq_len: args.max_seq_len,
        request_tx,
        vision_config: config.vision.clone(),
        default_temperature,
        default_top_k,
        default_top_p,
        default_top_n_sigma,
        default_min_p,
        tool_call_parser,
        reasoning_parser: reasoning_parser_box,
        think_end_token_id: think_end_token,
        think_start_token_id: think_start_token,
        tool_max_tokens: args.tool_max_tokens,
        sampling_presets,
        tool_call_start_token_id: tool_call_start_token,
        auto_compact_threshold: args.auto_compact,
        model_ready: model_ready.clone(),
        request_timeout: args.request_timeout,
        // Behavior and effective_context from MODEL.toml, embedded at build time.
        effective_context: 0, // TODO: embed effective_context in TargetPtxSet
        behavior: {
            let mut b = ptx_set.behavior.clone();
            if let Some(cli_budget) = args.max_thinking_budget {
                b.max_thinking_budget = cli_budget;
            }
            b
        },
        disable_thinking: args.disable_thinking,
        default_chat_template_kwargs: args
            .default_chat_template_kwargs
            .as_ref()
            .and_then(|s| crate::openai::ChatTemplateKwargs::from_json(s)),
        response_store,
        rate_limiter,
        conversation_store,
        dump_writer,
        auth,
    });

    serve_phases::log_behavior_audit(&args, &ptx_set);

    // 9-11. Build router + start HTTP server (extracted: serve_router.rs).
    crate::main_modules::serve_router::build_and_serve(state, model_ready, &args.bind, args.port)
        .await
}

/// Resolve `--require-auth` / `--auth-tokens-file` / `--auth-token` into an
/// optional `AuthConfig`. Validates at startup so misconfigurations fail
/// loudly instead of letting an unauthenticated server run silently.
fn build_auth_config(args: &cli::ServeArgs) -> Result<Option<Arc<crate::auth::AuthConfig>>> {
    if !args.require_auth {
        if args.auth_tokens_file.is_some() || args.auth_token.is_some() {
            tracing::warn!(
                "--auth-tokens-file / --auth-token supplied without --require-auth; \
                 tokens are loaded but the auth gate is OFF. Pass --require-auth to enforce."
            );
        }
        return Ok(None);
    }
    let cfg = match (&args.auth_tokens_file, &args.auth_token) {
        (Some(path), None) => crate::auth::AuthConfig::from_file(path)?,
        (None, Some(tok)) => {
            tracing::warn!(
                "--auth-token sets the bearer token via the command line; the value \
                 is visible to other local users via `ps`/`/proc/<pid>/cmdline`. \
                 Use --auth-tokens-file with permissions 0600 in production."
            );
            crate::auth::AuthConfig::from_inline(tok)?
        }
        (None, None) => {
            return Err(anyhow::anyhow!(
                "--require-auth was set but neither --auth-tokens-file nor \
                 --auth-token was supplied. Pick one (a tokens file is preferred)."
            ));
        }
        (Some(_), Some(_)) => unreachable!("clap conflicts_with should have rejected this"),
    };
    tracing::info!(
        "auth: require_auth=ON ({} bearer token{} loaded)",
        cfg.token_count(),
        if cfg.token_count() == 1 { "" } else { "s" },
    );
    Ok(Some(Arc::new(cfg)))
}
