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

    // CLI `--lm-head-dtype` override (replaces ATLAS_LMHEAD_BF16). Validate eagerly (PCND).
    // Sets both `lm_head_bf16_override` (skip/keep-quantized signal consumed by
    // `skip_lm_head_quantization()`) and `lm_head_fp8` (when quantizing, pick FP8 w8a16
    // over NVFP4). `fp8` reuses `Some(false)` ("force quantized lm_head") and additionally
    // routes that quantization to FP8 — additive, leaves nvfp4/bf16/default byte-identical.
    let (lm_head_bf16_override, lm_head_fp8) = match args.lm_head_dtype.as_str() {
        "default" => (None, false),
        "bf16" => (Some(true), false),
        // `Some(false)` = force the model's NVFP4-packed lm_head (skip_lm_head_quantization
        // returns false). BF16-out fast path (w4a16_gemv) — NOT use_fp32_logits, which would
        // force host-side sampling (~6 tok/s). Decode-speed lever; quality-gate for argmax flips.
        "nvfp4" => (Some(false), false),
        // FP8: force a quantized lm_head, but use runtime FP8 (E4M3, per-row scales,
        // w8a16_gemv decode) instead of NVFP4. Mirrors the NVFP4 path's structure.
        "fp8" => (Some(false), true),
        other => {
            anyhow::bail!(
                "--lm-head-dtype must be 'default', 'bf16', 'nvfp4', or 'fp8', got '{other}'"
            )
        }
    };
    config.lm_head_bf16_override = lm_head_bf16_override;
    config.lm_head_fp8 = lm_head_fp8;

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

    // QV1 (2026-05-26): kernel ↔ model quant compatibility validation.
    //
    // `ptx_for_config` selects on (model_type, hidden_size) but not on
    // QUANT. With ATLAS_TARGET_QUANT=* the build emits one bundle per
    // model whose label happens to be the first variant compiled
    // ("nvfp4") even when the bundle contains native FP8 dispatch too.
    // For now we accept the historically-compatible pairs hardcoded in
    // `quant_pair_compatible` (and only those). Anything else hard
    // errors RIGHT HERE with an explicit "rebuild with X" message,
    // before any weight loading runs and any silent garbage path can
    // be entered. A future refinement moves the compat list into
    // MODEL.toml `[kernel].supported_quants`.
    let model_quant = canonicalize_model_quant(&config);
    let kernel_quant = ptx_set.target.quant;
    if !quant_pair_compatible(kernel_quant, &model_quant) {
        anyhow::bail!(
            "Kernel/model QUANT MISMATCH. Kernel target: {} (quant={kernel_quant}). \
             Model declares quant={model_quant} ({}). \
             The compiled kernel set has no known dispatch path for \
             quant '{model_quant}' — loading would produce silent garbage. \
             Rebuild with ATLAS_TARGET_QUANT={model_quant} (or =* to bundle multiple \
             variants) and restart.",
            ptx_set.target,
            describe_quant_source(&config),
        );
    }
    tracing::info!(
        "Selected kernel target: {} ({} modules) — quant compat: kernel={kernel_quant} \
         model={model_quant} OK",
        ptx_set.target,
        ptx_set.modules.len(),
    );

    // Text-only kernel target + a checkpoint that ships a vision tower: honor the
    // TARGET spec and serve text-only rather than failing the build at
    // `vision_encoder module not loaded`. Some VL checkpoints (e.g.
    // Kbenkhaled/Qwen3.5-27B-NVFP4) carry a `vision_config`, but their Atlas
    // kernel target (qwen3.5-27b) ships no `vision_encoder` PTX module. Drop the
    // vision tower to text-only; image inputs are unsupported until the target
    // is rebuilt with vision.
    if config.vision.is_some()
        && !ptx_set
            .modules
            .iter()
            .any(|(name, _)| *name == "vision_encoder")
    {
        tracing::warn!(
            "Checkpoint declares a vision tower but kernel target {} ships no \
             vision_encoder module — serving TEXT-ONLY (image inputs ignored). \
             Rebuild the target with vision to enable images.",
            ptx_set.target,
        );
        config.vision = None;
    }

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

    // MTP throughput-aware gate is applied at RUNTIME, not here. The earlier
    // static "FP8 ⇒ MTP off" weight-quant heuristic was removed: hardcoding the
    // decision against the weight format wrongly bars a future FP8 checkpoint
    // where MTP would help, and it conflated weight format (a proxy) with the
    // thing that actually decides MTP economics — the per-config verify-step
    // cost relative to a plain decode step. The scheduler now MEASURES that
    // ratio over the first decode steps of serving and auto-disables MTP only
    // when it is provably net-negative (verify multiplier ≥ 1 + num_drafts).
    // See `scheduler::mtp_gate`.

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
    // 2026-06-18: the previously-documented warm-Marconi-restore × MTP
    // corruption on hybrid SSM models is RESOLVED. Verified by a greedy
    // ground-truth A/B at batch=1 (the level MTP runs at — MTP is gated to
    // `active.len() == 1` in the scheduler): a real 4-turn agentic
    // conversation (incl. tool-call turns) produced byte-identical token
    // streams with Marconi ON vs OFF (full SSM recompute), 12/12 turns. The
    // #155 lineage (decode-era block-aligned snapshots, the
    // commit_verify_state_async live-state invariant, finish-leaf
    // sync_secondary) closed the interaction. Any residual divergence seen
    // only at batch>1 is FP8 low-margin argmax tie-breaking from
    // batch-size-dependent MoE-kernel rounding (a known FP8 quality-floor
    // property present for fresh non-cached sequences too), not a Marconi
    // state-management defect — so no warning is emitted here.
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

    // Fail-fast: every kernel handle the selected --kv-cache-dtype's dispatch
    // arms need must resolve NOW — not at first dispatch after a multi-minute
    // weight load (or, worse, via a silent wrong-kernel fall-through).
    // Validates each distinct per-layer dtype (high-precision / boundary
    // layers can differ from the base dtype).
    {
        let mut distinct: Vec<spark_runtime::kv_cache::KvCacheDtype> = vec![kv_dtype];
        for d in &layer_dtypes {
            if !distinct.contains(d) {
                distinct.push(*d);
            }
        }
        for d in distinct {
            spark_model::layers::qwen3_attention::validate_required_kv_kernels(
                gpu.as_ref(),
                d,
                config.head_dim,
            )
            .context("kv-cache kernel preflight failed")?;
        }
    }
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

    // Kernel load audit: print the table of every kernel resolved during model
    // construction (grouped by module/operation family) + flag any MISSING
    // (handle 0 → silent slower-fallback dispatch). Catches build/codegen
    // regressions like a dropped pipelined GEMM at load time, not as a
    // mystery slowdown.
    tracing::info!(
        "{}",
        spark_runtime::kernel_audit::render_kernel_table(
            &ptx_set.modules,
            atlas_kernels::KERNEL_SET_HASH,
        )
    );

    // Phase 6.3 — HSS config built early so the EP worker can install it.
    let early_high_speed_swap_cfg = serve_phases::build_high_speed_swap_config(&args)?;

    // EP worker: rank > 0 enters command loop, returns when head exits.
    let mut model_opt = Some(model);
    if serve_phases::maybe_run_ep_worker(&args, &mut model_opt, &early_high_speed_swap_cfg)? {
        return Ok(());
    }
    let model = model_opt.expect("head retains model on rank 0");

    // TQ+ InnerQ: opt-in via `TURBO_INNERQ=N` (N = calibration token count).
    // Once enabled, the kernel-side apply pass starts accumulating K² stats
    // and the scheduler polls `maybe_finalize` per prefill chunk; once N
    // tokens have flowed through, scales activate and stay live for the
    // process lifetime. CUDA-only: the driver talks to the CUDA Driver API
    // directly via `atlas_core::registry`, which doesn't exist on metal.
    #[cfg(feature = "cuda")]
    if let Some(driver) = spark_model::layers::qwen3_attention::InnerQDriver::from_env() {
        match driver.start() {
            Ok(()) => {
                let _ = spark_model::layers::qwen3_attention::INNERQ.set(driver);
            }
            Err(e) => {
                tracing::warn!("InnerQ calibration disabled: start() failed: {e:#}");
            }
        }
    }

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

    // (AM1 attractor-mask registration removed 2026-06-03 — see
    // decode_logits_seq.rs / compile_tools.rs; `lean` was an Atlas-only
    // decode artifact, now fixed at the grammar `first_char` rule.)

    // Tokenizer-derived runtime: vocab cap, reasoning parser, think tokens,
    // im_start hard-stop, tool-call open/close tokens, and the XGrammar
    // engine.
    let serve_phases::TokenizerRuntime {
        reasoning_parser_box,
        think_end_token,
        think_start_token,
        code_fence_token,
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
    // EP gate. v1 single-sequence worker protocol required max_batch_size=1
    // because each cmd targeted one slot and the head's per-token broadcast
    // loop had no way to address slot N. v2 adds a per-cmd seq_id preamble
    // (set ATLAS_EP_PROTOCOL=v2) so the worker routes commands by slot_idx
    // and runs decode() per-seq. The head's decode_batch_dispatch EP branch
    // stages each seq's logits row to host between decode() calls so all N
    // rows survive into process_decode_logits — without that, the single-row
    // logits buffer overwrites and N>1 produces garbage.
    let max_batch_size = if world_size > 1 {
        if scheduler_model.ep_protocol_v2() {
            tracing::info!(
                "EP v2 active: honoring max_batch_size={}",
                args.max_batch_size,
            );
            args.max_batch_size
        } else {
            tracing::info!("EP v1 active: forcing max_batch_size=1");
            1
        }
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
            code_fence_token,
            tool_call_start_token,
            tool_call_end_token,
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
            if let Some(cli_disable) = args.disable_tool_grammar {
                b.disable_tool_grammar = cli_disable;
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

/// QV1 (2026-05-26): canonicalize the model's declared quantization to
/// one of `"fp8"`, `"nvfp4"`, `"bf16"`, or `"unknown"`. Reads
/// `quantization_config.quant_method`/`quant_algo`/`format` and applies
/// the heuristics needed across ModelOpt + compressed-tensors checkpoints.
/// Returns `"bf16"` when no quant config is present (the HF default for
/// unquantized BF16 weights).
fn canonicalize_model_quant(config: &atlas_core::config::ModelConfig) -> String {
    let Some(qc) = config.quantization_config.as_ref() else {
        return "bf16".to_string();
    };
    let method = qc.quant_method.to_ascii_lowercase();
    let algo = qc.quant_algo.to_ascii_lowercase();
    let fmt = qc.format.to_ascii_lowercase();
    // NVFP4 detection — explicit algo OR a format string containing "nvfp4"
    // (compressed-tensors: "nvfp4-pack-quantized" et al).
    //
    // ModelOpt "MIXED_PRECISION" (e.g. Nemotron-Super-120B-A12B-NVFP4,
    // Qwen3.6-35B-A3B-NVFP4) canonicalizes to "nvfp4": it is nvfp4-base
    // plus a few FP8 modules. Dispatch is per-MODULE and tensor-aware, NOT
    // by this string — the loader probes `*.weight_scale` presence and
    // dequants FP8→BF16 (weight_loader/nemotron.rs:78-108, quant_helpers.rs
    // dense_auto), and the lm_head MIXED_PRECISION path is already handled
    // (factory/build.rs:144). The nvfp4 kernel bundle also carries native
    // FP8/BF16 paths (see quant_pair_compatible: nvfp4↔fp8, nvfp4↔bf16).
    // So routing MIXED_PRECISION to the nvfp4 bundle is correct and cannot
    // silently mis-route an FP8 module (it would fault at load, not corrupt).
    if algo == "nvfp4" || algo == "mixed_precision" || fmt.contains("nvfp4") {
        return "nvfp4".into();
    }
    // FP8 detection — explicit algo OR method/format containing "fp8".
    if algo == "fp8" || method.contains("fp8") || fmt.contains("fp8") {
        return "fp8".into();
    }
    // compressed-tensors with no FP8/NVFP4 marker is usually GPTQ/AWQ —
    // we don't currently dispatch those on Atlas; report verbatim so
    // the bail message is precise.
    if !algo.is_empty() {
        return algo;
    }
    if !method.is_empty() {
        return method;
    }
    "unknown".into()
}

/// QV1 helper: short debug string of where the quant declaration came
/// from, used in the bail message so the operator can locate the
/// mis-declared field quickly.
fn describe_quant_source(config: &atlas_core::config::ModelConfig) -> String {
    match config.quantization_config.as_ref() {
        Some(qc) => format!(
            "quant_method={:?}, quant_algo={:?}, format={:?}",
            qc.quant_method, qc.quant_algo, qc.format
        ),
        None => "no quantization_config in config.json".into(),
    }
}

/// QV1: returns `true` iff the kernel target's declared quant string is
/// known to handle the model's canonicalized quant.
///
/// The current Atlas build emits one bundle per (hw, model) regardless
/// of how many quant variants it dispatches at runtime: the bundle
/// label is whichever `ATLAS_TARGET_QUANT` value the build script
/// happened to record first (today: always `"nvfp4"`). Each bundle
/// nonetheless contains native FP8 / native NVFP4 / BF16-dequant code
/// paths for the same model. This compat table makes that explicit.
///
/// When new quants appear (e.g. FP4 E2M1 on a future SM), add the new
/// entry here AND the dispatch path in the weight loader. The
/// canonical home for this list will eventually be MODEL.toml
/// `[kernel].supported_quants` — until then, hardcode keeps the
/// fail-fast working without a build-time plumb-through.
fn quant_pair_compatible(kernel_quant: &str, model_quant: &str) -> bool {
    if kernel_quant == model_quant {
        return true;
    }
    matches!(
        (kernel_quant, model_quant),
        // The NVFP4-labeled bundle today carries native FP8 paths
        // (FP8 fused MoE batch1/2/3, w8a16_gemv decode, FP8 prefill).
        ("nvfp4", "fp8") |
        // The NVFP4 bundle also handles unquantized BF16 inputs via
        // runtime dequant → quantize. Slow but correct.
        ("nvfp4", "bf16") |
        // BF16 reference bundle handles any quant by dequant on load.
        ("bf16", "fp8") |
        ("bf16", "nvfp4")
    )
}

#[cfg(test)]
mod qv1_tests {
    use super::*;

    // canonicalize_model_quant is exercised via integration through
    // the server boot path; unit-testing it requires building
    // ModelConfig which has no `Default` impl (it's intentionally
    // bound to a loaded model). The pair-compatibility table is a
    // pure function and worth a unit test.

    #[test]
    fn compat_self_pair() {
        assert!(quant_pair_compatible("nvfp4", "nvfp4"));
        assert!(quant_pair_compatible("fp8", "fp8"));
        assert!(quant_pair_compatible("bf16", "bf16"));
    }

    #[test]
    fn compat_nvfp4_handles_fp8_and_bf16() {
        assert!(quant_pair_compatible("nvfp4", "fp8"));
        assert!(quant_pair_compatible("nvfp4", "bf16"));
    }

    #[test]
    fn incompat_unknown_rejected() {
        assert!(!quant_pair_compatible("nvfp4", "gptq-4bit"));
        assert!(!quant_pair_compatible("fp8", "nvfp4"));
    }
}
