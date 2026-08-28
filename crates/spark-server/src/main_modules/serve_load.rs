// SPDX-License-Identifier: AGPL-3.0-only

//! The model-dependent half of startup — phases 1-10.
//!
//! Split out of `serve.rs` so it can be run MORE THAN ONCE. `startup()` keeps
//! the process-scoped prelude (banner, signal listeners, the dashboard thread,
//! flag validation); everything here is derived from a specific checkpoint and
//! is what a hot-swap re-runs against a different one.
//!
//! Three things this move must not break, all of them silent failures:
//!
//! * **Plain-log byte-identity.** `spark serve <M> --no-tui` output is a grep
//!   contract. No line was added, removed or reordered — this is a pure move.
//! * **`progress::phase(i, …)` indices are POSITIONAL** against
//!   `ProgressModel::PHASE_NAMES`; renumbering desyncs the checklist with no
//!   error anywhere.
//! * **Process-scoped state must be CARRIED, not rebuilt.** `response_store`,
//!   `conversation_store` and `rate_limiter` live on `AppState` but outlive any
//!   model; re-running their constructors on a swap silently drops stored
//!   conversations and resets rate-limit buckets. See `Carried` below.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use super::serve::{
    Prepared, canonicalize_model_quant, describe_quant_source, parse_default_chat_template_kwargs,
    quant_pair_compatible, resolve_vision_max_pixels,
};
use crate::api::InferenceRequest;
use crate::main_modules::AppState;
use crate::main_modules::serve_phases;
use crate::tokenizer::ChatTokenizer;
use crate::{
    cli, conversation_store, rate_limiter, response_store, scheduler, scheduling_policy,
    session_manager,
};

/// Load a model and build everything derived from it.
///
/// State that OUTLIVES any model and must be carried across a swap.
///
/// These live on `AppState` but are not derived from the checkpoint. Rebuilding
/// them on a swap silently drops stored conversations and responses and resets
/// every rate-limit bucket — no error, just a user noticing their history gone.
/// Passing them in makes carrying them the only way to load a second model:
/// the compiler asks for them, so nobody has to remember.
#[derive(Clone)]
pub(crate) struct Carried {
    pub response_store: std::sync::Arc<response_store::ResponseStore>,
    pub rate_limiter: std::sync::Arc<rate_limiter::RateLimiter>,
    pub conversation_store: std::sync::Arc<conversation_store::ConversationStore>,
}

impl Carried {
    /// First boot: build them once, from the environment.
    ///
    /// Built here and then installed on the HOST, before the listener binds,
    /// so everything that is process-scoped is reachable while no model is
    /// loaded — and so there is exactly one of each: handlers refund through
    /// the same limiter the middleware debits, and read the same stores a swap
    /// carries forward.
    pub fn from_env() -> Self {
        Self {
            // `from_env` already hands back an Arc.
            response_store: response_store::ResponseStore::from_env(),
            rate_limiter: rate_limiter::RateLimiter::from_env(),
            conversation_store: conversation_store::ConversationStore::from_env(),
        }
    }

    /// A swap: take them from the model being replaced.
    pub fn from_previous(previous: &AppState) -> Self {
        Self {
            response_store: previous.response_store.clone(),
            rate_limiter: previous.rate_limiter.clone(),
            conversation_store: previous.conversation_store.clone(),
        }
    }
}

/// `Ok(None)` means this rank is an EP worker: it ran its command loop and has
/// nothing for the async tail to serve.
pub(crate) fn load_model(
    mut args: cli::ServeArgs,
    tui_handles_tx: Option<std::sync::mpsc::Sender<crate::tui::RunHandles>>,
    carried: Carried,
) -> Result<Option<Prepared>> {
    // 0. Resolve model directory from HF ID or path
    spark_runtime::progress::phase(1, "model resolve");
    let model_dir = serve_phases::resolve_model_dir(&args)?;

    tracing::info!("Port: {}", args.port);

    // Report what is actually in force. This line used to be the literal
    // "f32 (full precision)" unconditionally, printed before the model config
    // even loaded — so it could not have reflected the resolved state even in
    // principle, and it said f32 in every run of the campaign that ran f16.
    tracing::info!(
        "SSM decode h-state dtype: {} (--ssm-h-dtype)",
        if spark_model::layers::qwen3_ssm::ssm_h_f16_pool_enabled() {
            "f16 + f16-sized pools (stage 3)"
        } else if spark_model::layers::qwen3_ssm::ssm_h_fp16_enabled() {
            "f16"
        } else {
            "f32 (full precision)"
        }
    );

    // 1. Load model config (supports HF config.json and Mistral params.json)
    spark_runtime::progress::phase(2, "config");
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

    // Vision area bound, resolved ONCE and installed on the config before
    // anything derived from it exists.
    //
    // Ordering is load-bearing: the vision encoder sizes every device buffer
    // from this number, and it is constructed inside `build_model` below. This
    // used to be resolved AFTER the model was built, which is how the CPU
    // preprocessor and the GPU encoder came to hold two different ideas of the
    // maximum image — the preprocessor clamped to 1280px, the encoder
    // allocated for 6400 patches, and nothing connected them.
    let vision_max_pixels = resolve_vision_max_pixels(&args, &model_dir)?;
    if let Some(v) = config.vision.as_mut() {
        v.max_pixels = vision_max_pixels;
    }
    match vision_max_pixels {
        Some(px) => tracing::info!(
            "Vision area bound: {} px ({})",
            px,
            if args.vision_max_pixels > 0 {
                "--vision-max-pixels"
            } else {
                // Deliberately does not name a file: the bound comes from
                // whichever of preprocessor_config.json /
                // processor_config.json the checkpoint actually ships, and
                // `read_preprocessor_max_pixels` logs the resolved path and
                // key on its own line. Naming one here was wrong for every
                // unsloth checkpoint, which ships only the other.
                "checkpoint processor config / ATLAS_VISION_MAX_PIXELS"
            }
        ),
        None => tracing::info!(
            "Vision area bound: none declared — falling back to the 1280px long-side clamp"
        ),
    }

    // Remote image fetching. Logged at WARN when on, because it is the one
    // vision setting that changes what the server is allowed to REACH rather
    // than how it processes what it was given — an operator reading the boot
    // log should see it without looking for it.
    let remote_image_policy = crate::api::chat::remote_image::RemoteImagePolicy {
        enabled: args.vision_allow_remote_images,
        max_bytes: args.vision_remote_image_max_mb.saturating_mul(1024 * 1024),
        timeout_secs: args.vision_remote_image_timeout_s,
        allow_private: args.vision_remote_image_allow_private,
    };
    if remote_image_policy.enabled {
        tracing::warn!(
            "Remote image fetching ENABLED (--vision-allow-remote-images): this server will \
             issue outbound HTTP to URLs supplied in chat requests. Cap {} MiB, timeout {} s, \
             private/loopback/link-local destinations {}.",
            args.vision_remote_image_max_mb,
            remote_image_policy.timeout_secs,
            if remote_image_policy.allow_private {
                "ALLOWED (--vision-remote-image-allow-private)"
            } else {
                "refused"
            }
        );
    } else {
        tracing::info!(
            "Remote image fetching disabled (default); image_url parts carrying an http(s) \
             URL are refused with a 400. Send base64 data: URIs, or pass \
             --vision-allow-remote-images."
        );
    }

    // Video decoding. Probed at BOOT, not on the first request: a deployment
    // that enabled video and has no ffmpeg is misconfigured, and the operator
    // should learn that from the startup log rather than from a user's failed
    // request an hour later.
    let video_ffmpeg = spark_model::video_decode_ffmpeg::FfmpegPolicy {
        enabled: args.video_allow_ffmpeg,
        binary: args.video_ffmpeg_path.clone(),
        max_frames: args.video_max_frames,
        timeout_secs: args.video_decode_timeout_s,
        ..Default::default()
    };
    match spark_model::video_decode_ffmpeg::probe(&video_ffmpeg) {
        spark_model::video_decode_ffmpeg::Availability::Ready(v) => tracing::info!(
            "Video decoding ENABLED via {} ({}); sampling at {} fps, max {} frames",
            args.video_ffmpeg_path,
            v,
            args.video_fps,
            args.video_max_frames,
        ),
        // WARN, not a hard failure: the server still serves text and images
        // perfectly well, and refusing to boot would turn a video
        // misconfiguration into a total outage. Every video request will fail
        // with the binary named, so the condition is not silent either way.
        spark_model::video_decode_ffmpeg::Availability::Missing(why) => tracing::warn!(
            "Video decoding was ENABLED (--video-allow-ffmpeg) but the decoder is NOT \
             USABLE: {why}. Every video request will fail. Install ffmpeg (apt install \
             ffmpeg) or point --video-ffmpeg-path at it. Animated GIF still decodes \
             in-process; text and image serving are unaffected.",
        ),
        spark_model::video_decode_ffmpeg::Availability::Disabled => tracing::info!(
            "Video decoding disabled (default). Animated GIF decodes in-process; every \
             other container needs ffmpeg — pass --video-allow-ffmpeg to enable it."
        ),
    }

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
    spark_runtime::progress::phase(3, "gpu init");
    //
    // Each kernel target declares which (model_type, hidden_size) pairs it supports
    // via [[model_types]] in MODEL.toml. Exact hidden_size matches win over
    // wildcards. Config-identical checkpoints (Qwen3.6-27B vs Qwen3.8-27B both
    // parse to (qwen3_5, 5120)) are disambiguated by matching each colliding
    // target's declared `match_names` against these checkpoint references; a
    // tie that does not break to exactly one target is a hard error here (never
    // a build-order pick), and `--kernel-target` pins the choice explicitly.
    let model_dir_str = model_dir.display().to_string();
    let model_refs: Vec<&str> = [
        args.model.as_deref(),
        args.model_name.as_deref(),
        Some(model_dir_str.as_str()),
    ]
    .into_iter()
    .flatten()
    .collect();
    let ptx_set = atlas_kernels::ptx_for_config(
        &config.model_type,
        config.hidden_size,
        &model_refs,
        args.kernel_target.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?
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
    // Record the RESOLVED target identity for the dashboard's kernel table.
    // It used to re-run resolution from (model_type, hidden_size), but that
    // shape no longer identifies a target on its own (see the tie-break
    // above) — publishing the outcome is exact and cannot disagree with it.
    crate::tui::data::kernels::publish_loaded_target(ptx_set.target.model, ptx_set.target.quant);

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

    // Resolve num_drafts: explicit --num-drafts (any value) → MODEL.toml
    // [behavior].default_num_drafts → engine default. After this call
    // `args.num_drafts` is Some and `args.resolved_num_drafts()` is valid.
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
    spark_runtime::progress::phase(4, "topology");
    let serve_phases::Topology {
        world_size,
        tp_size: _tp_size,
        ep_size,
        tp_rank: _tp_rank,
        ep_rank,
    } = serve_phases::resolve_topology(&args, &mut config)?;
    // FP8 KV calibration precedence (highest wins): an explicit
    // --fp8-kv-calibration-tokens ALWAYS wins — including 0, which
    // force-disables calibration on a model whose MODEL.toml enables it
    // (the previous `> 0` sentinel could not express that). Omitted falls
    // back to MODEL.toml [behavior].fp8_kv_calibration_tokens (0 if absent).
    config.fp8_kv_calibration_tokens = args
        .fp8_kv_calibration_tokens
        .unwrap_or(ptx_set.behavior.fp8_kv_calibration_tokens);
    // Unconditional: the serde(skip) default is 0.0, and the CLI default (2.0)
    // is the real one. Validated ≥ 1.0 in `validate_serve_args`.
    config.fp8_kv_headroom = args.fp8_kv_headroom;

    // 3. Load model weights
    spark_runtime::progress::phase(5, "weight load");
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
    spark_runtime::weights::auto_detect_weight_prefix(&store, &mut config);

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

    // 3b. Pre-warm cuBLASLt so request 1 does not pay its lazy init.
    // Measured (2026-08-22, 35B flagship, dgx1): once QKVZ prefill routes
    // through cuBLASLt, the FIRST request read ~0.9 s slower than warm ones —
    // handle create + 64 MB workspace + the library's kernel-image load, all
    // deferred to first use. Cold TTFT is a headline metric; load time is not.
    // Failure is logged inside and never fails the serve.
    #[cfg(feature = "cuda")]
    spark_runtime::cublaslt::prewarm(0);

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
    spark_runtime::progress::phase(6, "kv cache");
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
    let prefix_cache = serve_phases::build_prefix_cache(&args, &config);
    let comm = serve_phases::init_nccl_comm(
        &args,
        gpu.as_ref(),
        world_size,
        max_batch_tokens,
        config.hidden_size,
    )?;
    // Carried on the config rather than written into the environment: the old
    // `unsafe set_var` claimed "called before any threads are spawned", which
    // was false by this point (tokio pool, this blocking thread, the signal
    // listener, the TUI thread, the OOM watchdog), and a concurrent getenv
    // during setenv is UB.
    config.profile = args.profile;
    serve_phases::cap_vocab_size_to_tokenizer(&model_dir, &mut config);
    let serve_phases::KvCacheConfig {
        effective_kv_dtype_str: _,
        kv_dtype,
        layer_dtypes,
        hss_cache_blocks_per_seq,
    } = serve_phases::resolve_kv_cache_config(
        &args,
        &config,
        ptx_set.behavior.default_kv_dtype,
        store.fp8_kv_scale_count(),
    )?;

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
    // LoRA adapters: resolve + load BEFORE `gpu` is moved into build_model.
    // `lora_states` must outlive build_model (lora_args borrows &l.store) and
    // stays alive until after AppState construction (adapter name clones).
    // NLLB uses its own encoder-decoder LoRA path (resolved below), not the
    // decoder-only pool loader — skip the latter to avoid a family rejection.
    let is_nllb = matches!(config.model_type.as_str(), "m2m_100" | "nllb");
    let lora_states = if is_nllb {
        Vec::new()
    } else {
        serve_phases::load_lora_adapters(&args, gpu.as_ref())?
    };
    if !lora_states.is_empty() && world_size > 1 {
        anyhow::bail!(
            "--lora-adapter requires world_size=1 in v0 (got {world_size}); \
             TP adapter sharding is M3"
        );
    }
    let lora_args = if lora_states.is_empty() {
        None
    } else {
        Some(spark_model::factory::LoraBuildArgs {
            adapters: lora_states
                .iter()
                .map(|l| spark_model::lora::LoraAdapterInput {
                    name: l.name.clone(),
                    store: &l.store,
                    peft: l.peft_config.clone(),
                })
                .collect(),
            max_lora_rank: args.max_lora_rank,
            max_loras: args.max_loras,
        })
    };
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
    // NLLB / M2M-100: resolve the translation language pair to token ids from
    // the checkpoint tokenizer (the ChatTokenizer isn't built until after the
    // model). Only for encoder-decoder checkpoints; other models pass `None`.
    let nllb_lang: Option<(u32, u32)> = if matches!(config.model_type.as_str(), "m2m_100" | "nllb")
    {
        let src = args.src_lang.as_deref().ok_or_else(|| {
            anyhow::anyhow!("serving an NLLB/M2M-100 checkpoint requires --src-lang")
        })?;
        let tgt = args.tgt_lang.as_deref().ok_or_else(|| {
            anyhow::anyhow!("serving an NLLB/M2M-100 checkpoint requires --tgt-lang")
        })?;
        let tk = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("nllb: load tokenizer for lang resolve: {e}"))?;
        let src_id = tk
            .token_to_id(src)
            .ok_or_else(|| anyhow::anyhow!("unknown --src-lang token '{src}'"))?;
        let tgt_id = tk
            .token_to_id(tgt)
            .ok_or_else(|| anyhow::anyhow!("unknown --tgt-lang token '{tgt}'"))?;
        tracing::info!("NLLB translation: {src}({src_id}) -> {tgt}({tgt_id})");
        Some((src_id, tgt_id))
    } else {
        None
    };
    // NLLB PEFT adapter: resolve the first `--lora-adapter NAME=DIR` to a dir;
    // NllbGpuModel loads it via its own encoder-decoder LoRA apply.
    let nllb_lora_dir: Option<std::path::PathBuf> = if is_nllb {
        match args.lora_adapter.first() {
            Some((_name, spec)) => Some(
                crate::model_resolver::resolve_adapter_dir(spec, args.cache_dir.as_deref())
                    .context("resolving NLLB --lora-adapter")?,
            ),
            None => None,
        }
    } else {
        None
    };
    // Advertise the NLLB adapter name so a request's `adapter` field validates
    // (present → apply LoRA per-request, absent → base). Kept distinct from the
    // model name so base routing falls through to slot -1.
    let nllb_adapter_name: Option<String> = if nllb_lora_dir.is_some() {
        args.lora_adapter.first().map(|(n, _)| n.clone())
    } else {
        None
    };
    let model = serve_phases::build_model(
        &args,
        &config,
        // Moved, not borrowed: the model keeps the ledger so it can free the
        // weights at teardown. Nothing after this point reads the store.
        store,
        gpu,
        max_batch_tokens,
        kv_dtype,
        inference_reserve,
        layer_dtypes,
        hss_cache_blocks_per_seq,
        prefix_cache,
        comm,
        dflash_args,
        lora_args,
        nllb_lang,
        nllb_lora_dir,
    )?;

    // Kernel load audit + the fail-closed boot gate. Every lookup is eager, so
    // by here the audit holds this model's COMPLETE lookup set — see
    // `serve_phases::kernel_gate`, which owns the report, the gate and
    // `--check-kernels`.
    // Under `--check-kernels` this call does not return: it prints the report
    // and exits with the unresolved count as the process status.
    spark_runtime::progress::phase(7, "kernel audit");
    serve_phases::audit_and_gate(&args, &ptx_set)?;

    // Phase 6.3 — HSS config built early so the EP worker can install it.
    let early_high_speed_swap_cfg = serve_phases::build_high_speed_swap_config(&args)?;

    // EP worker: rank > 0 enters command loop, returns when head exits.
    let mut model_opt = Some(model);
    if serve_phases::maybe_run_ep_worker(&args, &mut model_opt, &early_high_speed_swap_cfg)? {
        // An EP worker (rank > 0) never serves HTTP: it ran its command loop and
        // the head has exited. `None` = nothing for the async tail to do.
        return Ok(None);
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
    } = serve_phases::load_sampling_defaults(&model_dir, &args, &sampling_presets.non_thinking);

    // 6. Load tokenizer
    spark_runtime::progress::phase(8, "tokenizer");
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
        args.disable_template_overrides,
    )?;

    // (AM1 attractor-mask registration removed 2026-06-03 — see
    // decode_logits_seq.rs / compile_tools.rs; `lean` was an Atlas-only
    // decode artifact, now fixed at the grammar `first_char` rule.)

    // Tokenizer-derived runtime: vocab cap, reasoning parser, think tokens,
    // im_start hard-stop, tool-call open/close tokens, and the XGrammar
    // engine.
    let serve_phases::TokenizerRuntime {
        vocab_masks,
        limits: tokenizer_limits,
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
    spark_runtime::progress::phase(9, "scheduler");
    let (request_tx, request_rx) = mpsc::channel::<InferenceRequest>(args.max_num_seqs);
    // LoRA adapter-rotation control channel (POST /v1/lora/active). Small: it
    // carries only control messages, applied one-at-a-time at quiescence.
    let (rotation_tx, rotation_rx) = mpsc::channel::<scheduler::LoraRotation>(8);

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
    // mHC highway models (#753 item B): multi-seq decode runs the per-seq
    // highway loop (decode_a2) with per-sequence PLE/QSA state; batched
    // prefill/mixed steps are serialized scheduler-side. Concurrency is
    // honored — the earlier clamp-to-1 mitigation is lifted.
    if scheduler_model.hc_mult() > 0 && max_batch_size > 1 {
        tracing::info!(
            "mHC highway model: concurrency {max_batch_size} via the per-seq \
             highway decode loop (batched highway kernels are the perf \
             follow-up)"
        );
    }
    // Derived ceiling (wave-14a): the decode-metadata layout, logits rows and
    // scratch block-table envelope are all DERIVED from max_batch_size
    // (`spark_runtime::buffers::DecodeMetaLayout`, rows = max(32, bs) —
    // byte-identical to the old fixed 32-row layout for every bs <= 32).
    // DECODE_META_MAX_ROWS is the validated policy ceiling; above it, fail
    // here at serve time instead of mid-decode (aacd29cb's safety intent).
    anyhow::ensure!(
        max_batch_size <= spark_runtime::buffers::DECODE_META_MAX_ROWS,
        "--max-batch-size {max_batch_size} exceeds the derived decode-metadata \
         ceiling of {} rows (DECODE_META_MAX_ROWS)",
        spark_runtime::buffers::DECODE_META_MAX_ROWS
    );
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
        args.resolved_num_drafts()
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
    // DFlash mode: the drafter proposes on raw argmax, so the verify steps
    // must judge acceptance on the same (GOLD) basis — skipping the
    // rep_pen/DRY pre-sample pipeline — or drafter and verifier disagree by
    // construction and accept craters. ATLAS_DFLASH_MASKED_VERIFY=1 routes
    // verify PICKS back through the pre-sample masking (unmasked
    // special-token leak fix); that is handled at the pick sites via
    // `verify_pipeline_helper::dflash_masked_verify_enabled()` and must NOT
    // flip this bool — this selects the verify architecture, not the pick
    // basis.
    let dflash_verify_raw_argmax = args.dflash;
    // DS4F hard-limit lane (2026-07-21): the served-context ceiling the
    // scheduler enforces per decode step (§C-3), not just as a KV-allocation
    // ceiling trued-up on completion. Travels with the run's other hard stops.
    // Per-model watchdog tunables. Built here, before the scheduler thread
    // spawns — the installer this replaces ran from `log_behavior_audit`,
    // which is called well after the spawn.
    let watchdog_params = crate::scheduler::WatchdogParams::from_behavior(
        &ptx_set.behavior,
        args.max_inter_tool_prose,
        args.content_loop_min_repeats,
    );
    // The run's levers. Shared with the dashboard so `/watchdog on|off`
    // toggles this run's flag; the MODEL.toml `[behavior]` value is its
    // starting position.
    let sched_levers = std::sync::Arc::new(crate::scheduler::levers::SchedLevers::from_env());
    sched_levers.set_loop_watchdog(crate::scheduler::resolve_content_loop_watchdog(
        ptx_set.behavior.enable_loop_watchdog,
        std::env::var("ATLAS_CONTENT_LOOP_WATCHDOG").ok().as_deref(),
        args.content_loop_watchdog,
    ));
    // The run's snapshot cell, shared with the dashboard for the same reason
    // and by the same route as the levers.
    let sched_snapshot = std::sync::Arc::new(crate::scheduler::snapshot::SnapshotCell::default());
    if let Some(tx) = &tui_handles_tx {
        let _ = tx.send(crate::tui::RunHandles {
            levers: sched_levers.clone(),
            snapshot: sched_snapshot.clone(),
        });
    }
    let run_levers = sched_levers.clone();
    let run_snapshot = sched_snapshot.clone();
    let sched_limits = crate::scheduler::limits::SchedLimits {
        max_seq_len: args.max_seq_len,
        ..tokenizer_limits
    };
    // Capture the runtime handle IN async context so the scheduler OS thread
    // can detach terminal stream sends (Done/Error) as tokio tasks.
    scheduler::capture_runtime_handle();
    // RETAINED, not detached. A swap has to know when the scheduler has
    // actually finished: dropping every `Arc<AppState>` closes `request_tx`,
    // the loop drains and returns, and only THEN is the model free to tear
    // down. Without the handle there is no way to wait for that, and the
    // teardown would race a scheduler still touching the weights.
    let scheduler_handle = std::thread::spawn(move || {
        scheduler::run(
            scheduler_model,
            request_rx,
            rotation_rx,
            scheduler_eos,
            max_batch_size,
            use_speculative,
            dflash_verify_raw_argmax,
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
            vocab_masks,
            sched_limits,
            watchdog_params,
            run_levers,
            run_snapshot,
        );
    });

    // Tool call parser resolution: CLI > MODEL.toml > defaults table.
    let tool_call_parser = serve_phases::resolve_tool_call_parser(&args, &ptx_set, &config)?;

    // 8. Build app state
    // Carried, never rebuilt — see `Carried`.
    let Carried {
        response_store,
        rate_limiter,
        conversation_store,
    } = carried;
    serve_phases::log_response_store_audit(&response_store, &rate_limiter);
    let dump_writer = serve_phases::open_dump_writer(&args);
    // #27: build the STAGEABLE registry (name -> {peer_stage_id, peft}). The peer
    // WeightManifest carries no r/alpha, so the peft scaling is parsed from each
    // adapter's local CONFIG_DIR/adapter_config.json HERE (fail-fast at startup —
    // a wrong/absent scale must never be discovered mid-serve).
    let lora_peer_addr = spark_model::lora::lora_peer_env();
    let mut lora_stageable = std::collections::HashMap::new();
    for (name, peer_id, dir) in &args.lora_stageable {
        let cfg_path = std::path::Path::new(dir).join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path).with_context(|| {
            format!(
                "--lora-stageable '{name}': read peft config {}",
                cfg_path.display()
            )
        })?;
        let peft = atlas_core::config::parse_peft_adapter_config(&raw)
            .with_context(|| format!("--lora-stageable '{name}': parse {}", cfg_path.display()))?;
        lora_stageable.insert(
            name.clone(),
            crate::main_modules::promotion::StageableAdapter {
                peer_stage_id: peer_id.clone(),
                peft,
            },
        );
    }
    if !lora_stageable.is_empty() && lora_peer_addr.is_none() {
        anyhow::bail!(
            "--lora-stageable given ({} adapter(s)) but $ATLAS_LORA_PEER is unset; \
             demand promotion needs a weight peer to RDMA-stage from",
            lora_stageable.len()
        );
    }
    if !lora_stageable.is_empty() && lora_states.is_empty() {
        anyhow::bail!(
            "--lora-stageable needs a resident pool to promote INTO; start with at \
             least one --lora-adapter (and --max-loras > that count for cache headroom)"
        );
    }
    // DISK-stageable registry (the no-RDMA sibling): name -> (resolved dir, peft).
    // The dir is resolved HF-id-or-path via the model_resolver; peft is parsed +
    // rank-checked at startup (fail-fast like load_lora_adapters). The disk swap
    // re-reads adapter_config.json at promote time, so this copy is only for
    // advertising + rank validation.
    let mut lora_disk_stageable = std::collections::HashMap::new();
    for (name, spec) in &args.lora_stageable_disk {
        if lora_states.iter().any(|s| &s.name == name) || lora_stageable.contains_key(name) {
            anyhow::bail!(
                "--lora-stageable-disk '{name}' collides with a resident/peer-stageable \
                 adapter name"
            );
        }
        let dir = crate::model_resolver::resolve_adapter_dir(spec, args.cache_dir.as_deref())
            .with_context(|| format!("--lora-stageable-disk '{name}': resolve '{spec}'"))?;
        let cfg_path = dir.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path).with_context(|| {
            format!(
                "--lora-stageable-disk '{name}': read peft config {}",
                cfg_path.display()
            )
        })?;
        let peft = atlas_core::config::parse_peft_adapter_config(&raw).with_context(|| {
            format!(
                "--lora-stageable-disk '{name}': parse {}",
                cfg_path.display()
            )
        })?;
        if peft.r > args.max_lora_rank {
            anyhow::bail!(
                "--lora-stageable-disk '{name}' r={} > --max-lora-rank {}",
                peft.r,
                args.max_lora_rank
            );
        }
        lora_disk_stageable.insert(name.clone(), (dir, peft));
    }
    if !lora_disk_stageable.is_empty() && lora_states.is_empty() {
        anyhow::bail!(
            "--lora-stageable-disk needs a resident pool to promote INTO; start with at \
             least one --lora-adapter and --max-loras > that count"
        );
    }
    // The disk swap re-points a cache slot only when rotation is armed
    // (decode runs eager). ATLAS_LORA_ROTATE=1 arms it; a peer being set also
    // forces eager decode, so accept either.
    if !lora_disk_stageable.is_empty()
        && !spark_model::lora::lora_rotate_env()
        && lora_peer_addr.is_none()
    {
        anyhow::bail!(
            "--lora-stageable-disk needs rotation armed: set ATLAS_LORA_ROTATE=1 so decode \
             runs eager and the disk swap can re-point a cache slot"
        );
    }
    // Promotion is armed when a LoRA pool is resident AND there is at least one
    // stageable source — a peer-backed registry (needs a peer) OR a disk-backed
    // registry. Otherwise every field is inert and a miss 404s byte-identically
    // to today. The coalescer is backing-agnostic; peer and disk misses share it.
    let promotion = if (!lora_stageable.is_empty() && lora_peer_addr.is_some()
        || !lora_disk_stageable.is_empty())
        && !lora_states.is_empty()
    {
        Some(Arc::new(
            crate::main_modules::promotion::PromotionManager::default(),
        ))
    } else {
        None
    };
    if promotion.is_some() {
        tracing::info!(
            "LoRA #27: {} peer + {} disk stageable adapter(s) armed for demand promotion \
             (peer={:?}, cache headroom={} slots)",
            lora_stageable.len(),
            lora_disk_stageable.len(),
            lora_peer_addr,
            args.max_loras.saturating_sub(lora_states.len()),
        );
    }

    // Fail-fast at startup: a typo'd operator default (unknown key or
    // unknown reasoning_effort value) must abort the boot, not warn and
    // serve a different tier.
    let default_kwargs = args
        .default_chat_template_kwargs
        .as_deref()
        .map(parse_default_chat_template_kwargs)
        .transpose()?
        .unwrap_or_default();

    let state = Arc::new(AppState {
        tokenizer,
        model_name,
        adapter_name: nllb_adapter_name
            .clone()
            .or_else(|| lora_states.first().map(|l| l.name.clone())),
        adapter_names: if let Some(ref n) = nllb_adapter_name {
            vec![n.clone()]
        } else {
            lora_states.iter().map(|l| l.name.clone()).collect()
        },
        active_adapter: std::sync::Arc::new(std::sync::Mutex::new(
            lora_states.first().map(|l| l.name.clone()),
        )),
        max_seq_len: args.max_seq_len,
        request_tx,
        rotation_tx: if lora_states.is_empty() {
            None
        } else {
            Some(rotation_tx)
        },
        chat: crate::api::chat::levers::ChatLevers::resolve(
            ptx_set.behavior.tscg,
            ptx_set.behavior.disable_cwd_hint_injection,
        ),
        vision_config: config.vision.clone(),
        vision_max_pixels,
        remote_image_policy,
        video_ffmpeg,
        video_fps: args.video_fps,
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
            // Server-level preserve_thinking pin outranks the MODEL.toml
            // [behavior] value (request-body kwargs still win in
            // api/chat/prepare.rs). Previously this key was silently
            // ignored by the CLI parser.
            if let Some(p) = default_kwargs.preserve_thinking {
                b.preserve_thinking = Some(p);
            }
            b
        },
        disable_thinking: args.disable_thinking,
        default_thinking: default_kwargs.thinking,
        default_reasoning_effort: default_kwargs.reasoning_effort,
        response_store,
        rate_limiter,
        conversation_store,
        dump_writer,
        lora_stageable,
        lora_peer_addr,
        promotion,
        promoted_slots: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
        lora_disk_stageable,
    });

    serve_phases::log_behavior_audit(&args, &ptx_set);

    // 9-11. Router + HTTP server run on the async side; hand them the pieces.
    Ok(Some(Prepared {
        state,
        bind: args.bind,
        port: args.port,
        scheduler: scheduler_handle,
    }))
}

#[cfg(test)]
#[path = "serve_load_tests.rs"]
mod tests;
