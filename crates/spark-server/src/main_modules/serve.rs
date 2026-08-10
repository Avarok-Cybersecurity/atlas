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

use crate::cli;
use crate::main_modules::AppState;

/// What the blocking startup hands to the async tail.
///
/// A struct rather than a tuple since the scheduler handle joined it: five
/// positional fields where two are `Arc`s and two are addresses is a swap
/// waiting to happen at the call site.
pub(crate) struct Prepared {
    pub state: Arc<AppState>,
    pub bind: String,
    pub port: u16,
    /// The scheduler thread. A swap joins it after the drain — that join is
    /// what proves the model is no longer in use and safe to tear down.
    pub scheduler: std::thread::JoinHandle<()>,
}

/// How startup ended — three genuinely different outcomes, which an
/// `Option<Prepared>` conflated: `None` used to mean "EP worker, no router",
/// and reusing it for "no model yet" would exit the process instead of leaving
/// the dashboard up.
enum Startup {
    /// A model is loaded; serve it.
    Serve(Prepared),
    /// EP worker rank: no router here, and the head owns the lifetime.
    Worker,
    /// No model was named. The dashboard is the front door: the process stays
    /// up so the Library can be browsed, and exits when the user asks it to.
    AwaitingModel,
}

/// Bring the engine up, then serve.
///
/// Startup — weight load, KV allocation, kernel audit, graph capture — is ~50s of
/// SYNCHRONOUS CPU/IO/CUDA work containing not one `await`. Running it in the body
/// of this future would mean whatever polls the future is blocked for that whole
/// time: an `async fn` that never yields is a blocking call wearing `async`, and
/// it is why `q`/Ctrl+C appeared dead during a model load — nothing, not even the
/// signal listener, could make progress until loading finished.
///
/// So startup runs on the blocking pool and is AWAITED here. That await is a real
/// yield point, which is what lets `main` race this future against a shutdown
/// channel, and it keeps the async workers free regardless of how the runtime is
/// sized.
pub(crate) async fn serve(
    args: cli::ServeArgs,
    tui_progress: Option<std::sync::mpsc::Receiver<crate::tui::capture_layer::ProgressEvent>>,
) -> Result<()> {
    // One host for the process lifetime, created BEFORE startup so the
    // dashboard can hold it and trigger a swap. The load publishes into it.
    let host = Arc::new(crate::main_modules::model_host::ModelHost::empty());
    // Recorded before `args` moves into startup: the FIRST swap restores to
    // this if its load fails, and without it the first swap is the one swap
    // with no safety net.
    host.set_args(args.clone());
    // Captured before `args` moves into startup. The listener binds once for
    // the process lifetime, so a model chosen later serves on THIS address
    // whatever its recipe says — see the port check in `model_swap`.
    let (bind_addr, bind_port) = (args.bind.clone(), args.port);
    // Before any load: the policy must be in force from the moment the listener
    // is up, including while no model is loaded.
    host.set_auth(build_auth_config(&args)?);
    // Likewise process-scoped, and in force before the first model exists.
    host.set_process(super::serve_load::Carried::from_env());
    let startup_host = host.clone();
    match tokio::task::spawn_blocking(move || startup(args, tui_progress, startup_host)).await?? {
        Startup::Serve(prepared) => {
            host.publish(prepared.state);
            // The first load's scheduler belongs to the host too, or the first
            // swap would have nothing to join and would tear down a model with
            // a live scheduler still holding its weights.
            host.set_scheduler(prepared.scheduler);
            crate::main_modules::serve_router::build_and_serve(host, &prepared.bind, prepared.port)
                .await
        }
        Startup::Worker => Ok(()),
        // Nothing to serve YET — but the listener still comes up. Waiting for
        // shutdown instead would mean a model chosen from the Library loads
        // into a process that never binds a port, so it reaches "serving" with
        // nothing to serve it. The socket is bound once for the process
        // lifetime; until a model is published every route answers 503
        // `model_not_loaded`, which is the same shape a client already handles
        // during startup.
        Startup::AwaitingModel => {
            crate::main_modules::serve_router::build_and_serve(host, &bind_addr, bind_port).await
        }
    }
}

fn startup(
    args: cli::ServeArgs,
    tui_progress: Option<std::sync::mpsc::Receiver<crate::tui::capture_layer::ProgressEvent>>,
    host: Arc<crate::main_modules::model_host::ModelHost>,
) -> Result<Startup> {
    tracing::info!("Atlas Spark starting...");
    tracing::info!("Licensed under AGPL-3.0-only — see /LICENSE in this container");
    // Before anything writes: a nearly-full disk shows up later as a download
    // that dies mid-shard or as page-cache thrashing that reads like a
    // regression. One line now is cheaper than either diagnosis.
    crate::disk_guard::warn_if_nearly_full(args.cache_dir.as_deref());
    spark_runtime::progress::phase(0, "banner");

    // Clean shutdown: SIGINT/SIGTERM now request a drain-and-exit instead of
    // killing the process mid-write. In TUI mode Ctrl+C additionally arrives
    // as a key event (raw mode) and calls the same request().
    crate::tui::shutdown::install_signal_listeners();

    // Publishes each run's levers to the dashboard (see `tui::start`). `None`
    // in plain mode / on a worker rank, where nothing consumes them.
    let mut tui_handles_tx: Option<std::sync::mpsc::Sender<crate::tui::RunHandles>> = None;

    // Start the dashboard thread as early as possible so the operator watches
    // the load, not a blank screen. Everything it reads is process-global
    // (log ring, progress channel, metrics, scheduler snapshot) plus this
    // args snapshot for the badge chips. Head node only.
    if let Some(progress_rx) = tui_progress
        && args.rank == 0
    {
        let tx = crate::tui::start(args.clone(), progress_rx, host.clone());
        // Every later load republishes through this, so the dashboard follows
        // the model that is actually serving.
        host.set_tui_handles(tx.clone());
        tui_handles_tx = Some(tx);
    }

    // Reject contradictory flag combinations up front (issue #288) — before the
    // multi-minute model load — with a message that tells humans and AI agents
    // exactly what to change. Hard error, never a warning.
    if let Err(msg) = cli::validate_serve_args(&args) {
        anyhow::bail!("{msg}");
    }

    // Publish the kernel-path flags the command line owns, BEFORE anything can
    // read them. Each of these used to be an `ATLAS_*` variable read at its own
    // call site; they are configuration, so they belong on the command line
    // where `--help` lists them, `ps` shows them, and a recipe can be read
    // without a ten-line env preamble. The environment stays honoured as a
    // fallback for scripts that predate the flags.
    super::serve_flags::publish_kernel_flags(&args);

    // No model named: the dashboard is the front door. Everything above this
    // point is process-scoped — banner, signal listeners, the TUI thread, flag
    // validation — and everything below is model-dependent, which is exactly
    // the boundary a swap re-runs from.
    if args.model.is_none() && args.model_from_path.is_none() {
        // A dashboard is what makes a modelless boot useful. Without one there
        // is nothing to pick a model WITH, so this is a hard error on stderr
        // rather than a server that sits forever answering nothing — the shape
        // that looks healthy to a supervisor and serves no one.
        if tui_handles_tx.is_none() {
            anyhow::bail!(
                "no model given, and no dashboard to choose one from.\n\
                 Pass a MODEL (or --model-from-path), or run on a TTY without \
                 --no-tui to browse the Library."
            );
        }
        tracing::info!("No model specified — open the Library to choose one");
        return Ok(Startup::AwaitingModel);
    }

    // `None` = EP worker rank: it ran its command loop and the head has exited.
    let carried = host
        .process()
        .expect("process-scoped state is installed before startup");
    match super::serve_load::load_model(args, tui_handles_tx, carried)? {
        Some(prepared) => Ok(Startup::Serve(prepared)),
        None => Ok(Startup::Worker),
    }
}

/// Parse the vLLM-style `--default-chat-template-kwargs` JSON
/// (`{"enable_thinking":bool,"thinking_budget":u32}`) into the neutral
/// thinking directive. The CLI is its own edge: the JSON shape is parsed
/// here, not in the openai wire module. Mapping matches the
/// `chat_template_kwargs` rung of the request-body ladder: an explicit
/// budget wins, then the enable flag; unknown/empty JSON → Unspecified
/// (with a warning — the old code ignored bad JSON silently).
pub(super) fn parse_default_thinking(s: &str) -> crate::ir::ThinkingDirective {
    use crate::ir::ThinkingDirective;

    #[derive(serde::Deserialize)]
    struct Kwargs {
        enable_thinking: Option<bool>,
        thinking_budget: Option<u32>,
    }

    if s.trim().is_empty() {
        return ThinkingDirective::Unspecified;
    }
    let kw: Kwargs = match serde_json::from_str(s) {
        Ok(kw) => kw,
        Err(e) => {
            tracing::warn!("--default-chat-template-kwargs is not valid JSON ({e}); ignoring");
            return ThinkingDirective::Unspecified;
        }
    };
    match (kw.thinking_budget, kw.enable_thinking) {
        (Some(b), _) if b > 0 => ThinkingDirective::On { budget: Some(b) },
        (Some(_), _) => ThinkingDirective::Off,
        (None, Some(true)) => ThinkingDirective::On { budget: None },
        (None, Some(false)) => ThinkingDirective::Off,
        (None, None) => ThinkingDirective::Unspecified,
    }
}

/// Resolve `--require-auth` / `--auth-tokens-file` / `--auth-token` into an
/// optional `AuthConfig`. Validates at startup so misconfigurations fail
/// loudly instead of letting an unauthenticated server run silently.
pub(super) fn build_auth_config(
    args: &cli::ServeArgs,
) -> Result<Option<Arc<crate::auth::AuthConfig>>> {
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

pub(super) fn resolve_vision_max_pixels(args: &cli::ServeArgs) -> Result<Option<usize>> {
    if args.vision_max_pixels > 0 {
        return Ok(Some(args.vision_max_pixels));
    }
    let Some(raw) = std::env::var("ATLAS_VISION_MAX_PIXELS").ok() else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed == "0" {
        return Ok(None);
    }
    let parsed = trimmed.parse::<usize>().with_context(|| {
        format!("ATLAS_VISION_MAX_PIXELS must be a positive integer, got {raw:?}")
    })?;
    Ok((parsed > 0).then_some(parsed))
}

/// QV1 (2026-05-26): canonicalize the model's declared quantization to
/// one of `"fp8"`, `"nvfp4"`, `"bf16"`, or `"unknown"`. Reads
/// `quantization_config.quant_method`/`quant_algo`/`format` and applies
/// the heuristics needed across ModelOpt + compressed-tensors checkpoints.
/// Returns `"bf16"` when no quant config is present (the HF default for
/// unquantized BF16 weights).
pub(super) fn canonicalize_model_quant(config: &atlas_core::config::ModelConfig) -> String {
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
    // FP8 detection — explicit algo OR method/format containing "fp8", OR
    // compressed-tensors' `float-quantized` block-FP8 (e.g.
    // Hcompany/Holo-3.1-*-FP8: `quant_method="compressed-tensors"`,
    // `format="float-quantized"`, num_bits=8). That format string contains no
    // literal "fp8", so match it explicitly. Canonicalizing to "fp8" lets the
    // nvfp4 kernel bundle accept it (quant_pair_compatible: nvfp4↔fp8) — the
    // loader detects the FP8E4M3 weight dtype as Fp8Dequanted and requants
    // FP8→BF16→NVFP4 from the 2D `.weight_scale` (nvfp4_detect.rs).
    if algo == "fp8" || method.contains("fp8") || fmt.contains("fp8") || fmt.contains("float-quant")
    {
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
pub(super) fn describe_quant_source(config: &atlas_core::config::ModelConfig) -> String {
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
pub(super) fn quant_pair_compatible(kernel_quant: &str, model_quant: &str) -> bool {
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
