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
    // `map_err` rather than `?` on a `String`: the message is already a
    // formatted what/why/fix block and `anyhow!` keeps it verbatim.
    host.set_process(super::serve_load::Carried::from_env().map_err(|e| anyhow::anyhow!("{e}"))?);
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

/// Parsed `--default-chat-template-kwargs`: the server-level defaults
/// applied when the client request carries no thinking parameters.
#[derive(Debug, Default, PartialEq)]
pub(super) struct DefaultChatTemplateKwargs {
    /// Neutral thinking directive (budget side). `Unspecified` when the
    /// flag sets none of the thinking keys.
    pub thinking: crate::ir::ThinkingDirective,
    /// Template-side default effort string, injected in
    /// `api/chat/prepare.rs` when the request carries none and thinking
    /// is on. `None` = the cross-template `"medium"` fallback in
    /// `tokenizer/chat_render.rs` applies.
    pub reasoning_effort: Option<crate::ir::ReasoningEffort>,
    /// Server-level `preserve_thinking` pin; overrides the MODEL.toml
    /// `[behavior]` value, is overridden per-request.
    pub preserve_thinking: Option<bool>,
}

/// Parse the vLLM-style `--default-chat-template-kwargs` JSON
/// (`{"enable_thinking":bool,"thinking_budget":u32,
/// "reasoning_effort":str,"preserve_thinking":bool}`). The CLI is its
/// own edge: the JSON shape is parsed here, not in the openai wire
/// module. Thinking-directive mapping matches the `chat_template_kwargs`
/// rung of the request-body ladder: an explicit budget wins, then the
/// enable flag, then the effort ladder.
///
/// FAIL-FAST (PCND, changed 2026-08-15): invalid JSON, an unknown KEY,
/// or an unknown `reasoning_effort` VALUE abort startup instead of being
/// warned about and ignored — a typo'd operator default must never boot
/// a server that silently serves a different tier.
pub(super) fn parse_default_chat_template_kwargs(
    s: &str,
) -> anyhow::Result<DefaultChatTemplateKwargs> {
    use crate::ir::ThinkingDirective;

    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Kwargs {
        enable_thinking: Option<bool>,
        thinking_budget: Option<u32>,
        reasoning_effort: Option<String>,
        preserve_thinking: Option<bool>,
    }

    if s.trim().is_empty() {
        return Ok(DefaultChatTemplateKwargs::default());
    }
    let kw: Kwargs = serde_json::from_str(s)
        .map_err(|e| anyhow::anyhow!("--default-chat-template-kwargs is not valid JSON: {e}"))?;
    // One vocabulary (ir::parse_wire_effort) for CLI and request body, so
    // the operator default can never mean something a request couldn't.
    let (reasoning_effort, effort_directive) = match kw.reasoning_effort.as_deref() {
        None => (None, None),
        Some(v) => match crate::ir::parse_wire_effort(v) {
            Some((template_effort, directive)) => (template_effort, Some(directive)),
            None => anyhow::bail!(
                "--default-chat-template-kwargs reasoning_effort {v:?}: expected one of \
                 none, minimal, low, medium, high, xhigh, max"
            ),
        },
    };
    let thinking = match (kw.thinking_budget, kw.enable_thinking) {
        (Some(b), _) if b > 0 => ThinkingDirective::On { budget: Some(b) },
        (Some(_), _) => ThinkingDirective::Off,
        (None, Some(true)) => ThinkingDirective::On { budget: None },
        (None, Some(false)) => ThinkingDirective::Off,
        // No explicit thinking keys: an effort default carries the same
        // directive a client-sent effort would, so the budget rung and
        // the template directive stay in lockstep for effort-silent
        // requests.
        (None, None) => effort_directive.unwrap_or(ThinkingDirective::Unspecified),
    };
    Ok(DefaultChatTemplateKwargs {
        thinking,
        reasoning_effort,
        preserve_thinking: kw.preserve_thinking,
    })
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

/// Resolve the vision area bound: operator override, else the checkpoint's
/// own `preprocessor_config.json`, else `None` (preprocessor falls back to
/// its historical 1280px long-side clamp).
///
/// The checkpoint half is the point. `preprocessor_config.json` was in the
/// download plan but nothing ever parsed it, so a model declaring
/// `size = {longest_edge: 16777216}` — 4096² of permitted area — was still
/// clamped to 1280 on the long side, about a tenth of that. The operator flag
/// existed but only ever LOWERED the cap, so there was no way to serve a
/// checkpoint at the resolution it was built for.
pub(super) fn resolve_vision_max_pixels(
    args: &cli::ServeArgs,
    model_dir: &std::path::Path,
) -> Result<Option<usize>> {
    if args.vision_max_pixels > 0 {
        return Ok(Some(args.vision_max_pixels));
    }
    if let Ok(raw) = std::env::var("ATLAS_VISION_MAX_PIXELS") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() && trimmed != "0" {
            let parsed = trimmed.parse::<usize>().with_context(|| {
                format!("ATLAS_VISION_MAX_PIXELS must be a positive integer, got {raw:?}")
            })?;
            if parsed > 0 {
                return Ok(Some(parsed));
            }
        }
    }
    if let Some(px) = read_preprocessor_max_pixels(model_dir) {
        return Ok(Some(px));
    }
    // Token-budget spelling, tried LAST: a checkpoint that states an area
    // states it exactly, while a token budget has to be converted and clamped.
    Ok(read_preprocessor_max_tokens_as_pixels(model_dir))
}

/// The largest pre-merge patch count the ViT scratch is allocated for
/// (`spark-model`'s `enc_impl/init.rs` `CEILING_MAX_PATCHES`). Duplicated as a
/// literal because `spark-server` must not depend on the encoder's internals
/// just to clamp a config number; a drift here shows up as the WARN below
/// naming a ceiling that no longer matches, not as a silent overrun.
const VISION_CEILING_MAX_PATCHES: usize = 16_384;

/// GLM-5.3 declares its image budget in TOKENS, not pixels: its
/// `processor_config.json[image_processor]` carries `max_image_tokens`,
/// `patch_size` and `merge_size` and no `size`/`max_pixels` at all. Without
/// this arm `resolve_vision_max_pixels` returns `None` for that checkpoint and
/// every image silently takes the historical 1280px long-side clamp.
///
/// `px = max_image_tokens * (patch_size * merge_size)^2`, then clamped to
/// `CEILING_MAX_PATCHES * patch_size^2`. The clamp is the point: GLM declares
/// 8000 tokens = 32000 pre-merge patches, exactly 2x what the encoder's
/// scratch is sized for, so the FULL declared budget is not serveable. Clamping
/// here turns that into a downscaled image; not clamping turns it into a failed
/// H2D copy deep inside the scheduler.
///
/// ⚠ Uses the same explicit three-entry `SOURCES` table as
/// `read_preprocessor_max_pixels`, and for the same reason: GLM's
/// `video_processor.max_image_tokens` is **240000**, thirty times the image
/// budget. A recursive search for the first `max_image_tokens` in the document
/// would pick that one up on some checkpoints and over-admit every still image.
pub(super) fn read_preprocessor_max_tokens_as_pixels(model_dir: &std::path::Path) -> Option<usize> {
    for (file, nest) in PROCESSOR_SOURCES {
        let Some((scope, path)) = processor_scope(model_dir, file, nest) else {
            continue;
        };
        let tokens = scope
            .get("max_image_tokens")
            .and_then(serde_json::Value::as_u64)
            .filter(|&t| t > 0)? as usize;
        // Both default to GLM-5.3's declared geometry, which is also Qwen's.
        let patch = scope
            .get("patch_size")
            .and_then(serde_json::Value::as_u64)
            .filter(|&p| p > 0)
            .unwrap_or(14) as usize;
        let merge = scope
            .get("merge_size")
            .and_then(serde_json::Value::as_u64)
            .filter(|&m| m > 0)
            .unwrap_or(2) as usize;
        let px = tokens
            .saturating_mul(patch * merge)
            .saturating_mul(patch * merge);
        let ceiling = VISION_CEILING_MAX_PATCHES * patch * patch;
        let resolved = if px > ceiling {
            tracing::warn!(
                "Vision token budget {} tokens = {} px exceeds the encoder ceiling of {}                  patches = {} px — clamping. The checkpoint's full declared budget is NOT                  serveable; pass --vision-max-pixels explicitly to pin the operating point.",
                tokens,
                px,
                VISION_CEILING_MAX_PATCHES,
                ceiling,
            );
            ceiling
        } else {
            px
        };
        tracing::info!(
            "Vision area bound {} px derived from max_image_tokens={} (patch {}, merge {}) in              {}{}",
            resolved,
            tokens,
            patch,
            merge,
            path.display(),
            nest.map(|k| format!(" [{k}]")).unwrap_or_default(),
        );
        return Some(resolved);
    }
    None
}

/// `(image_mean, image_std, min_image_tokens, max_image_tokens)`, exactly the
/// four `VisionConfig` fields the processor config supplies.
pub(super) type ImageStats = (
    Option<[f32; 3]>,
    Option<[f32; 3]>,
    Option<usize>,
    Option<usize>,
);

/// Per-channel normalisation statistics plus the token-budget bounds, read off
/// the same processor config and over the same explicit source table.
///
/// 🔴 The mean/std pair is the single most dangerous number in the vision path.
/// GLM-5.3 normalises with CLIP statistics while Atlas's preprocessor has
/// SigLIP `[0.5; 3]` hard-coded; feeding an image through the wrong pair yields
/// a confident, fluent, WRONG description with nothing logged. Returned as an
/// all-or-nothing pair and only when both are three finite values with no zero
/// in the std (a zero divides in the patch loop), so a malformed config
/// degrades to today's hard-coded behaviour rather than to garbage.
pub(super) fn read_preprocessor_image_stats(model_dir: &std::path::Path) -> ImageStats {
    for (file, nest) in PROCESSOR_SOURCES {
        let Some((scope, path)) = processor_scope(model_dir, file, nest) else {
            continue;
        };
        let triple = |key: &str| -> Option<[f32; 3]> {
            let arr = scope.get(key)?.as_array()?;
            if arr.len() != 3 {
                return None;
            }
            let mut out = [0.0f32; 3];
            for (i, v) in arr.iter().enumerate() {
                let f = v.as_f64()?;
                if !f.is_finite() {
                    return None;
                }
                out[i] = f as f32;
            }
            Some(out)
        };
        let mean = triple("image_mean");
        let std = triple("image_std").filter(|s| s.iter().all(|c| *c != 0.0));
        let (mean, std) = match (mean, std) {
            (Some(m), Some(s)) => (Some(m), Some(s)),
            // All-or-nothing: half a pair is worse than none, because the
            // preprocessor would mix one family's mean with the other's std.
            _ => (None, None),
        };
        let min_tokens = scope
            .get("min_image_tokens")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize);
        let max_tokens = scope
            .get("max_image_tokens")
            .and_then(serde_json::Value::as_u64)
            .filter(|&v| v > 0)
            .map(|v| v as usize);
        if mean.is_none() && min_tokens.is_none() && max_tokens.is_none() {
            continue;
        }
        tracing::info!(
            "Vision preprocessing stats from {}{}: mean={:?} std={:?} tokens={:?}..{:?}",
            path.display(),
            nest.map(|k| format!(" [{k}]")).unwrap_or_default(),
            mean,
            std,
            min_tokens,
            max_tokens,
        );
        return (mean, std, min_tokens, max_tokens);
    }
    (None, None, None, None)
}

/// Ordered by precedence. `preprocessor_config.json` is the dedicated
/// image-processor file, so it wins if a checkpoint somehow ships both.
const PROCESSOR_SOURCES: [(&str, Option<&str>); 3] = [
    ("preprocessor_config.json", None),
    ("preprocessor_config.json", Some("image_processor")),
    ("processor_config.json", Some("image_processor")),
];

/// Open one `SOURCES` entry and return the addressed object plus its path.
/// `None` on absence or malformation — a checkpoint we cannot read must keep
/// the historical behaviour rather than fail to serve.
fn processor_scope(
    model_dir: &std::path::Path,
    file: &str,
    nest: Option<&str>,
) -> Option<(serde_json::Value, std::path::PathBuf)> {
    let path = model_dir.join(file);
    let text = std::fs::read_to_string(&path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&text).ok()?;
    let scope = match nest {
        Some(key) => json.get(key)?.clone(),
        None => json,
    };
    Some((scope, path))
}

/// Pull the IMAGE area bound out of the checkpoint's processor config, if it
/// ships one. `None` on any absence or malformation — a model without the
/// file, or with one we cannot read, must keep the historical behaviour
/// rather than fail to serve.
///
/// TWO FILENAMES, because the ecosystem uses both and a checkpoint ships one
/// or the other, never both. HF's `save_pretrained` writes
/// `preprocessor_config.json` with the image fields at the top level
/// (`Qwen/Qwen3.6-35B-A3B-FP8`), while a combined processor writes
/// `processor_config.json` with each modality under its own key
/// (`unsloth/Qwen3.6-27B-NVFP4` → `image_processor.size.longest_edge`).
/// Reading only the first is why the unsloth checkpoints — the ones actually
/// being served — still ran the 1280px fallback after the area bound was
/// supposedly honoured: they declare 16777216 and nothing looked in the file
/// that says so.
///
/// The nested form is also why this cannot be a search for the first
/// `longest_edge` in the document. `processor_config.json` carries a SECOND,
/// LARGER bound under `video_processor` (25165824 vs the image 16777216);
/// picking that one would over-admit every still image by half again its
/// permitted area. The image bound is addressed explicitly.
///
/// Two spellings are accepted within whichever object we land on. HF's
/// Qwen2VL/Qwen3VL processors write `size = {longest_edge, shortest_edge}`,
/// where — despite the names — both are pixel COUNTS, not edge lengths
/// (16777216 = 4096², 65536 = 256²). Older/other processors write
/// `max_pixels` directly.
pub(super) fn read_preprocessor_max_pixels(model_dir: &std::path::Path) -> Option<usize> {
    for (file, nest) in PROCESSOR_SOURCES {
        let Some((scope, path)) = processor_scope(model_dir, file, nest) else {
            continue;
        };
        let from_size = scope
            .get("size")
            .and_then(|s| s.get("longest_edge"))
            .and_then(serde_json::Value::as_u64);
        let direct = scope.get("max_pixels").and_then(serde_json::Value::as_u64);
        let Some(px) = from_size.or(direct).filter(|&p| p > 0) else {
            continue;
        };
        tracing::info!(
            "Vision area bound {} px from {}{} (was: hard-coded 1280px long side)",
            px,
            path.display(),
            nest.map(|k| format!(" [{k}]")).unwrap_or_default(),
        );
        return Some(px as usize);
    }
    None
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
        // The NVFP4-labeled bundle carries the EXL3 (QTIP trellis) dispatch
        // too — `exl3_matmul.cu`, `exl3_moe.cu`, `exl3_reconstruct.cu` compile
        // into it, which is why the gb10 targets report 183 kernels rather than
        // 180. Two ways a checkpoint reaches it, both real:
        //   * kept PACKED and decoded in-kernel (`ATLAS_EXL3_NATIVE`, and the
        //     routed-expert arm GLM-5.3 uses), or
        //   * materialized to NVFP4/BF16 at load by `exl3_materialize`.
        // Without this pair a `quant_method: "exl3"` pack is refused before any
        // weight loads, even though every kernel it needs is present.
        ("nvfp4", "exl3") |
        // BF16 reference bundle handles any quant by dequant on load.
        ("bf16", "fp8") |
        ("bf16", "nvfp4")
    )
}

#[cfg(test)]
#[path = "serve_tests.rs"]
mod tests;
