// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the serve-facing entry points of the RetroInfer port: the
//! CLI values `port_apply` publishes, their resolution (CLI, then the
//! `AVAROK_RETRO_*` environment twin, then the default), the startup
//! validation, and the registration of the sparse-seam selector when the
//! `retro_attn` row or `--retro-shadow` is on.
//!
//! Resolution of "retro attention on": `--retro-attn` when given, else the
//! `retro_attn` row (`AVAROK_RETRO_ATTN` under the toggle grammar, else the
//! target's declaration, `false` everywhere). Shadow mode (`--retro-shadow` /
//! `AVAROK_RETRO_SHADOW`) runs on its own and never changes the output. With
//! both off nothing registers and nothing here runs again.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};

use super::config::RetroParams;

/// The `--retro-*` serve flags as `port_apply` passes them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CliValues {
    pub retro_attn: bool,
    pub retrieval_budget: Option<f64>,
    pub estimation_budget: Option<f64>,
    pub segment_tokens: Option<u32>,
    pub avg_cluster: Option<u32>,
    pub sink: Option<u32>,
    pub window: Option<u32>,
    pub min_context: Option<u32>,
    pub update_interval: Option<u32>,
    pub shadow: bool,
    pub index_cache_mb: Option<u32>,
}

/// Environment twins of the flags.
pub mod env {
    pub const RETRIEVAL_BUDGET: &str = "AVAROK_RETRO_RETRIEVAL_BUDGET";
    pub const ESTIMATION_BUDGET: &str = "AVAROK_RETRO_ESTIMATION_BUDGET";
    pub const SEGMENT_TOKENS: &str = "AVAROK_RETRO_SEGMENT_TOKENS";
    pub const AVG_CLUSTER: &str = "AVAROK_RETRO_AVG_CLUSTER";
    pub const SINK: &str = "AVAROK_RETRO_SINK";
    pub const WINDOW: &str = "AVAROK_RETRO_WINDOW";
    pub const MIN_CONTEXT: &str = "AVAROK_RETRO_MIN_CONTEXT";
    pub const UPDATE_INTERVAL: &str = "AVAROK_RETRO_UPDATE_INTERVAL";
    pub const SHADOW: &str = "AVAROK_RETRO_SHADOW";
    pub const INDEX_CACHE_MB: &str = "AVAROK_RETRO_INDEX_CACHE_MB";
    pub const SHADOW_EVERY: &str = "AVAROK_RETRO_SHADOW_EVERY";
    pub const SHADOW_LAYERS: &str = "AVAROK_RETRO_SHADOW_LAYERS";
    /// Debug: one `blocks=[..]` log line per (decode step, layer) for the
    /// L02 NIAH recall report (`niah_grid.py --selection-log`). Off by default;
    /// the toggle grammar of the other twins. Reads device lists back, so it is
    /// slow and meant for diagnosis runs only.
    pub const SELECTION_LOG: &str = "AVAROK_RETRO_SELECTION_LOG";
}

static CLI: OnceLock<CliValues> = OnceLock::new();
static PARAMS: OnceLock<RetroParams> = OnceLock::new();
static REGISTERED: AtomicBool = AtomicBool::new(false);

fn toggle(raw: Option<&str>) -> Option<bool> {
    let v = raw?.trim();
    if v.is_empty() {
        return None;
    }
    Some(!matches!(
        v.to_ascii_lowercase().as_str(),
        "0" | "false" | "off" | "no"
    ))
}

fn num<T: std::str::FromStr>(name: &str, raw: Option<String>) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    match raw.as_deref().map(str::trim) {
        None | Some("") => Ok(None),
        Some(v) => v
            .parse::<T>()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("{name}={v:?} is not a valid number: {e}")),
    }
}

/// Merge CLI values over `get`'s environment over the defaults and validate.
/// `get` is `std::env::var` in a serve; tests pass a table.
pub fn resolve_params(
    cli: &CliValues,
    get: &dyn Fn(&str) -> Option<String>,
) -> Result<RetroParams> {
    let d = RetroParams::default();
    let pick = |c: Option<u32>, name: &str, def: u32| -> Result<u32> {
        Ok(c.or(num::<u32>(name, get(name))?).unwrap_or(def))
    };
    let pickf = |c: Option<f64>, name: &str, def: f64| -> Result<f64> {
        Ok(c.or(num::<f64>(name, get(name))?).unwrap_or(def))
    };
    let shadow_layers = match get(env::SHADOW_LAYERS).as_deref().map(str::trim) {
        None | Some("") | Some("all") => None,
        Some(list) => Some(
            list.split(',')
                .map(|s| {
                    s.trim().parse::<u32>().with_context(|| {
                        format!(
                            "{}={list:?}: `{s}` is not a layer index",
                            env::SHADOW_LAYERS
                        )
                    })
                })
                .collect::<Result<Vec<u32>>>()?,
        ),
    };
    let p = RetroParams {
        retrieval_budget: pickf(
            cli.retrieval_budget,
            env::RETRIEVAL_BUDGET,
            d.retrieval_budget,
        )?,
        estimation_budget: pickf(
            cli.estimation_budget,
            env::ESTIMATION_BUDGET,
            d.estimation_budget,
        )?,
        segment_tokens: pick(cli.segment_tokens, env::SEGMENT_TOKENS, d.segment_tokens)?,
        avg_cluster: pick(cli.avg_cluster, env::AVG_CLUSTER, d.avg_cluster)?,
        sink: pick(cli.sink, env::SINK, d.sink)?,
        window: pick(cli.window, env::WINDOW, d.window)?,
        min_context: pick(cli.min_context, env::MIN_CONTEXT, d.min_context)?,
        update_interval: pick(cli.update_interval, env::UPDATE_INTERVAL, d.update_interval)?,
        kmeans_iters: d.kmeans_iters,
        shadow: cli.shadow || toggle(get(env::SHADOW).as_deref()).unwrap_or(false),
        index_cache_mb: pick(cli.index_cache_mb, env::INDEX_CACHE_MB, d.index_cache_mb)?,
        shadow_every: pick(None, env::SHADOW_EVERY, d.shadow_every)?,
        shadow_layers,
    };
    p.validate()?;
    Ok(p)
}

fn real_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// Startup validation (`port_apply` `@port:validate`): every flag and env
/// twin parses and is in range. Runs whether or not the row is on, so a typo
/// is a startup error instead of a silent default.
pub fn validate(cli: &CliValues) -> Result<()> {
    resolve_params(cli, &real_env).map(|_| ())
}

/// The resolved parameters (defaults until `apply` ran).
pub fn params() -> &'static RetroParams {
    PARAMS.get_or_init(|| {
        let cli = CLI.get().cloned().unwrap_or_default();
        resolve_params(&cli, &real_env).unwrap_or_default()
    })
}

/// Whether the selection debug log is on (`AVAROK_RETRO_SELECTION_LOG`, off
/// unless set; read once).
pub fn selection_log() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| toggle(real_env(env::SELECTION_LOG).as_deref()).unwrap_or(false))
}

/// Retro attention replaces dense decode attention: the CLI flag, else the row.
pub fn output_enabled() -> bool {
    CLI.get().is_some_and(|c| c.retro_attn) || crate::layers::ops::port_rows::retro_attn::row()
}

/// Whether the registered retro selector wants decode CUDA graphs off for the
/// serve. Read once at model build by `Qwen3AttentionLayer::decode_graph_unsupported`
/// (a selector that keeps per-sequence state on the host cannot let a captured
/// graph replay for another sequence; v1 runs retro decode eagerly). False on
/// every serve that did not register it.
pub fn vetoes_decode_graphs() -> bool {
    REGISTERED.load(Ordering::Relaxed)
}

/// Publish the CLI values and register the selector when retro attention or
/// shadow mode is on (`port_apply` `@port:apply`). Off: nothing happens.
pub fn apply(cli: CliValues) -> Result<()> {
    let _ = CLI.set(cli);
    let p = params().clone();
    let on = output_enabled();
    super::metrics::GLOBAL.set_active(on);
    if !on && !p.shadow {
        return Ok(());
    }
    if on {
        tracing::warn!(
            "retro attention is ON (row retro_attn / --retro-attn): full-attention decode at or above {} tokens \
             reads {:.1}% of clusters exactly and {:.1}% as centroids. This is APPROXIMATE attention; outputs \
             differ from dense attention. Decode runs without CUDA graphs while it is on.",
            p.min_context,
            p.retrieval_budget * 100.0,
            p.estimation_budget * 100.0,
        );
    }
    if p.shadow {
        tracing::info!(
            "retro shadow mode is ON: every {} decode steps (layers {}), the retro zones are measured against dense \
             attention on the CPU; atlas_retro_shadow_* carry the results{}",
            p.shadow_every,
            p.shadow_layers
                .as_ref()
                .map_or("all".to_string(), |l| format!("{l:?}")),
            if on { "" } else { "; output unchanged" },
        );
    }
    let sel = super::selector::RetroSelector::new(p, on);
    crate::layers::qwen3_attention::sparse_seam::register(std::sync::Arc::new(sel))?;
    REGISTERED.store(true, Ordering::Relaxed);
    Ok(())
}
