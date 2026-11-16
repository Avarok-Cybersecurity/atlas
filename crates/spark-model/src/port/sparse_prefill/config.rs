// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: resolve the block-sparse prefill knobs (CLI flag, else environment, else default)
//! and the `prefill_sparse_attn` row into one sealed [`Config`], and refuse values that cannot be served.

use std::sync::OnceLock;

use anyhow::{Result, bail};

use super::{
    DEFAULT_MIN_TOKENS, DEFAULT_STRIDE, DEFAULT_THRESHOLD, ENV_MIN_TOKENS, ENV_STRIDE,
    ENV_THRESHOLD, STRIDES,
};

/// The three `spark serve` flags as given (`None` = absent).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Knobs {
    pub min_tokens: Option<u32>,
    pub threshold: Option<f32>,
    pub stride: Option<u32>,
}

impl Knobs {
    /// Whether any flag was given.
    pub fn any(&self) -> bool {
        self.min_tokens.is_some() || self.threshold.is_some() || self.stride.is_some()
    }
}

/// What the layer hook reads on every chunk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// The `prefill_sparse_attn` row. Off: the hook returns before reading anything else.
    pub enabled: bool,
    /// A chunk goes sparse when it attends to at least this many keys.
    pub min_tokens: u32,
    /// Softmax mass each query block keeps, in (0, 1]; 1.0 takes the dense kernel.
    pub threshold: f32,
    /// Antidiagonal stride, one of [`STRIDES`].
    pub stride: u32,
}

impl Config {
    /// The configuration with the row off and every knob at its default.
    pub const OFF: Config = Config {
        enabled: false,
        min_tokens: DEFAULT_MIN_TOKENS,
        threshold: DEFAULT_THRESHOLD,
        stride: DEFAULT_STRIDE,
    };

    /// Whether this configuration means "keep every block" (the dense kernel serves it).
    pub fn keeps_all(&self) -> bool {
        self.threshold.is_nan() || self.threshold >= 1.0
    }

    /// The boot log line for a configuration with the row on.
    pub fn boot_line(&self) -> String {
        format!(
            "prefill_sparse_attn ON (APPROXIMATE): block-sparse prefill attention for chunks attending to \
             >= {} tokens, threshold {}, stride {}, {}-token blocks (sink + diagonal band always kept){}",
            self.min_tokens,
            self.threshold,
            self.stride,
            super::BLOCK,
            if self.keeps_all() {
                "; threshold 1.0 keeps every block, so the dense kernel serves (identity gate)"
            } else {
                ""
            }
        )
    }
}

/// The published flags (`port_apply::apply`; first call wins).
static CLI: OnceLock<Knobs> = OnceLock::new();

/// Publish the serve flags. Called once from `port_apply::apply`, before any model is built.
pub fn set_cli(k: Knobs) {
    if CLI.set(k).is_err() && CLI.get() != Some(&k) {
        tracing::warn!("sparse_prefill: flags were already published; keeping the first set");
    }
}

/// The process-wide configuration. An invalid value here (only reachable when `validate_flags` did not
/// run, as in a test binary) logs an error and serves with the row off.
pub fn resolved() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let cli = CLI.get().copied().unwrap_or_default();
        match resolve(super::row_on(), &cli, |name| std::env::var(name).ok()) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("sparse_prefill: {e:#}; block-sparse prefill stays off");
                Config::OFF
            }
        }
    })
}

/// [`Config`] as a pure function of the row, the flags and a variable lookup. Every knob is checked
/// whether or not the row is on, so a bad value is a startup error either way.
pub fn resolve(row: bool, cli: &Knobs, var: impl Fn(&str) -> Option<String>) -> Result<Config> {
    let min_tokens = match cli.min_tokens {
        Some(v) => v,
        None => parse_env(&var, ENV_MIN_TOKENS, |s| s.parse::<u32>().ok())?
            .unwrap_or(DEFAULT_MIN_TOKENS),
    };
    let threshold = match cli.threshold {
        Some(v) => v,
        None => {
            parse_env(&var, ENV_THRESHOLD, |s| s.parse::<f32>().ok())?.unwrap_or(DEFAULT_THRESHOLD)
        }
    };
    let stride = match cli.stride {
        Some(v) => v,
        None => parse_env(&var, ENV_STRIDE, |s| s.parse::<u32>().ok())?.unwrap_or(DEFAULT_STRIDE),
    };
    if min_tokens == 0 {
        bail!("--prefill-sparse-min-tokens ({ENV_MIN_TOKENS}) must be at least 1, got 0");
    }
    let threshold_ok = threshold.is_finite() && threshold > 0.0 && threshold <= 1.0;
    if !threshold_ok {
        bail!("--prefill-sparse-threshold ({ENV_THRESHOLD}) must be in (0, 1], got {threshold}");
    }
    if !STRIDES.contains(&stride) {
        bail!("--prefill-sparse-stride ({ENV_STRIDE}) must be one of {STRIDES:?}, got {stride}");
    }
    Ok(Config {
        enabled: row,
        min_tokens,
        threshold,
        stride,
    })
}

/// An environment value: `Ok(None)` when unset or blank, an error when set but unparsable.
fn parse_env<T>(
    var: &impl Fn(&str) -> Option<String>,
    name: &str,
    parse: impl Fn(&str) -> Option<T>,
) -> Result<Option<T>> {
    match var(name) {
        None => Ok(None),
        Some(raw) if raw.trim().is_empty() => Ok(None),
        Some(raw) => match parse(raw.trim()) {
            Some(v) => Ok(Some(v)),
            None => bail!("{name}={raw:?} is not a valid value"),
        },
    }
}
