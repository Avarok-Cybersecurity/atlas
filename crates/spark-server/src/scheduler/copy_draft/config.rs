// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the copy drafter's serve configuration, resolved once.
//!
//! Mode: `--copy-draft {off,chain,rounds,auto}` if given, else the
//! `copy_draft` row (`AVAROK_COPY_DRAFT`, then `kernels/<hw>/HARDWARE.toml`;
//! `off` on every target). Knobs: `--copy-draft-min-match` (default 8, the
//! agreement a copy source needs before it may fill the MTP chain),
//! `--copy-draft-max-len` (default 15, the longest lookup round) and
//! `--copy-draft-advantage` (default 1.3, the yield gate's factor), each also
//! readable from its `AVAROK_COPY_DRAFT_*` variable when the flag is absent.
//!
//! [`validate_args`] runs from `port_apply::validate` and turns every unknown
//! or out-of-range value into a startup error; [`apply_args`] publishes the
//! flags before any model is built. [`config`] is the one resolution every
//! consumer reads (`OnceLock`, so a route stays constant for the whole serve).

use std::sync::OnceLock;

/// Which copy plans the planner may choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// No copy plan; the legacy n-gram mode keeps today's scan.
    Off,
    /// Copy chains fill the MTP verify chain (same width), gated by yield.
    Chain,
    /// Lookup rounds only: a confident copy of up to `max_len` tokens is
    /// verified alone through the variable-width verify.
    Rounds,
    /// Both: a lookup round when confident, else a chain when it pays.
    Auto,
}

/// The accepted values of `--copy-draft` / `AVAROK_COPY_DRAFT`.
pub(crate) const MODES: [&str; 4] = ["off", "chain", "rounds", "auto"];

impl Mode {
    pub fn parse(raw: &str) -> Option<Mode> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Mode::Off),
            "chain" => Some(Mode::Chain),
            "rounds" => Some(Mode::Rounds),
            "auto" => Some(Mode::Auto),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Off => "off",
            Mode::Chain => "chain",
            Mode::Rounds => "rounds",
            Mode::Auto => "auto",
        }
    }

    pub fn chains(self) -> bool {
        matches!(self, Mode::Chain | Mode::Auto)
    }

    pub fn rounds(self) -> bool {
        matches!(self, Mode::Rounds | Mode::Auto)
    }
}

pub(crate) const MODE_ENV: &str = "AVAROK_COPY_DRAFT";
pub(crate) const MIN_MATCH_ENV: &str = "AVAROK_COPY_DRAFT_MIN_MATCH";
pub(crate) const MAX_LEN_ENV: &str = "AVAROK_COPY_DRAFT_MAX_LEN";
pub(crate) const ADVANTAGE_ENV: &str = "AVAROK_COPY_DRAFT_ADVANTAGE";

pub(crate) const DEFAULT_MIN_MATCH: usize = 8;
pub(crate) const DEFAULT_MAX_LEN: usize = 15;
pub(crate) const DEFAULT_ADVANTAGE: f32 = 1.3;
/// Agreement bounds: below 2 a copy is noise; above 64 no source can reach
/// it inside the backward extension the planner trusts.
pub(crate) const MIN_MATCH_RANGE: (usize, usize) = (2, 64);
/// Lookup round length bounds: 15 drafts + the anchor = 16 verify rows, the
/// widest variable-width verify this sprint relies on.
pub(crate) const MAX_LEN_RANGE: (usize, usize) = (1, 15);

/// The resolved configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Config {
    pub mode: Mode,
    pub min_match: usize,
    pub max_len: usize,
    pub advantage: f32,
    /// `cli`, `env` or `target`: where the mode came from (boot log).
    pub mode_source: &'static str,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            mode: Mode::Off,
            min_match: DEFAULT_MIN_MATCH,
            max_len: DEFAULT_MAX_LEN,
            advantage: DEFAULT_ADVANTAGE,
            mode_source: "target",
        }
    }
}

/// The flag values as given (`None` = absent).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Cli {
    pub mode: Option<String>,
    pub min_match: Option<u32>,
    pub max_len: Option<u32>,
    pub advantage: Option<f32>,
}

impl Cli {
    pub fn from_args(args: &crate::cli::serve_args_port::PortArgs) -> Self {
        Self {
            mode: args.copy_draft.clone(),
            min_match: args.copy_draft_min_match,
            max_len: args.copy_draft_max_len,
            advantage: args.copy_draft_advantage,
        }
    }
}

fn int_in(name: &str, v: u64, (lo, hi): (usize, usize)) -> Result<usize, String> {
    let v = v as usize;
    if (lo..=hi).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{name} = {v} is outside {lo}..={hi}"))
    }
}

/// Resolve the configuration as a pure function of the flags, the row's
/// resolved value (`row`, with `row_from_env` naming its source) and a
/// variable lookup. Every unknown or out-of-range value is an error.
pub(crate) fn resolve(
    cli: &Cli,
    row: &str,
    row_from_env: bool,
    var: impl Fn(&str) -> Option<String>,
) -> Result<Config, String> {
    let mut c = Config::default();
    if let Some(raw) = var(MODE_ENV)
        && Mode::parse(&raw).is_none()
    {
        return Err(format!(
            "{MODE_ENV}={raw:?} is not one of {}",
            MODES.join(", ")
        ));
    }
    match &cli.mode {
        Some(raw) => {
            c.mode = Mode::parse(raw).ok_or_else(|| {
                format!("--copy-draft {raw:?} is not one of {}", MODES.join(", "))
            })?;
            c.mode_source = "cli";
        }
        None => {
            c.mode = Mode::parse(row).unwrap_or(Mode::Off);
            c.mode_source = if row_from_env { "env" } else { "target" };
        }
    }
    let env_int = |name: &str| -> Result<Option<u64>, String> {
        match var(name) {
            None => Ok(None),
            Some(raw) => raw
                .trim()
                .parse::<u64>()
                .map(Some)
                .map_err(|_| format!("{name}={raw:?} is not an integer")),
        }
    };
    let min_match = match cli.min_match {
        Some(v) => Some(v as u64),
        None => env_int(MIN_MATCH_ENV)?,
    };
    if let Some(v) = min_match {
        c.min_match = int_in("copy draft min match", v, MIN_MATCH_RANGE)?;
    }
    let max_len = match cli.max_len {
        Some(v) => Some(v as u64),
        None => env_int(MAX_LEN_ENV)?,
    };
    if let Some(v) = max_len {
        c.max_len = int_in("copy draft max len", v, MAX_LEN_RANGE)?;
    }
    let advantage = match cli.advantage {
        Some(v) => Some(v),
        None => match var(ADVANTAGE_ENV) {
            None => None,
            Some(raw) => Some(
                raw.trim()
                    .parse::<f32>()
                    .map_err(|_| format!("{ADVANTAGE_ENV}={raw:?} is not a number"))?,
            ),
        },
    };
    if let Some(v) = advantage {
        if !(v.is_finite() && v > 0.0 && v <= 100.0) {
            return Err(format!("copy draft advantage = {v} is outside (0, 100]"));
        }
        c.advantage = v;
    }
    Ok(c)
}

static CLI: OnceLock<Cli> = OnceLock::new();

/// The process resolution. The row is resolved with its own pure resolver
/// over the same inputs `resolved_port()` uses, so `port_apply` (which runs
/// this) never seals the shared port table before another lane's
/// `@port:apply` line has published its flags.
fn resolve_process(cli: &Cli) -> Result<Config, String> {
    use spark_model::layers::ops::port_rows::copy_draft as row;
    let r = row::resolve(
        avarok_kernels::TARGET_DEFAULTS.port.copy_draft,
        std::env::var(row::ENV).ok().as_deref(),
    );
    resolve(cli, r.value, r.from_env(), |name| std::env::var(name).ok())
}

/// Startup validation (`port_apply::validate`).
pub(crate) fn validate_args(args: &crate::cli::serve_args_port::PortArgs) -> anyhow::Result<()> {
    resolve_process(&Cli::from_args(args))
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("copy draft: {e}"))
}

/// Publish the flags (`port_apply::apply`) and log the resolved setup once
/// when the drafter is on. An off drafter logs nothing.
pub(crate) fn apply_args(args: &crate::cli::serve_args_port::PortArgs) {
    let _ = CLI.set(Cli::from_args(args));
    let c = config();
    if c.mode != Mode::Off {
        tracing::info!(
            "copy draft: mode={} ({}) min_match={} max_len={} advantage={}",
            c.mode.as_str(),
            c.mode_source,
            c.min_match,
            c.max_len,
            c.advantage,
        );
    }
}

/// The serve's copy-draft configuration.
pub(crate) fn config() -> &'static Config {
    static CONFIG: OnceLock<Config> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let cli = CLI.get().cloned().unwrap_or_default();
        resolve_process(&cli).unwrap_or_else(|e| {
            // `validate_args` refuses this at startup; a process that never
            // ran it (a test binary) keeps the drafter off.
            tracing::warn!("copy draft: {e}; the copy drafter stays off");
            Config::default()
        })
    })
}

/// True when any copy plan (or the legacy index) is in use.
pub(crate) fn active() -> bool {
    config().mode != Mode::Off
}
