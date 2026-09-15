// SPDX-License-Identifier: AGPL-3.0-only

//! Named serve presets: `spark serve <preset-name>`.
//!
//! A preset is declared per kernel target in MODEL.toml `[[serve_presets]]`
//! (see `atlas_kernels::ServePreset`) and names ONE checkpoint together with
//! the flags and `ATLAS_*` gates it was validated under. This module is the
//! only place that turns a preset into a running configuration, and it does so
//! with two rules that make every entry a DEFAULT rather than a pin:
//!
//! * **Flags** are rendered to argv through the recipe schema
//!   (`recipe::schema`) and the command line is re-parsed by clap — the same
//!   round trip a recipe takes, so clap stays the single source of truth for
//!   the flag surface and a typo in MODEL.toml fails at startup with the flag
//!   named. A flag the operator passed is NOT rendered at all (decided from
//!   clap's `ValueSource`, not from comparing values), so an explicit
//!   `--num-drafts 1` beats a preset `num_drafts = 2` even though 1 is also
//!   clap's default.
//! * **Environment** variables are set only when unset. An operator's value is
//!   kept and the deviation is logged at WARN, because a run that departs from
//!   the validated configuration should say so in its own log.
//!
//! Expansion happens in `main` right after argument parsing and BEFORE
//! `serve()`: the host records the args for swap-restore, the dashboard draws
//! them as badge chips and `validate_serve_args` checks them, and all three
//! must see the expanded configuration, not the two-word one the operator
//! typed. The `ATLAS_*` gates are read lazily (`std::env::var` at the layer
//! constructors, or `OnceLock`s touched on first dispatch), so setting them
//! before any model code runs is sufficient — and setting them later is not.

use std::ffi::OsString;

use anyhow::{Context, Result, bail};
use atlas_kernels::ServePreset;
use clap::{CommandFactory, FromArgMatches, Parser as _, parser::ValueSource};

use crate::cli::{Cli, Command, ServeArgs};
use crate::recipe::schema;

/// Serve flags an env value may reference as `{name}`. Deliberately tiny: a
/// placeholder exists for a cap that MUST track a flag (the QSA per-sequence
/// cap equals `--max-seq-len`), not as a templating language. Extending it is
/// a one-line change here plus the match arm in `substitute`.
pub(crate) const PLACEHOLDERS: &[&str] = &["max_seq_len", "max_prefill_tokens"];

/// A preset matched by the positional `MODEL` argument.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PresetMatch {
    /// The kernel-target directory that declares it (`--kernel-target` value).
    pub target: &'static str,
    pub preset: &'static ServePreset,
}

/// Is `spec` a preset name? A directory that exists on disk, or anything with
/// a `/` in it (an HF id), never is — those keep their ordinary meaning even
/// if a preset happened to share the spelling.
pub(crate) fn lookup(spec: &str) -> Option<PresetMatch> {
    if spec.contains('/') || std::path::Path::new(spec).is_dir() {
        return None;
    }
    atlas_kernels::preset_named(spec).map(|(target, preset)| PresetMatch { target, preset })
}

/// What a preset did to this invocation — logged once the subscriber is up.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct Applied {
    pub preset: String,
    pub target: String,
    pub description: String,
    /// Rendered flag defaults that were appended, e.g. `--max-seq-len 32768`.
    pub flags_applied: Vec<String>,
    /// Flags the operator passed, which the preset therefore left alone.
    pub flags_overridden: Vec<String>,
    /// `(VAR, value)` pairs set into the environment.
    pub env_applied: Vec<(String, String)>,
    /// `(VAR, operator's value)` pairs found already set and left alone.
    pub env_kept: Vec<(String, String)>,
}

/// Render the preset's flag defaults for every flag the operator did NOT pass,
/// plus `--kernel-target <declaring target>` on the same condition.
///
/// `user_set(id)` answers for a clap arg id (the `ServeArgs` field name).
/// Returns `(argv to append, applied descriptions, overridden flags)`. Keys
/// that are not serve flags at all are refused here rather than dropped: a
/// key that renders to nothing would silently serve the unmodified config.
pub(crate) fn default_argv(
    m: PresetMatch,
    user_set: &dyn Fn(&str) -> bool,
) -> Result<(Vec<String>, Vec<String>, Vec<String>)> {
    let mut argv = Vec::new();
    let mut applied = Vec::new();
    let mut overridden = Vec::new();
    for (key, value) in m.preset.flags {
        let Some(flag) = schema::flag_for(key) else {
            bail!(
                "serve preset {:?}: flags.{key} is not a `spark serve` flag, so a default for it \
                 would change nothing",
                m.preset.name
            );
        };
        // clap derive ids are the field names; the long flag is the dashed form.
        let id = flag.replace('-', "_");
        if user_set(&id) {
            overridden.push(format!("--{flag}"));
            continue;
        }
        match schema::argv_for(key, value) {
            Some(tokens) => {
                applied.push(tokens.join(" "));
                argv.extend(tokens);
            }
            // A presence-only flag defaulted to `false`: nothing to pass, the
            // server default (off) stands. Say so rather than vanish.
            None => applied.push(format!("--{flag} (off)")),
        }
    }
    if user_set("kernel_target") {
        overridden.push("--kernel-target".to_string());
    } else {
        applied.push(format!("--kernel-target {}", m.target));
        argv.push("--kernel-target".to_string());
        argv.push(m.target.to_string());
    }
    Ok((argv, applied, overridden))
}

/// Resolve `{max_seq_len}`-style placeholders against the EFFECTIVE args (after
/// the operator's overrides), so a cap declared as tracking a flag tracks the
/// flag the server will actually run with.
pub(crate) fn substitute(value: &str, args: &ServeArgs) -> Result<String> {
    let mut out = String::new();
    let mut rest = value;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        let Some(end) = after.find('}') else {
            bail!("preset env value {value:?} has an unterminated '{{'");
        };
        let name = &after[..end];
        let resolved = match name {
            "max_seq_len" => args.max_seq_len.to_string(),
            "max_prefill_tokens" => args.max_prefill_tokens.to_string(),
            other => bail!(
                "preset env value {value:?} references {{{other}}}, which is not a \
                 substitutable serve flag (one of {PLACEHOLDERS:?})"
            ),
        };
        out.push_str(&resolved);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// `(VAR, value)` pairs — what [`env_plan`] decides to set, and what it kept.
pub(crate) type EnvPairs = Vec<(String, String)>;

/// Decide the environment: `(to set, kept)`. `current(var)` is the value
/// already in the environment, if any; a set variable is kept verbatim.
pub(crate) fn env_plan(
    preset: &ServePreset,
    args: &ServeArgs,
    current: &dyn Fn(&str) -> Option<String>,
) -> Result<(EnvPairs, EnvPairs)> {
    let mut apply = Vec::new();
    let mut kept = Vec::new();
    for (var, value) in preset.env {
        match current(var) {
            Some(existing) => kept.push((var.to_string(), existing)),
            None => apply.push((var.to_string(), substitute(value, args)?)),
        }
    }
    Ok((apply, kept))
}

/// Parse the process command line, expanding a serve preset if the positional
/// `MODEL` names one. Replaces `Cli::parse()` in `main`; identical for every
/// other invocation (same `--help` / error exits).
pub(crate) fn parse_cli() -> Result<(Cli, Option<Applied>)> {
    parse_from(std::env::args_os().collect())
}

/// [`parse_cli`] over an explicit argv.
pub(crate) fn parse_from(argv: Vec<OsString>) -> Result<(Cli, Option<Applied>)> {
    // `get_matches_from` exits on `--help`/`--version`/a bad flag exactly as
    // `Cli::parse()` does, so nothing about a non-preset invocation changes.
    let matches = Cli::command().get_matches_from(argv.clone());
    let cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());
    let Command::Serve(args) = &cli.command else {
        return Ok((cli, None));
    };
    let Some(m) = args.model.as_deref().and_then(lookup) else {
        return Ok((cli, None));
    };
    let sub = matches
        .subcommand_matches("serve")
        .expect("Command::Serve was parsed from the serve subcommand");
    // `try_contains_id` guards `value_source`, which panics on an id clap does
    // not know (a preset key that is not a field — refused by name below).
    let user_set = |id: &str| {
        sub.try_contains_id(id).is_ok() && sub.value_source(id) == Some(ValueSource::CommandLine)
    };
    let (extra, flags_applied, flags_overridden) = default_argv(m, &user_set)?;

    let mut full = argv;
    full.extend(extra.iter().map(OsString::from));
    let cli = Cli::try_parse_from(&full).with_context(|| {
        format!(
            "serve preset {:?} (declared by kernel target {}) rendered an invalid command line \
             — appended: {}",
            m.preset.name,
            m.target,
            extra.join(" ")
        )
    })?;
    let Command::Serve(args) = &cli.command else {
        unreachable!("re-parsing a serve command line yields a serve command");
    };

    let (env_applied, env_kept) = env_plan(m.preset, args, &|var| {
        std::env::var_os(var).map(|v| v.to_string_lossy().into_owned())
    })?;
    for (var, value) in &env_applied {
        // SAFETY: called from `main` before the tracing subscriber, the
        // dashboard thread or any model code exists; the only other threads
        // are the tokio runtime's parked workers, which read no environment.
        // Same footing as `tui::logo`'s COLORTERM publication.
        unsafe { std::env::set_var(var, value) };
    }
    Ok((
        cli,
        Some(Applied {
            preset: m.preset.name.to_string(),
            target: m.target.to_string(),
            description: m.preset.description.to_string(),
            flags_applied,
            flags_overridden,
            env_applied,
            env_kept,
        }),
    ))
}

/// The startup record of what the preset decided — every value in force is
/// named, because a configuration nobody can read back is a configuration
/// nobody can reproduce.
pub(crate) fn log_applied(a: &Applied) {
    tracing::info!(
        "Serve preset {:?} (kernel target {}): {}",
        a.preset,
        a.target,
        a.description
    );
    tracing::info!(
        "Preset flag defaults applied: {}",
        if a.flags_applied.is_empty() {
            "(none)".to_string()
        } else {
            a.flags_applied.join(" ")
        }
    );
    if !a.flags_overridden.is_empty() {
        tracing::info!(
            "Preset flag defaults OVERRIDDEN by the command line: {}",
            a.flags_overridden.join(" ")
        );
    }
    for (var, value) in &a.env_applied {
        tracing::info!("Preset env default applied: {var}={value}");
    }
    for (var, value) in &a.env_kept {
        tracing::warn!(
            "Preset env default NOT applied: {var} was already set to {value:?} — the operator's \
             value stands, and this run departs from the validated configuration"
        );
    }
}

#[cfg(test)]
#[path = "serve_presets_tests.rs"]
mod tests;
