// SPDX-License-Identifier: AGPL-3.0-only
//! The `mtp_vocab_rank` row: which LM-head rows the MTP draft-only head holds
//! (`id` = the first `--mtp-vocab` token ids, the baseline; `freq` = the most
//! frequent tokens of the shipped table, `layers/mtp_head/vocab_rank.rs`).
//! Resolution: `--mtp-vocab-rank` when given (published by `port_apply` before
//! the table is first read; reported as an override), else
//! `AVAROK_MTP_VOCAB_RANK` (`id` / `freq`, case-insensitive), else the target's
//! declaration. An unknown environment value falls back to the declaration
//! here and is a startup error through [`validate`], never a panic in the
//! resolver.

use std::sync::OnceLock;

use crate::layers::ops::target_defaults::{Resolved, Source};

pub const ENV: &str = "AVAROK_MTP_VOCAB_RANK";

/// Every value the row accepts.
pub const ALLOWED: &[&str] = &["id", "freq"];

static CLI: OnceLock<&'static str> = OnceLock::new();

/// Map a raw value onto one of [`ALLOWED`] (trimmed, case-insensitive).
pub fn parse(raw: &str) -> Option<&'static str> {
    let v = raw.trim().to_ascii_lowercase();
    ALLOWED.iter().copied().find(|a| *a == v)
}

/// Publish `--mtp-vocab-rank` (once, before the port table is first read).
/// `None` publishes nothing.
pub fn set_cli(v: Option<&str>) -> anyhow::Result<()> {
    let Some(raw) = v else {
        return Ok(());
    };
    let value = parse(raw)
        .ok_or_else(|| anyhow::anyhow!("--mtp-vocab-rank {raw:?}: expected one of {ALLOWED:?}"))?;
    if CLI.set(value).is_err() && CLI.get() != Some(&value) {
        anyhow::bail!("--mtp-vocab-rank published twice with different values");
    }
    Ok(())
}

/// Reject an `AVAROK_MTP_VOCAB_RANK` value the row does not know.
pub fn validate(env_value: Option<&str>) -> anyhow::Result<()> {
    match env_value {
        Some(raw) if parse(raw).is_none() => {
            anyhow::bail!("{ENV}={raw:?}: expected one of {ALLOWED:?}")
        }
        _ => Ok(()),
    }
}

/// The row against the target's declaration and the raw environment value.
pub fn resolve(declared: &'static str, raw: Option<&str>) -> Resolved<&'static str> {
    resolve_with(CLI.get().copied(), declared, raw)
}

/// [`resolve`] with the CLI value passed in (pure, for tests).
pub fn resolve_with(
    cli: Option<&'static str>,
    declared: &'static str,
    raw: Option<&str>,
) -> Resolved<&'static str> {
    if let Some(v) = cli {
        return Resolved {
            value: v,
            source: Source::Env,
        };
    }
    if let Some(v) = raw.and_then(parse) {
        return Resolved {
            value: v,
            source: Source::Env,
        };
    }
    Resolved {
        value: parse(declared).unwrap_or("id"),
        source: Source::Target,
    }
}

pub fn row() -> &'static str {
    crate::layers::ops::target_defaults_port::resolved_port()
        .mtp_vocab_rank
        .value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_declaration_holds_without_an_override() {
        let r = resolve_with(None, "id", None);
        assert_eq!((r.value, r.source), ("id", Source::Target));
        let r = resolve_with(None, "freq", None);
        assert_eq!((r.value, r.source), ("freq", Source::Target));
    }

    #[test]
    fn the_environment_overrides_and_an_unknown_value_falls_back() {
        let r = resolve_with(None, "id", Some(" FREQ "));
        assert_eq!((r.value, r.source), ("freq", Source::Env));
        let r = resolve_with(None, "id", Some("top"));
        assert_eq!((r.value, r.source), ("id", Source::Target));
        assert!(validate(Some("top")).is_err());
        assert!(validate(Some("freq")).is_ok());
        assert!(validate(None).is_ok());
    }

    #[test]
    fn the_flag_wins_over_the_environment() {
        let r = resolve_with(Some("id"), "freq", Some("freq"));
        assert_eq!((r.value, r.source), ("id", Source::Env));
    }

    #[test]
    fn an_unknown_declaration_reads_as_the_baseline() {
        assert_eq!(resolve_with(None, "bogus", None).value, "id");
    }

    #[test]
    fn parse_accepts_exactly_the_allowed_values() {
        assert_eq!(parse("id"), Some("id"));
        assert_eq!(parse("Freq"), Some("freq"));
        assert_eq!(parse(""), None);
        assert_eq!(parse("frequency"), None);
    }
}
