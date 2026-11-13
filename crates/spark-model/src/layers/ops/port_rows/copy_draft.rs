// SPDX-License-Identifier: AGPL-3.0-only
//! The `copy_draft` row: which copy-drafter plans the scheduler may use
//! (`off`, `chain`, `rounds`, `auto`; lane L13, `spark-server`
//! `scheduler/copy_draft`). Resolution: the CLI value if `spark serve` got
//! `--copy-draft` (published by `port_apply`, applied in spark-server), else
//! `AVAROK_COPY_DRAFT` when it names one of the allowed values (any case,
//! trimmed), else the target's declaration. An unknown environment value is
//! a startup error raised by spark-server's `port_apply::validate`, never a
//! panic here; this resolver falls back to the declaration for it.
use crate::layers::ops::target_defaults::{Resolved, Source};

pub const ENV: &str = "AVAROK_COPY_DRAFT";

/// The values the row accepts, `off` first (every target's baseline).
pub const ALLOWED: [&str; 4] = ["off", "chain", "rounds", "auto"];

/// `raw` as one of [`ALLOWED`], or `None`.
pub fn parse(raw: &str) -> Option<&'static str> {
    let v = raw.trim().to_ascii_lowercase();
    ALLOWED.iter().copied().find(|a| *a == v)
}

pub fn resolve(declared: &'static str, raw: Option<&str>) -> Resolved<&'static str> {
    match raw.and_then(parse) {
        Some(value) => Resolved {
            value,
            source: Source::Env,
        },
        None => Resolved {
            value: parse(declared).unwrap_or("off"),
            source: Source::Target,
        },
    }
}

pub fn row() -> &'static str {
    crate::layers::ops::target_defaults_port::resolved_port()
        .copy_draft
        .value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_or_unknown_env_keeps_the_declaration() {
        assert_eq!(
            resolve("off", None),
            Resolved {
                value: "off",
                source: Source::Target
            }
        );
        assert_eq!(
            resolve("off", Some("sometimes")),
            Resolved {
                value: "off",
                source: Source::Target
            }
        );
        assert_eq!(resolve("chain", None).value, "chain");
        // A declaration outside the set reads as off.
        assert_eq!(resolve("bogus", None).value, "off");
    }

    #[test]
    fn a_known_env_value_wins_in_any_case() {
        for v in ALLOWED {
            assert_eq!(
                resolve("off", Some(v)),
                Resolved {
                    value: v,
                    source: Source::Env
                }
            );
        }
        assert_eq!(resolve("off", Some(" Auto ")).value, "auto");
        assert_eq!(
            resolve("chain", Some("OFF")),
            Resolved {
                value: "off",
                source: Source::Env
            }
        );
    }

    #[test]
    fn every_target_declares_off() {
        // The sprint never flips this row on any target (master plan 3.5).
        assert_eq!(avarok_kernels::TARGET_DEFAULTS.port.copy_draft, "off");
    }
}
