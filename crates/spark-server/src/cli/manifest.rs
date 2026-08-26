// SPDX-License-Identifier: AGPL-3.0-only

//! Emitting the serve flag surface as a machine-readable snapshot.
//!
//! **This is a build artifact, not a public interface.** `ServeArgs` still has
//! no `Serialize` derive and this does not give it one: nothing here promises
//! that a flag will keep its name, and a rename shows up as a diff in whatever
//! consumes the snapshot rather than as a silent break. That is the point —
//! `atlas-recipes` currently hand-transcribes this surface from a *Python*
//! launcher that predates the Rust one, and the drift is invisible until a
//! launch dies inside a container.
//!
//! The reflection is the same one [`crate::tui::lib_fields`] already does, for
//! the same reason: clap is the single source of truth for the flag surface,
//! so reading it beats writing it down twice. What this adds over the TUI's
//! view is the **action** — whether a flag is presence-only — which is what a
//! downstream renderer cannot guess and gets wrong in a way that only fails on
//! the machine, at launch.

use clap::{ArgAction, CommandFactory};
use serde::Serialize;

/// Schema version of the emitted document.
///
/// Bumped when the *shape* changes, never when a flag does. A consumer pins
/// this and refuses a document it cannot read, rather than half-reading one.
pub const SCHEMA_VERSION: u32 = 1;

/// The whole snapshot.
#[derive(Debug, Serialize)]
pub struct Manifest {
    /// Shape of this document.
    pub schema_version: u32,
    /// Version of the engine that produced it.
    pub spark_version: String,
    /// Every serve flag, in clap's declaration order.
    pub flags: Vec<Flag>,
}

/// One flag, as clap describes it.
#[derive(Debug, Serialize)]
pub struct Flag {
    /// Recipe `defaults:` spelling — the long name with underscores.
    pub key: String,
    /// The long flag itself, hyphenated, without the leading dashes.
    pub flag: String,
    /// Whether the flag takes no value at all.
    ///
    /// The field a transcription gets wrong. `--gdn-fused-norm` is
    /// presence-only and `--ssm-h-dtype` is not; rendering either as the other
    /// produces a command clap refuses, on the serving machine, after the
    /// operator has already reviewed it.
    pub presence_only: bool,
    /// Whether clap parses the value as a bool.
    pub is_bool: bool,
    /// The closed set of accepted values, when there is one.
    ///
    /// Empty means free-form. Taken from `cli::flag_values`, the same module
    /// `validate_serve_args` enforces against, so a picker built from this
    /// cannot offer a value the launch would refuse.
    pub options: Vec<String>,
    /// clap's default, when it declares one.
    pub default: Option<String>,
    /// First line of help, for a label.
    pub help: Option<String>,
    /// Recipe spellings that mean this flag but do not match its name.
    ///
    /// A consumer cannot derive these: `max_model_len` is vLLM's spelling kept
    /// in the recipes for cross-runtime familiarity, and the flag is
    /// `--max-seq-len`. Omitting them would make a generated table silently
    /// stop claiming keys that shipping recipes actually set — which is the
    /// class of bug this whole document exists to end.
    pub recipe_aliases: Vec<String>,
}

/// Build the snapshot by asking clap.
#[must_use]
pub fn build() -> Manifest {
    let bool_parser = clap::builder::ValueParser::bool().type_id();
    let command = super::ServeArgs::command();

    let flags = command
        .get_arguments()
        .filter_map(|arg| {
            // No long name is the positional MODEL. Help and version are
            // clap's own and describe nothing a recipe can set.
            let long = arg.get_long()?;
            match arg.get_action() {
                ArgAction::Help
                | ArgAction::HelpShort
                | ArgAction::HelpLong
                | ArgAction::Version => return None,
                _ => {}
            }
            let presence_only = matches!(arg.get_action(), ArgAction::SetTrue);
            let is_bool = arg.get_value_parser().type_id() == bool_parser;
            let options = if is_bool && presence_only {
                Vec::new()
            } else if is_bool {
                vec!["true".to_owned(), "false".to_owned()]
            } else {
                super::flag_values::options_for_flag(long).unwrap_or_default()
            };
            Some(Flag {
                key: long.replace('-', "_"),
                flag: long.to_owned(),
                presence_only,
                is_bool,
                options,
                default: arg
                    .get_default_values()
                    .first()
                    .map(|v| v.to_string_lossy().into_owned()),
                help: arg
                    .get_help()
                    .map(|h| h.to_string().lines().next().unwrap_or_default().to_owned()),
                recipe_aliases: crate::recipe::schema::RENAMES
                    .iter()
                    .filter(|(_, flag)| *flag == long)
                    .map(|(key, _)| (*key).to_owned())
                    .collect(),
            })
        })
        .collect();

    Manifest {
        schema_version: SCHEMA_VERSION,
        spark_version: super::ATLAS_VERSION.to_owned(),
        flags,
    }
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
