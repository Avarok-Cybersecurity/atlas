// SPDX-License-Identifier: AGPL-3.0-only

//! What the Config form knows about `spark serve` flags it could add or pick.
//!
//! Everything here is READ OUT OF CLAP at runtime — long name, help line,
//! default, arity — rather than written down again. `ServeArgs` is the single
//! source of truth for the flag surface, and a hand-kept copy of it in the
//! dashboard is exactly the list that drifts: it offers a flag the server
//! dropped, or misses the one flag the user came to add. The enumerated value
//! sets come from `cli::flag_values`, the same module the CLI validator
//! enforces, so the picker can never offer a value a launch would refuse.

use std::sync::OnceLock;

use clap::{ArgAction, CommandFactory as _};

/// One addable `spark serve` flag, shaped for the form.
#[derive(Clone, Debug)]
pub struct FieldSpec {
    /// The `defaults:`-map spelling the override will be keyed by
    /// (underscores; `recipe::schema::flag_for` maps it back to the flag).
    pub key: String,
    /// clap's long name (dashes).
    pub flag: String,
    /// First line of the clap help, for the add-list row.
    pub help: String,
    /// The WHOLE clap help, for the picker's side panel. Several of these doc
    /// comments run to paragraphs (`kv_cache_dtype` documents its experimental
    /// variants at length) and the first line alone was all the picker could
    /// show — the panel is where the rest becomes readable.
    pub help_full: String,
    /// clap's default, when the flag has one. `None` means the server
    /// computes the value itself — the form must ask for one, not invent one.
    pub default: Option<String>,
    /// The closed value set, empty for free-form flags. Booleans get
    /// `true`/`false` here because clap knows their type; string enums come
    /// from `cli::flag_values`.
    pub options: Vec<String>,
}

/// Every `spark serve` flag the form can carry, built once per process (the
/// flag surface cannot change while it runs).
///
/// Excluded on principle, not by name: the positional MODEL (a recipe already
/// owns it) and repeatable flags (`--lora-adapter` and friends), because a
/// recipe's `defaults:` map holds one value per key and offering a field the
/// form cannot represent would truncate the user's input to its last entry.
pub fn serve_fields() -> &'static [FieldSpec] {
    static FIELDS: OnceLock<Vec<FieldSpec>> = OnceLock::new();
    FIELDS.get_or_init(build)
}

/// The spec for a form row's key, matched THROUGH the flag it renders to, so
/// a recipe's vLLM spellings (`max_model_len`, `host`) find the same spec as
/// the native ones (`max_seq_len`, `bind`).
pub fn spec_for_key(key: &str) -> Option<&'static FieldSpec> {
    let flag = crate::recipe::schema::flag_for(key)?;
    serve_fields().iter().find(|s| s.flag == flag)
}

fn build() -> Vec<FieldSpec> {
    let bool_parser_id = clap::builder::ValueParser::bool().type_id();
    let command = crate::cli::ServeArgs::command();
    command
        .get_arguments()
        .filter_map(|arg| {
            // No long name: the positional MODEL. Help/version never appear
            // on a derived args struct, but cost nothing to refuse.
            let long = arg.get_long()?;
            match arg.get_action() {
                ArgAction::Append | ArgAction::Count => return None,
                ArgAction::Help
                | ArgAction::HelpShort
                | ArgAction::HelpLong
                | ArgAction::Version => return None,
                _ => {}
            }
            let is_bool = arg.get_value_parser().type_id() == bool_parser_id;
            let options = if is_bool {
                // `recipe::schema::argv_for` renders these per clap's action:
                // a presence-only (SetTrue) flag becomes bare `--flag`/absent,
                // a value-taking bool keeps `true`/`false` on the line.
                vec!["true".to_string(), "false".to_string()]
            } else {
                crate::cli::flag_values::options_for_flag(long).unwrap_or_default()
            };
            // The LONG help: clap keeps only the first paragraph of a doc
            // comment in `get_help`, and the panel exists for the rest.
            let help_full = arg
                .get_long_help()
                .or_else(|| arg.get_help())
                .map(|h| h.to_string())
                .unwrap_or_default();
            Some(FieldSpec {
                key: long.replace('-', "_"),
                flag: long.to_string(),
                help: arg
                    .get_help()
                    .map(|h| h.to_string().lines().next().unwrap_or("").to_string())
                    .unwrap_or_default(),
                help_full,
                default: arg
                    .get_default_values()
                    .first()
                    .map(|v| v.to_string_lossy().into_owned()),
                options,
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "lib_fields_tests.rs"]
mod tests;
