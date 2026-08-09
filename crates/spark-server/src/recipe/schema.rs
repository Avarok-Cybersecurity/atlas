// SPDX-License-Identifier: AGPL-3.0-only

//! Recipe `defaults:` keys → `spark serve` flags.
//!
//! **clap stays the single source of truth for the flag surface.** `ServeArgs`
//! is not given a `Serialize` derive: `atlas-recipes` is a separate repo that
//! cannot be renamed atomically with this one, so making the CLI a public
//! serialization format would turn every future flag rename into a compat
//! break. Instead a recipe is converted to argv and handed back through
//! `ServeArgs::try_parse_from` — clap validates it exactly as if a person had
//! typed it, and a key this table gets wrong fails loudly at parse.
//!
//! Most keys are the field name with underscores swapped for dashes. The
//! exceptions below are real and were verified against `serve_args.rs`; each
//! one silently serves the wrong thing if it is dropped.

/// Keys whose recipe spelling differs from the flag.
///
/// Verified 2026-07-31 against `cli/serve_args.rs`: neither `max_model_len`
/// nor `tensor_parallel` exists as a field, and the listen address is `--bind`.
const RENAMES: &[(&str, &str)] = &[
    // vLLM's spelling, kept in the recipes for cross-runtime familiarity.
    ("max_model_len", "max-seq-len"),
    ("tensor_parallel", "tp-size"),
    ("host", "bind"),
];

/// `defaults:` keys that are not `spark serve` flags at all.
///
/// `port` IS a flag and is not listed here.
const NOT_FLAGS: &[&str] = &[];

/// The flag for a recipe key, or `None` if the key is not a flag.
pub fn flag_for(key: &str) -> Option<String> {
    if NOT_FLAGS.contains(&key) {
        return None;
    }
    if let Some((_, flag)) = RENAMES.iter().find(|(k, _)| *k == key) {
        return Some((*flag).to_string());
    }
    Some(key.replace('_', "-"))
}

/// Render one `key: value` pair as argv.
///
/// A `true` boolean becomes a bare `--flag`, because that is how clap's
/// `SetTrue` flags are written. A `false` boolean is **omitted rather than
/// negated**: `--enable-prefix-caching false` is not accepted by a `SetTrue`
/// flag, and emitting it would fail the parse for a recipe that is merely
/// restating a default.
///
/// The exception is a flag clap declares as taking an explicit bool value —
/// `--disable-tool-grammar` is `Option<bool>` — where `false` is meaningful and
/// must be passed through. Those are listed in `EXPLICIT_BOOLS`.
pub fn argv_for(key: &str, value: &str) -> Option<Vec<String>> {
    let flag = flag_for(key)?;
    let dashed = format!("--{flag}");
    match value {
        "true" if !EXPLICIT_BOOLS.contains(&key) => Some(vec![dashed]),
        "false" if !EXPLICIT_BOOLS.contains(&key) => None,
        other => Some(vec![dashed, other.to_string()]),
    }
}

/// Flags declared as `Option<bool>` in `ServeArgs`, which take a value.
const EXPLICIT_BOOLS: &[&str] = &["disable_tool_grammar"];

#[cfg(test)]
#[path = "schema_tests.rs"]
mod tests;
