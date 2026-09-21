// SPDX-License-Identifier: AGPL-3.0-only

//! What `spark serve --hermetic` closes, as data — so both crates read ONE
//! table.
//!
//! It lives here, in the lower crate, because two very different consumers
//! need it and neither may own it:
//!
//! * `spark-server`'s `cli::hermetic` EXPANDS a requested `hermetic=true` into
//!   these keys before a recipe renders, and enforces them in its resolvers.
//! * `gate::bench` REFUSES a BENCH.toml entry that pins `hermetic=true`
//!   without also pinning these, because `scoring::check_record` compares the
//!   record's serve overrides against the baseline's pins IN BOTH DIRECTIONS.
//!   A record carrying three keys against a baseline pinning one fails with
//!   "present on the record but not pinned by the baseline" — after the run,
//!   having spent the GPU hours.
//!
//! A copy in each crate would drift, and the failure mode of drift here is a
//! gate that cannot be discharged by the run it asks for.

/// The serve keys `--hermetic` forces, and the values it forces them to.
pub const CLOSED_KEYS: &[(&str, &str)] =
    &[("enable_prefix_caching", "false"), ("mtp_gate", "force")];

/// Does this override map request the hermetic regime?
///
/// Exact-token compare against `"true"`, not a truthiness test: `hermetic` is
/// a `--serve-override` value, so it arrives as a string, and anything that is
/// not the affirmative spelling must read as "not requested" rather than as
/// "probably yes".
pub fn is_requested(overrides: &std::collections::BTreeMap<String, String>) -> bool {
    overrides.get("hermetic").map(String::as_str) == Some("true")
}

/// Which keys `--hermetic` closes are missing from `overrides`, or present at
/// the wrong value. Empty means the set is complete and consistent.
pub fn missing_pins(
    overrides: &std::collections::BTreeMap<String, String>,
) -> Vec<(&'static str, &'static str)> {
    CLOSED_KEYS
        .iter()
        .filter(|(k, v)| overrides.get(*k).map(String::as_str) != Some(*v))
        .map(|(k, v)| (*k, *v))
        .collect()
}

#[cfg(test)]
#[path = "hermetic_tests.rs"]
mod hermetic_tests;
