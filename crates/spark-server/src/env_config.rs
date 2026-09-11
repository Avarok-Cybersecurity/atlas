// SPDX-License-Identifier: AGPL-3.0-only

//! Strict parsing for the process-scoped `ATLAS_*` configuration variables.
//!
//! ## What this exists to stop
//!
//! The process-scoped config — rate limits, the response store, the
//! conversation store — was read like this:
//!
//! ```ignore
//! let rpm = std::env::var("ATLAS_RATE_LIMIT_RPM")
//!     .ok()
//!     .and_then(|s| s.parse().ok())   // ← a typo lands here
//!     .unwrap_or(0);                  // ← and silently becomes "off"
//! ```
//!
//! `ATLAS_RATE_LIMIT_RPM=1oo` (letter o) parses as nothing, falls through to
//! the default, and the default for a rate limit is **0, which means the limit
//! is not enforced at all**. The operator set a limit, the server started
//! cleanly, printed nothing, and served unlimited. Every variable in this
//! family had the same shape: `ATLAS_STORE_TTL_SECONDS=1h` is a 24-hour TTL,
//! `ATLAS_CONVERSATION_MAX_ENTRIES=10_000` (the spelling the doc comment uses!)
//! is the default 10 000 by luck rather than by parse.
//!
//! This is the repo's PCND rule — production code must not silently default; it
//! must require explicit config or fail fast naming the key — and the repo
//! already applies it elsewhere: `ATLAS_VISION_MAX_PIXELS` hard-errors with
//! "must be a positive integer, got …". These variables did not.
//!
//! ## Shape
//!
//! [`parse_min`] is pure — it takes the raw value rather than reading the
//! environment — so the decision is separable from the I/O (SBIO) and testable
//! without `set_var`, which is process-global and races every other test in the
//! binary. Each `from_env` does the reading and hands the strings here.
//!
//! Empty and whitespace-only are treated as unset, not as errors: exporting
//! `ATLAS_STORE_DIR=` to mean "off" is an established habit, and the previous
//! code already fell back for them.

use std::fmt::Display;
use std::str::FromStr;

/// Parse an optional numeric override, refusing a malformed or out-of-range
/// value instead of silently substituting the default.
///
/// `min` is the smallest value that means anything for this key; `meaning`
/// describes what the key controls and is quoted back in the error, because
/// "invalid value" without saying what a valid one would be leaves the reader
/// exactly where they started.
///
/// Returns `Ok(None)` when the variable is unset or blank — the caller applies
/// its own documented default, which is the one case where defaulting is right
/// because nobody asked for anything else.
pub fn parse_min<T>(
    key: &str,
    raw: Option<&str>,
    min: T,
    meaning: &str,
) -> Result<Option<T>, String>
where
    T: FromStr + PartialOrd + Display + Copy,
{
    let Some(raw) = raw else { return Ok(None) };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let parsed: T = trimmed
        .parse()
        .map_err(|_| describe(key, raw, min, meaning, "is not a whole number"))?;
    if parsed < min {
        return Err(describe(
            key,
            raw,
            min,
            meaning,
            "is below the smallest value this setting accepts",
        ));
    }
    Ok(Some(parsed))
}

/// The one place the wording of these errors is decided.
///
/// Shaped like `cli::validate`'s `Violation` — what, why, fix — because the
/// repo already holds that a diagnostic without a `fix` is half of one, and an
/// operator reading this has a shell open and wants to know what to type.
fn describe<T: Display>(key: &str, raw: &str, min: T, meaning: &str, problem: &str) -> String {
    format!(
        "{key}={raw:?} {problem}.\n      \
         why: {meaning} — expected a whole number >= {min}.\n      \
         fix: correct the value, or unset {key} to use the built-in default. \
         It is NOT ignored: the server refuses to start rather than serve a \
         configuration you did not ask for."
    )
}

#[cfg(test)]
#[path = "env_config_tests.rs"]
mod tests;
