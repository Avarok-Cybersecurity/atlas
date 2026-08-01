// SPDX-License-Identifier: AGPL-3.0-only

//! What to do about an error, as opposed to what the error was.
//!
//! `{"error":{"message":"no model is loaded","type":"model_not_loaded"}}` is
//! accurate and useless: it names a state without naming the way out of it.
//! Someone who does not already know the Library exists cannot act on it.
//!
//! This is the same idea as `cli::validate`'s `Violation { what, why, fix }`
//! — the repo already holds that a diagnostic without a `fix` is only half of
//! one — applied to the HTTP surface. (Not an intra-doc link: that module is
//! private, and linking to it from documented code is a rustdoc error here.)
//!
//! ## Why the hint is appended to `message` as well as emitted as `hint`
//!
//! Benchmarks live in `atlas-plugin`, and `spark-server` depends on
//! `atlas-plugin`, not the reverse — so the plugin crate cannot import this
//! table. Duplicating it there would put the same strings in two crates with
//! nothing keeping them in step. Instead the hint rides in `message`, which
//! every client already reads, so `atlas-plugin` gets it for free by surfacing
//! the body it was already receiving and discarding. `hint` is emitted
//! separately as well, for clients that want to render it distinctly.

/// The actionable half of an error, keyed by the stable `error.type` string.
///
/// Keyed on `type` rather than the status code because the code is far too
/// coarse: a 503 is "model still loading" and "no model chosen" and "shutting
/// down", and only the middle one is fixed by opening the Library.
pub fn hint_for(error_type: &str) -> Option<&'static str> {
    match error_type {
        // Each hint is the ACTION only, never a restatement of the condition:
        // it is appended to a message that has just said what is wrong, and
        // "no model is loaded — No model is loaded." is how you train a reader
        // to stop reading the second half.
        "model_not_loaded" => Some(
            "open the Library (press 4 in the dashboard), choose a model and a \
             recipe, and start it; then retry this request",
        ),
        "not_ready" => Some("the socket binds before the model finishes loading, so retry shortly"),
        "shutting_down" => Some("the server is draining and will not accept new work"),
        _ => None,
    }
}

/// `message`, with the hint for `error_type` appended when there is one.
///
/// Kept separate from [`hint_for`] so the joining rule — and in particular the
/// separator, which shows up in the middle of user-visible text — has exactly
/// one definition.
pub fn message_with_hint(message: &str, error_type: &str) -> String {
    match hint_for(error_type) {
        Some(h) => format!("{message} — {h}"),
        None => message.to_string(),
    }
}

// Reading an error body back out is `atlas_plugin::http`'s job, not this
// module's: the benchmark reader needs the identical rule including chunked
// de-framing, `spark-server` depends on that crate and not the reverse, so it
// is defined once there and called directly by both. See
// `atlas_plugin::http::{message_from_body, error_message_from_response}`.

#[cfg(test)]
#[path = "error_hints_tests.rs"]
mod tests;
