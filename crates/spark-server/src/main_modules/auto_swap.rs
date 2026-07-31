// SPDX-License-Identifier: AGPL-3.0-only

//! Ollama-style auto-swap: a request naming a different known model loads it.
//!
//! **Deliberately narrow.** Clients send arbitrary strings in `model` — the
//! benchmark harness sends whatever `--model` was typed, and Atlas has always
//! answered regardless (`lora_control.rs`: any unknown name falls through to
//! the installed adapter, never a 400). Turning a cosmetic mismatch into an
//! error would break every existing caller, and turning it into a swap would
//! make a typo a multi-minute outage.
//!
//! So only one case acts:
//!
//! | request `model`                     | action                        |
//! |-------------------------------------|-------------------------------|
//! | absent / empty                      | ignore — serve current        |
//! | not resolvable to a known recipe    | ignore — serve current        |
//! | resolves to the model already live  | ignore — no swap              |
//! | resolves to a DIFFERENT known model | swap, then serve              |
//!
//! And it is off unless `--auto-swap` is passed: even narrowed to known models,
//! one stray request is a multi-minute outage for every other client on the
//! box, and a benchmark sweep naming a sibling checkpoint would swap mid-run.

use crate::recipe::Recipe;

/// What a request's `model` field asks of the server.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    /// Serve on the model already loaded.
    ServeCurrent,
    /// Load this recipe first. Carries the recipe id, not the raw request
    /// string, so the caller launches something that was actually validated.
    SwapTo(String),
}

/// Is request-triggered swapping permitted for this deployment?
///
/// Deny wins. See `--no-auto-swap` for why that is not a clap conflict.
pub(crate) fn enabled(args: &crate::cli::ServeArgs) -> bool {
    args.auto_swap && !args.no_auto_swap
}

/// Decide what to do about `requested`, given what is live and what is known.
///
/// Matching is by exact HF id: a recipe's `model` field is the id the server
/// would be started with, and a fuzzy match here would swap to something the
/// caller did not ask for.
pub(crate) fn decide(requested: &str, live_model: &str, catalogue: &[Recipe]) -> Decision {
    let requested = requested.trim();
    if requested.is_empty() || requested == live_model {
        return Decision::ServeCurrent;
    }
    match catalogue
        .iter()
        .filter(|r| r.is_atlas())
        .find(|r| r.model == requested)
    {
        // A known model, and not the one running.
        Some(recipe) => Decision::SwapTo(recipe.id.clone()),
        // Unknown: serve on what is loaded, exactly as before this existed.
        None => Decision::ServeCurrent,
    }
}

#[cfg(test)]
#[path = "auto_swap_tests.rs"]
mod tests;
