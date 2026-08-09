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

/// Load `recipe_id` if the request asked for a model that is not the live one.
///
/// Blocking, and long. Call it from `spawn_blocking`.
///
/// The re-check after taking the guard is what makes this single-flight rather
/// than a stampede: while a request waited, the winner may have loaded exactly
/// the model it wanted, and repeating the load would be a second multi-minute
/// outage for nothing.
pub(crate) fn ensure_loaded(
    host: &std::sync::Arc<super::model_host::ModelHost>,
    recipe_id: &str,
    requested_model: &str,
    catalogue: &[Recipe],
) -> anyhow::Result<()> {
    // The guard lives in `model_swap::swap` so every caller is covered, and it
    // re-checks there — taking it here too would deadlock on a non-reentrant
    // mutex. This early exit is only an optimisation for the uncontended case.
    if host.live_model().as_deref() == Some(requested_model) {
        return Ok(());
    }
    let recipe = catalogue
        .iter()
        .find(|r| r.id == recipe_id)
        .ok_or_else(|| anyhow::anyhow!("recipe {recipe_id} vanished between decide and load"))?;
    let args = recipe.serve_args(&std::collections::BTreeMap::new())?;
    super::model_swap::swap(host, args)?;
    Ok(())
}

#[cfg(test)]
#[path = "auto_swap_tests.rs"]
mod tests;
