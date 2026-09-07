// SPDX-License-Identifier: AGPL-3.0-only

//! Is this expert's weight actually in the store?
//!
//! Two independent things can withhold a routed expert's tensors:
//!
//!  * **EP sharding** — the expert belongs to another rank.
//!  * **`--expert-category`** — the category never routes to it.
//!
//! The loaders answer per TENSOR NAME; the model-side loops answer per
//! `(prefix, expert)`. They must agree exactly: an expert the loader skipped
//! but the loop tries to read fails the load with "tensor not found", and an
//! expert the loop nulls but the loader kept wastes the memory the flag
//! exists to save.
//!
//! So this asks the question the same way the loader did — by building the
//! tensor name and parsing it back — rather than threading a layer index
//! through a dozen loader signatures where it could drift from the prefix
//! the names are actually built from.

use atlas_core::config::ModelConfig;

/// Whether expert `e` under `prefix` (e.g. `model.layers.3.mlp`) has weights
/// in this process.
pub(crate) fn expert_resident(config: &ModelConfig, prefix: &str, e: usize) -> bool {
    if !config.is_local_expert(e) {
        return false;
    }
    let Some(plan) = config.bel.as_ref() else {
        return true;
    };
    // The same name shape the loaders' skip rule parsed. A prefix it cannot
    // parse is one the loader could not parse either, so nothing was skipped
    // and everything is resident — the direction that fails as a loud
    // "tensor not found" rather than a silently nulled expert.
    let probe = format!("{prefix}.experts.{e}.gate_proj.weight");
    match spark_runtime::weights::parse_layer_expert(&probe) {
        Some((layer, expert)) => plan.is_loaded(layer, expert),
        None => true,
    }
}

#[cfg(test)]
#[path = "resident_tests.rs"]
mod tests;
