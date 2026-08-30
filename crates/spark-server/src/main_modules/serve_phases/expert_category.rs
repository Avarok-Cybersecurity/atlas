// SPDX-License-Identifier: AGPL-3.0-only

//! Resolve `--expert-category <name>` into a boot-time expert-loading plan.
//!
//! The category table is compiled into the binary from the model's
//! MODEL.toml (`atlas_kernels::TargetPtxSet::expert_categories`), so this is
//! a lookup, not a file read — a runtime box does not ship the kernels tree.
//!
//! Everything here fails fast and by name. Loading the wrong expert subset
//! does not produce an error later; it produces a serve that answers
//! slightly worse, forever, and the only moment that is detectable is now.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use atlas_core::config::bel::BelPlan;

/// Build the plan and install it on `config`, or leave it `None`.
///
/// Called after `ptx_for_config` (the table comes from there) and before the
/// weight load (which reads the plan to decide what to skip).
pub(crate) fn resolve_expert_category(
    requested: Option<&str>,
    ptx_set: &atlas_kernels::TargetPtxSet,
    config: &mut ModelConfig,
) -> Result<()> {
    let Some(name) = requested else {
        return Ok(());
    };

    if config.num_experts == 0 {
        bail!(
            "--expert-category {name} was given, but this is a dense checkpoint with no \
             mixture-of-experts layers — there are no experts to select. Drop the flag."
        );
    }

    let available: Vec<&str> = ptx_set.expert_categories.iter().map(|c| c.name).collect();
    let Some(cat) = ptx_set.expert_categories.iter().find(|c| c.name == name) else {
        if available.is_empty() {
            bail!(
                "--expert-category {name}: this model has no [expert_categories] table. Run \
                 the `expert-categories` benchmark against it (with the serve started with \
                 --expert-telemetry), paste the emitted block into the model's MODEL.toml, \
                 and rebuild — the table is read at build time."
            );
        }
        bail!(
            "--expert-category {name}: unknown category. This model's MODEL.toml declares: {}",
            available.join(", ")
        );
    };

    // The table lists only MoE layers, and the model may be hybrid, so a
    // layer's absence is not by itself an error. What IS an error is a layer
    // that keeps fewer experts than top-k selects: the router would have to
    // name a masked expert to fill its k slots.
    for (layer, ids) in cat.layers {
        if ids.len() < config.num_experts_per_tok {
            bail!(
                "--expert-category {name}: layer {layer} keeps only {} experts but the \
                 router selects {} per token, so top-k could not be filled from the loaded \
                 set. Re-run the expert-categories benchmark at a higher coverage.",
                ids.len(),
                config.num_experts_per_tok,
            );
        }
    }

    let plan = BelPlan::new(
        cat.name,
        cat.coverage,
        config.num_hidden_layers,
        config.num_experts,
        cat.layers.iter().map(|(l, ids)| (*l, ids.to_vec())),
    )
    .map_err(anyhow::Error::msg)?;

    let (resident, total) = plan.totals();
    let layers = plan.restricted_layers();
    tracing::info!(
        "Expert category \"{}\" (coverage {:.2}): {resident} of {total} routed experts across \
         {} layers — {:.0}% of routed-expert weights",
        plan.category,
        plan.coverage,
        layers.len(),
        if total == 0 {
            100.0
        } else {
            100.0 * resident as f64 / total as f64
        },
    );
    // Per layer, at debug: the summary above is the number an operator wants,
    // but "which experts, in which layer" is what makes a surprising answer
    // diagnosable.
    for layer in layers {
        if let Some(n) = plan.layer_count(layer) {
            tracing::debug!(
                "  layer {layer}: {n} of {} experts resident",
                plan.num_experts()
            );
        }
    }

    config.bel = Some(std::sync::Arc::new(plan));
    Ok(())
}

#[cfg(test)]
#[path = "expert_category_tests.rs"]
mod tests;
