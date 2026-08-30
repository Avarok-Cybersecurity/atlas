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
use atlas_core::config::bel::{BelPlan, CategorySource};

/// Build the plan and install it on `config`, or leave it `None`.
///
/// `requested` is one category or a comma-separated list. A list loads the
/// UNION of those categories' experts: the serve cannot know which category a
/// request belongs to, so handling several means holding what any of them
/// needs.
///
/// Called after `ptx_for_config` (the table comes from there) and before the
/// weight load (which reads the plan to decide what to skip).
pub(crate) fn resolve_expert_category(
    requested: Option<&str>,
    ptx_set: &atlas_kernels::TargetPtxSet,
    config: &mut ModelConfig,
) -> Result<()> {
    let Some(spec) = requested else {
        return Ok(());
    };

    let names: Vec<&str> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if names.is_empty() {
        bail!("--expert-category was given an empty list");
    }
    // A repeat is almost certainly a typo in a longer list, and silently
    // deduplicating it would hide which name was meant.
    for (i, n) in names.iter().enumerate() {
        if names[..i].contains(n) {
            bail!("--expert-category lists '{n}' more than once");
        }
    }

    if config.num_experts == 0 {
        bail!(
            "--expert-category {spec} was given, but this is a dense checkpoint with no \
             mixture-of-experts layers — there are no experts to select. Drop the flag."
        );
    }

    let available: Vec<&str> = ptx_set.expert_categories.iter().map(|c| c.name).collect();
    let mut sources = Vec::with_capacity(names.len());
    for name in &names {
        let Some(cat) = ptx_set.expert_categories.iter().find(|c| &c.name == name) else {
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
        sources.push(CategorySource {
            name: cat.name.to_string(),
            coverage: cat.coverage,
            layers: cat
                .layers
                .iter()
                .map(|(l, ids)| (*l, ids.to_vec()))
                .collect(),
        });
    }

    let plan = BelPlan::from_sources(sources, config.num_hidden_layers, config.num_experts)
        .map_err(anyhow::Error::msg)?;

    // The table lists only MoE layers, and the model may be hybrid, so a
    // layer's absence is not by itself an error. What IS an error is a layer
    // whose UNION keeps fewer experts than top-k selects: the router would
    // have to name a masked expert to fill its k slots. Checked on the union,
    // not per source, because the union is what gets loaded.
    for layer in plan.restricted_layers() {
        let kept = plan.layer_count(layer).unwrap_or(0);
        if kept < config.num_experts_per_tok {
            bail!(
                "--expert-category {spec}: layer {layer} keeps only {kept} experts but the \
                 router selects {} per token, so top-k could not be filled from the loaded \
                 set. Re-run the expert-categories benchmark at a higher coverage.",
                config.num_experts_per_tok,
            );
        }
    }

    let (resident, total) = plan.totals();
    let layers = plan.restricted_layers();
    let coverage = match plan.uniform_coverage() {
        Some(c) => format!("coverage {c:.2}"),
        None => "mixed coverage".to_string(),
    };
    tracing::info!(
        "Expert {} \"{}\" ({coverage}): {resident} of {total} routed experts across {} layers \
         — {:.0}% of routed-expert weights",
        if plan.sources.len() == 1 {
            "category"
        } else {
            "categories (union)"
        },
        plan.label(),
        layers.len(),
        if total == 0 {
            100.0
        } else {
            100.0 * resident as f64 / total as f64
        },
    );
    // Per source, when there is more than one: the union's size alone does not
    // say whether adding a category cost ten experts or a thousand.
    if plan.sources.len() > 1 {
        for src in &plan.sources {
            let own = BelPlan::from_sources(
                vec![src.clone()],
                config.num_hidden_layers,
                config.num_experts,
            )
            .map_err(anyhow::Error::msg)?;
            let (r, t) = own.totals();
            tracing::info!(
                "  \"{}\" alone (coverage {:.2}): {r} of {t} routed experts",
                src.name,
                src.coverage
            );
        }
    }
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
