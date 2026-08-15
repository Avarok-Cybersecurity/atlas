// SPDX-License-Identifier: AGPL-3.0-only

//! Pure resolution for a gate run: which (model variant, recipe) to serve,
//! which recipe keys the operator overrode, and which run parameters the
//! selected variant's baseline defines.
//!
//! Split from [`super::bench_selfstart`] (exact piecewise copy) so the
//! branching — box class, model variant, recipe binding, threshold coupling —
//! is testable without a GPU and readable without the serve/teardown
//! machinery around it. Nothing here starts anything.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use atlas_plugin::gate;

/// What a baseline says to serve.
#[derive(Debug)]
pub(super) struct Resolved {
    pub model: String,
    pub recipe_id: String,
    /// The resolved variant's thresholds/note/label, verbatim.
    pub entry: gate::ModelBaseline,
}

/// Pick the (model, recipe) a gate run should serve.
///
/// Split out from the serving itself so the branching — which box class, which
/// model variant, and whether a recipe is bound at all — is testable without a
/// GPU. Every refusal names both what was asked for and what exists; an
/// unresolvable baseline must never read as "nothing to serve".
///
/// `checkpoint` selects the model variant. `None` takes the one the baseline
/// marks `default = true` — a committed declaration, not a guess (assembly
/// refuses zero or two defaults outright) — and a checkpoint the baseline does
/// not carry is refused naming what exists, exactly as `--hardware` behaves on
/// its axis.
pub(super) fn resolve(
    baseline: &gate::GateBaseline,
    benchmark_id: &str,
    hardware: Option<&str>,
    checkpoint: Option<&str>,
) -> Result<Resolved> {
    let hw_key = match hardware {
        Some(h) => h.to_string(),
        None => {
            let mut keys = baseline.hardware.keys();
            match (keys.next(), keys.next()) {
                (Some(only), None) => only.clone(),
                // Two box classes and no instruction is not a coin flip: TTFT
                // ceilings differ per box, so guessing would score the run
                // against another machine's numbers.
                (Some(_), Some(_)) => bail!(
                    "{benchmark_id} has baselines for several box classes ([{}]); pass \
                     --hardware to say which one this run is for rather than guessing",
                    baseline
                        .hardware
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                (None, _) => bail!("{benchmark_id} has no hardware entries in its baseline"),
            }
        }
    };

    let (model, entry) = baseline.resolve(&hw_key, checkpoint)?;
    let recipe_id = entry.recipe.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "no recipe is bound to {model:?} on {hw_key:?} for {benchmark_id}. Self-start needs \
             one; either add `recipe` to the baseline entry or drive an existing server with \
             --url/--model and no --pull-request-gate."
        )
    })?;
    Ok(Resolved {
        model,
        recipe_id,
        entry: entry.clone(),
    })
}

/// Derive the run's baseline-coupled parameters from the SELECTED variant.
///
/// A benchmark may compute its own verdict against a knob that is also a
/// committed threshold — `BenchmarkDescriptor::threshold_params` declares the
/// pairs. The schema default can only be right for one variant (the agentic
/// Σ-wall default is the 35B's 1000 s ceiling, and the dense 27B's band is
/// roughly 2× that), so under the gate the value comes from the variant's own
/// `BENCH.toml` bound. Precedence is explicit and narrow:
///
/// 1. an operator's `--param KEY=…` wins untouched — stated intent;
/// 2. otherwise a `max` bound on the paired metric replaces the default;
/// 3. a variant with no such bound leaves the schema default standing.
///
/// Returns what was applied so the caller can PRINT it — a run whose effective
/// budget differs from the schema default must say where the number came from.
/// Every applied value still lands in the record's `params`, so the record
/// stays self-describing.
pub(super) fn apply_threshold_params(
    descriptor: &atlas_plugin::BenchmarkDescriptor,
    specs: &[atlas_plugin::ParamSpec],
    values: &mut atlas_plugin::ParamValues,
    entry: &gate::ModelBaseline,
    explicit: &[(String, String)],
) -> Result<Vec<(String, f64)>> {
    let mut applied = Vec::new();
    for (param, metric) in descriptor.threshold_params {
        if explicit.iter().any(|(k, _)| k == param) {
            continue;
        }
        let Some(max) = entry.metrics.get(*metric).and_then(|b| b.max) else {
            continue;
        };
        let spec = specs.iter().find(|s| s.key == *param).ok_or_else(|| {
            anyhow::anyhow!(
                "{} declares threshold param {param:?} but its schema has no such parameter — \
                 the declaration and the schema have drifted",
                descriptor.id
            )
        })?;
        // Through the spec's own parser, so the kind (and its bounds) cannot
        // be bypassed by this path any more than by a typed --param.
        let value = spec.kind.parse(&format!("{max}")).with_context(|| {
            format!("deriving --param {param} from the baseline's {metric} bound {max}")
        })?;
        values.set(param.to_string(), value);
        applied.push((param.to_string(), max));
    }
    Ok(applied)
}

/// Parse `--serve-override KEY=VALUE` pairs into recipe overrides.
///
/// Only splits and validates — whether the KEY exists is `Recipe::argv`'s
/// question, and it already refuses an unknown one, so re-checking here would
/// be a second copy of that rule.
///
/// `port` is refused: `serve_for` picks a free port and passes its own, so a
/// second opinion would either be silently dropped or race whatever else holds
/// the operator's port. Saying so beats both.
pub(super) fn parse_serve_overrides(pairs: &[String]) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for pair in pairs {
        let (key, value) = pair.split_once('=').with_context(|| {
            format!("--serve-override {pair:?} is not KEY=VALUE (e.g. kv_cache_dtype=fp8)")
        })?;
        let key = key.trim();
        ensure!(
            !key.is_empty(),
            "--serve-override {pair:?} has an empty key"
        );
        ensure!(
            key != "port",
            "--serve-override cannot set `port`: the gate binds a free port itself and serves \
             on it, so an override here would name a port nothing is listening on."
        );
        // Last wins, deliberately: repeating a key is how you edit a long
        // command line, and silently keeping the FIRST would contradict every
        // other CLI on the box.
        out.insert(key.to_string(), value.to_string());
    }
    Ok(out)
}

#[cfg(test)]
#[path = "bench_resolve_tests.rs"]
mod tests;
