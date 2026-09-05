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

/// Why a `--hardware` value did not land on a baseline slot.
///
/// Two refusals, because they are two different jobs for the operator, and a
/// single message sends one of them to the wrong one:
///
/// * [`Self::Unknown`] — the id names no box class Atlas recognises. No run
///   will ever fix it; the spelling is wrong (or the class needs registering
///   in `atlas_plugin::hardware::ids::KNOWN_HARDWARE_IDS` first).
/// * [`Self::NoRecordYet`] — the id is registered and nothing has been
///   measured on it. The spelling is right; the fix is to run the gate on that
///   box and commit the thresholds.
///
/// Before this split the second case read as the first. That is exactly the
/// state a hardware port is in for its whole duration — the id is real, the
/// records are not — so the campaign would spend that whole span being told,
/// wrongly, that it had typed the box class in wrong.
#[derive(Debug, thiserror::Error)]
pub(super) enum HardwareRefusal {
    #[error(
        "{hardware:?} is not a box class Atlas knows, so nothing can be scored against it. \
         Registered classes are [{registered}]; {benchmark_id} has baselines for [{measured}]."
    )]
    Unknown {
        benchmark_id: String,
        hardware: String,
        registered: String,
        measured: String,
    },
    #[error(
        "{benchmark_id} has no record yet for {hardware:?}. It is a registered box class with \
         nothing measured on it — run the gate on that box and commit one \
         (`spark benchmark run {benchmark_id} --hardware {hardware} --pull-request-gate`), \
         which needs a `[[benchmark]]` entry in kernels/{hardware}/<model>/BENCH.toml. \
         Today it has baselines for [{measured}]."
    )]
    NoRecordYet {
        benchmark_id: String,
        hardware: String,
        measured: String,
    },
}

impl HardwareRefusal {
    /// Classify a `hardware` key the baseline does not carry.
    ///
    /// The caller has already established the key is absent; this only decides
    /// WHICH absence it is, against the registry rather than the baseline.
    fn of(benchmark_id: &str, hardware: &str, baseline: &gate::GateBaseline) -> Self {
        let measured = baseline
            .hardware
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        if atlas_plugin::hardware::ids::is_known_hardware_id(hardware) {
            Self::NoRecordYet {
                benchmark_id: benchmark_id.to_string(),
                hardware: hardware.to_string(),
                measured,
            }
        } else {
            Self::Unknown {
                benchmark_id: benchmark_id.to_string(),
                hardware: hardware.to_string(),
                registered: atlas_plugin::hardware::ids::KNOWN_HARDWARE_IDS.join(", "),
                measured,
            }
        }
    }
}

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
/// A `hardware` the baseline does not carry is classified against the box-class
/// registry before it is refused — see [`HardwareRefusal`]: a registered class
/// with no records ("run it and commit one") is a different instruction than an
/// id Atlas does not know ("fix the spelling").
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

    // Classified BEFORE the baseline lookup, because the baseline can only
    // report what has been measured. See [`HardwareRefusal`].
    if !baseline.hardware.contains_key(&hw_key) {
        return Err(HardwareRefusal::of(benchmark_id, &hw_key, baseline).into());
    }
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
/// 2. otherwise the paired metric's bound replaces the default: `max` if the
///    baseline declares one (a ceiling like the agentic Σ-wall), else `min`
///    (a floor like the BFCL accuracies or the decode-rate floor). A paired
///    metric carrying BOTH bounds is ambiguous — which one the driver should
///    self-verdict against cannot be inferred — so that is a loud error, not
///    a guess;
/// 3. a variant with no such bound leaves the schema default standing.
///
/// Returns what was applied so the caller can PRINT it — a run whose effective
/// budget differs from the schema default must say where the number came from.
/// Every applied value still lands in the record's `params`, so the record
/// stays self-describing.
///
/// ★ `bench_variants::BenchState::choose_variant` (TUI) carries a textually
/// parallel copy of this bound selection — keep the two in step.
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
        let Some(bound) = entry.metrics.get(*metric) else {
            continue;
        };
        // The driver compares RAW values against the bar it is handed, while
        // the gate judge (`gate::scoring::compare`) allows the bound's noise
        // band (pass iff value + noise >= min, value - noise <= max). Hand the
        // driver the noise-adjusted bar, or a run inside the band passes the
        // gate and FAILs its own verdict — which the record then carries, and
        // CI requires verdict == PASS (bfcl-subset-echolp, 2026-08-16:
        // 86.35 vs min 86.50, noise 0.4 — gate pass, self-verdict fail).
        let noise = bound.noise.unwrap_or(0.0);
        let derived = match (bound.min, bound.max) {
            (Some(min), Some(max)) => bail!(
                "{} couples param {param:?} to metric {metric:?}, whose baseline bound \
                 declares BOTH min ({min}) and max ({max}) — ambiguous: the driver cannot \
                 tell which one to self-verdict against. Split the metric or drop a bound.",
                descriptor.id
            ),
            (None, Some(max)) => max + noise,
            (Some(min), None) => min - noise,
            (None, None) => continue,
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
        let value = spec.kind.parse(&format!("{derived}")).with_context(|| {
            format!("deriving --param {param} from the baseline's {metric} bound {derived}")
        })?;
        values.set(param.to_string(), value);
        applied.push((param.to_string(), derived));
    }
    Ok(applied)
}

/// Apply the selected variant's `[benchmarks.param_overrides]` pins — the
/// request-side sibling of `serve_overrides`, and the mechanism that lets a
/// gate's thresholds be calibrated on a NON-default instrument.
///
/// The concurrency gate is the motivating case: its committed floors were
/// measured on the C=1/4/8/16 ladder at isl 512 / osl 320, while the schema
/// defaults sweep C=1..32 at osl 128 — a gate run on the defaults would score
/// a different instrument against these thresholds. So the baseline entry pins
/// the parameters, and self-start applies them here. Precedence mirrors
/// [`apply_threshold_params`]: an operator's `--param KEY=…` wins untouched;
/// otherwise the pin replaces the schema default, routed through the spec's
/// own parser so the kind's bounds cannot be bypassed.
///
/// Two loud refusals rather than guesses:
/// * a pin naming a `threshold_params`-coupled parameter — that value comes
///   from the paired metric's bound, and a second source here would silently
///   fight it;
/// * a pin naming no schema parameter at all — the BENCH.toml and the driver
///   have drifted, and a silently-dropped pin runs the wrong instrument.
///
/// Returns what was applied so the caller can PRINT it; every applied value
/// also lands in the record's `params` (defaults included), and
/// `check_record` demands the pin on the record — so a record measured
/// without the pin cannot read green against the pinned thresholds.
pub(super) fn apply_param_overrides(
    descriptor: &atlas_plugin::BenchmarkDescriptor,
    specs: &[atlas_plugin::ParamSpec],
    values: &mut atlas_plugin::ParamValues,
    entry: &gate::ModelBaseline,
    explicit: &[(String, String)],
) -> Result<Vec<(String, String)>> {
    let mut applied = Vec::new();
    for (key, raw) in &entry.param_overrides {
        if descriptor.threshold_params.iter().any(|(p, _)| p == key) {
            bail!(
                "{}: baseline param override {key:?} names a threshold-coupled parameter — \
                 its value is derived from the paired metric's bound, and a second source \
                 here would fight it. Move the number into the metric's bound instead.",
                descriptor.id
            );
        }
        if explicit.iter().any(|(k, _)| k == key) {
            continue; // stated intent wins, exactly as for threshold params
        }
        let spec = specs
            .iter()
            .find(|s| s.key == key.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{}: baseline param override {key:?} names no parameter in the schema — \
                 the BENCH.toml pin and the driver have drifted, and running without it \
                 would measure a different instrument than the thresholds describe",
                    descriptor.id
                )
            })?;
        let value = spec
            .kind
            .parse(raw)
            .with_context(|| format!("applying the baseline's param override {key}={raw}"))?;
        values.set(key.clone(), value);
        applied.push((key.clone(), raw.clone()));
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

#[cfg(test)]
#[path = "bench_resolve_hardware_tests.rs"]
mod hardware_tests;

#[cfg(test)]
#[path = "bench_resolve_params_tests.rs"]
mod params_tests;
