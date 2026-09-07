// SPDX-License-Identifier: AGPL-3.0-only

//! Parse and validate `usage.expert_activation` off one response.
//!
//! Everything here fails LOUD. This benchmark's output is a table that
//! decides which experts a category gets loaded with, so a response that was
//! not actually instrumented, or whose counts do not add up, must stop the
//! run rather than quietly contribute nothing to an average.

use anyhow::{Result, bail};
use serde_json::Value;

/// One layer's routing from one response.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerActivation {
    pub layer: usize,
    /// `(expert_id, count, mass)`, expert ids ascending.
    pub experts: Vec<(u32, u32, f64)>,
}

/// One response's whole report.
#[derive(Debug, Clone, PartialEq)]
pub struct Activation {
    pub top_k: u32,
    pub num_experts: u32,
    pub tokens_routed: u64,
    pub unattributed_rows: u64,
    pub layers: Vec<LayerActivation>,
}

/// What to tell the operator when a response carries no report.
///
/// Split from the parse error because the two have different causes and
/// different fixes, and the whole point of the server's absent-vs-empty
/// contract is that the client can tell them apart.
pub fn missing_report_error(prompt_id: &str) -> anyhow::Error {
    anyhow::anyhow!(
        "response for prompt '{prompt_id}' carries no usage.expert_activation.\n\
         WHAT: this benchmark reads per-layer MoE routing off every response.\n\
         WHY: the serve was started without --expert-telemetry, or something \
         upstream stripped the usage extension.\n\
         FIX: restart the serve with --expert-telemetry (an MoE checkpoint is \
         required; a dense model is refused per-request with a 400)."
    )
}

/// Parse one response's `usage.expert_activation`.
///
/// Validates the invariants that separate a real measurement from a broken
/// instrument. `prompt_id` names the offending row so a failure is
/// actionable rather than a bare shape complaint.
pub fn parse(v: &Value, prompt_id: &str) -> Result<Activation> {
    let ctx = format!("expert_activation for prompt '{prompt_id}'");
    let obj = v
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{ctx}: expected an object, got {v}"))?;

    let u64_field = |name: &str| -> Result<u64> {
        obj.get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("{ctx}: missing or non-integer `{name}`"))
    };
    let top_k = u64_field("top_k")? as u32;
    let num_experts = u64_field("num_experts")? as u32;
    let tokens_routed = u64_field("tokens_routed")?;
    let unattributed_rows = u64_field("unattributed_rows")?;

    if top_k == 0 || num_experts == 0 {
        bail!("{ctx}: top_k={top_k} num_experts={num_experts} — not an MoE report");
    }

    let layers_val = obj
        .get("layers")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("{ctx}: missing `layers` array"))?;

    // Telemetry on, but nothing recorded. Distinct from the field being
    // absent: that is "not instrumented", this is "instrumented and the
    // model routed nothing", which for an MoE means a dense checkpoint or a
    // tap that did not fire. Either way it is not a measurement.
    if layers_val.is_empty() {
        bail!(
            "{ctx}: the report is present but has no layers — the serve is \
             instrumented yet recorded no routing. Either the model is dense \
             (no MoE layers to observe) or the prefill path taken by this \
             prompt is not instrumented. Not averaging a zero into the result."
        );
    }

    let mut layers = Vec::with_capacity(layers_val.len());
    let mut total_count: u64 = 0;
    for l in layers_val {
        let layer = l
            .get("layer")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("{ctx}: a layer entry has no `layer` index"))?
            as usize;
        let arr = |name: &str| -> Result<&Vec<Value>> {
            l.get(name)
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow::anyhow!("{ctx}: layer {layer} has no `{name}` array"))
        };
        let (ids, counts, mass) = (arr("experts")?, arr("counts")?, arr("mass")?);
        if ids.len() != counts.len() || ids.len() != mass.len() {
            bail!(
                "{ctx}: layer {layer} has mismatched arrays (experts {}, counts {}, mass {}) \
                 — they are parallel by contract",
                ids.len(),
                counts.len(),
                mass.len()
            );
        }
        let mut experts = Vec::with_capacity(ids.len());
        let mut prev: Option<u32> = None;
        for i in 0..ids.len() {
            let id = ids[i]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("{ctx}: layer {layer} has a non-integer id"))?
                as u32;
            if id >= num_experts {
                bail!("{ctx}: layer {layer} reports expert {id} of {num_experts}");
            }
            // Ascending order is the contract the aggregator and the emitted
            // TOML both rely on; a violation means the two ends disagree
            // about what the arrays are.
            if let Some(p) = prev
                && id <= p
            {
                bail!("{ctx}: layer {layer} expert ids are not strictly ascending ({p} then {id})");
            }
            prev = Some(id);
            let count = counts[i]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("{ctx}: layer {layer} has a non-integer count"))?;
            let m = mass[i]
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("{ctx}: layer {layer} has a non-numeric mass"))?;
            if !m.is_finite() || m < 0.0 {
                bail!("{ctx}: layer {layer} expert {id} has mass {m}");
            }
            total_count += count;
            experts.push((id, count as u32, m));
        }
        layers.push(LayerActivation { layer, experts });
    }

    // The vacuity pin. Every routed token position picks exactly `top_k`
    // experts, so the counts and the token total are two views of the same
    // thing. They drift apart only if the tap double-counted, dropped rows
    // silently, or reported a token total it did not measure — and a
    // category table built on any of those is wrong in a way no downstream
    // check would catch.
    let expected = tokens_routed.saturating_mul(u64::from(top_k));
    if total_count != expected {
        bail!(
            "{ctx}: Σcounts = {total_count} but tokens_routed × top_k = {expected}. \
             The routing report is internally inconsistent; refusing to fold it."
        );
    }

    Ok(Activation {
        top_k,
        num_experts,
        tokens_routed,
        unattributed_rows,
        layers,
    })
}

#[cfg(test)]
#[path = "usage_tests.rs"]
mod tests;
