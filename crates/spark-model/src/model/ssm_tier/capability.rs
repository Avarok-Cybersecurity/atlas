// SPDX-License-Identifier: AGPL-3.0-only

//! Capability gate: selecting a tier the model cannot populate is a STARTUP
//! ERROR, never a silent no-op.
//!
//! "Model-agnostic" does NOT mean every model has every tier. The SSM
//! snapshot tiers (Marconi spill, decode rollback ring) hold recurrent
//! state that only exists on hybrid SSM+attention models
//! ([`ModelConfig::has_recurrent_state`]); a pure-attention model — dense or
//! MoE — can never populate them. Before this gate, an SSM tier env var on
//! such a model was swallowed silently (the `num_ssm_layers > 0` arms in
//! `impl_a1` just skipped construction WITHOUT reading the vars, hiding even
//! hard misconfigurations like `ATLAS_SSM_DECODE_TIER=nvme` with no dir).
//! "Works on Holo, mysteriously does nothing on X" is exactly the failure
//! mode PCND forbids: require explicit config or fail fast.
//!
//! When the capability IS present — or when no tier var is set at all — this
//! gate reads env *presence* only and changes nothing: the byte-identical
//! default path is preserved.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;

/// Every env var that requests an SSM snapshot tier. Any of these set on a
/// model with no recurrent state is a startup error. (`ATLAS_SSM_SWAP_NS` /
/// `ATLAS_SSM_DECODE_NS` are namespace *overrides*, not tier selectors, and
/// are deliberately absent.)
const SSM_TIER_VARS: [&str; 5] = [
    "ATLAS_SSM_TIER",
    "ATLAS_SSM_RDMA_TIER",
    "ATLAS_SSM_SWAP",
    "ATLAS_SSM_DECODE_TIER",
    "ATLAS_SSM_DECODE_RING_ROLL",
];

/// Fail fast when an SSM tier is requested on a model that cannot populate
/// it. Called from `TransformerModel::new` BEFORE pool construction, weight
/// residency, or any peer connect (a capability error is a config error, not
/// a connectivity error).
pub(crate) fn ensure_ssm_tier_capability(config: &ModelConfig) -> Result<()> {
    let set: Vec<&str> = SSM_TIER_VARS
        .iter()
        .copied()
        .filter(|v| std::env::var_os(v).is_some())
        .collect();
    ensure_ssm_tier_capability_from(config, &set)
}

/// Env-free core (unit-testable without process-global `set_var` races):
/// `set_vars` is the list of SSM tier env vars found set.
pub(crate) fn ensure_ssm_tier_capability_from(
    config: &ModelConfig,
    set_vars: &[&str],
) -> Result<()> {
    if set_vars.is_empty() || config.has_recurrent_state() {
        return Ok(());
    }
    bail!(
        "model '{}' has no recurrent state (num_ssm_layers=0) — the SSM snapshot \
         tier cannot be populated by this model. Unset {} for this serve (fleet-wide \
         env files now need per-model curation; a tier request on an incapable model \
         was previously a SILENT no-op and is now a startup error).",
        config.model_type,
        set_vars.join(", "),
    );
}

#[cfg(test)]
#[path = "capability_tests.rs"]
mod tests;
