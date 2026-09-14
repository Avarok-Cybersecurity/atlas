// SPDX-License-Identifier: AGPL-3.0-only

//! What a campaign has to run, derived from the gate SSOT.
//!
//! Pure: takes the statuses `gate::check_gates` produced and a lookup for
//! measured durations, returns the units in the order a single box should run
//! them. Nothing here reads the filesystem, so a plan is testable by
//! construction.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use atlas_plugin::gate::{self, GateStatus};
use atlas_plugin::hardware::policy::Sensitivity;
use atlas_plugin::registry;

/// Where a unit's duration estimate came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estimate {
    /// The descriptor's `expected_secs`.
    Declared(u64),
    /// The newest completed run in `~/.atlas/runs`.
    Measured { secs: u64, recorded_at: u64 },
}

/// One benchmark the campaign runs: a plain gate, or one shard of a group.
#[derive(Clone, Debug)]
pub struct Unit {
    pub id: &'static str,
    /// The group this unit is a shard of, if any. The verdict belongs to the
    /// group; the unit only produces a record.
    pub group: Option<&'static str>,
    pub class: Sensitivity,
    pub estimate: Estimate,
    pub needs_confirmation: bool,
}

impl Unit {
    /// Seconds the scheduler plans with.
    pub fn secs(&self) -> u64 {
        match self.estimate {
            Estimate::Declared(s) | Estimate::Measured { secs: s, .. } => s,
        }
    }
}

/// The required gates that are not `Pass` at this commit, in `REQUIRED_GATES`
/// order — or the subset the caller named, each of which must be open.
pub fn remaining(
    statuses: &BTreeMap<String, GateStatus>,
    only: &[String],
) -> Result<Vec<&'static str>> {
    let open: Vec<&'static str> = gate::REQUIRED_GATES
        .iter()
        .copied()
        .filter(|id| !matches!(statuses.get(*id), Some(GateStatus::Pass)))
        .collect();
    if only.is_empty() {
        return Ok(open);
    }
    let mut chosen = Vec::new();
    for want in only {
        match open.iter().find(|id| **id == want.as_str()) {
            Some(id) => chosen.push(*id),
            None => bail!(
                "{want} already passes at this commit; pass no --gates to run everything \
                 that is still open ({})",
                if open.is_empty() {
                    "nothing".to_string()
                } else {
                    open.join(", ")
                }
            ),
        }
    }
    Ok(chosen)
}

/// The benchmarks that produce a gate's evidence: its shards for a group, the
/// gate itself otherwise.
pub fn expand(gate_id: &'static str) -> Vec<&'static str> {
    match gate::group::find(gate_id) {
        Some(group) => group.members.to_vec(),
        None => vec![gate_id],
    }
}

/// Units for these gates. `measured(id)` returns `(secs, recorded_at)` of the
/// newest completed run of `id`, when there is one.
pub fn units(
    gates: &[&'static str],
    measured: &dyn Fn(&str) -> Option<(u64, u64)>,
) -> Result<Vec<Unit>> {
    let mut out = Vec::new();
    for gate_id in gates {
        for id in expand(gate_id) {
            let Some(d) = registry::find(id) else {
                bail!("{id} is a required gate but not a registered benchmark");
            };
            let estimate = match measured(id) {
                Some((secs, recorded_at)) if secs > 0 => Estimate::Measured { secs, recorded_at },
                _ => Estimate::Declared(d.expected_secs),
            };
            out.push(Unit {
                id,
                group: gate::group::member_of(id).map(|g| g.id),
                class: d.sensitivity,
                estimate,
                needs_confirmation: d.needs_confirmation,
            });
        }
    }
    Ok(out)
}

/// The order one box runs its units in: the long correctness legs first (a
/// failure there is the one that must not wait five hours to be seen), then
/// the Speed class shortest-first, then everything else shortest-first.
///
/// Stable, so two runs of the same plan print the same list.
pub fn order_local(mut units: Vec<Unit>) -> Vec<Unit> {
    fn rank(u: &Unit) -> (u8, u64, &'static str) {
        let tier = match (u.group, u.class) {
            (Some(_), _) => 0,
            (None, Sensitivity::Speed) => 1,
            (None, Sensitivity::Correctness) => 2,
        };
        // Groups longest-first (so the slowest shard set starts first), the
        // rest shortest-first.
        let secs = if tier == 0 {
            u64::MAX - u.secs()
        } else {
            u.secs()
        };
        (tier, secs, u.id)
    }
    units.sort_by_key(rank);
    units
}

/// The wall-clock estimate of running `units` back to back.
pub fn serial_estimate_secs(units: &[Unit]) -> u64 {
    units.iter().map(Unit::secs).sum()
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod plan_tests;
