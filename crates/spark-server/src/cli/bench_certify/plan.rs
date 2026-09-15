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
    /// The benchmark id the child runs — for a shard, the GROUP's id.
    pub id: &'static str,
    /// The group this unit is a shard of, if any. The verdict belongs to the
    /// group; the unit only produces a record.
    pub group: Option<&'static str>,
    /// `(index, count)` of the slice this unit measures; `None` for a plain
    /// gate. Passed to the child as `--param shard=index/count`.
    pub shard: Option<(usize, usize)>,
    pub class: Sensitivity,
    pub estimate: Estimate,
    pub needs_confirmation: bool,
}

impl Unit {
    /// What the operator sees: `bfcl-subset[3/8]` for a shard, the id alone
    /// otherwise. Also the log file's stem.
    pub fn label(&self) -> String {
        match self.shard {
            Some((i, n)) => format!("{}[{i}/{n}]", self.id),
            None => self.id.to_string(),
        }
    }

    /// The filename-safe spelling: `bfcl-subset-s3of8`, the same tail the
    /// record itself carries, so a unit's log sits beside its record by name.
    pub fn file_stem(&self) -> String {
        format!("{}{}", self.id, gate::shard_suffix(self.shard))
    }

    /// The `--param shard=i/n` the child needs, if this unit is a shard.
    pub fn shard_param(&self) -> Option<String> {
        self.shard.map(|(i, n)| format!("shard={i}/{n}"))
    }
}

/// How many shards a group's draw is cut into when nothing at this commit
/// says otherwise: two per box that will run, so the list scheduler has
/// slices to balance around the long gates (kat-equality is ~65 min; a
/// quarter of the echolp draw is ~45), and one box alone runs the whole draw
/// — a shard costs a server start and a warm-up, and there is nothing to
/// balance against. `--shards N` overrides; a partition already begun at
/// this commit is finished at its own count regardless
/// (`gate::shards_owed`).
pub fn shard_count(boxes: usize) -> usize {
    if boxes <= 1 { 1 } else { 2 * boxes }
}

/// Wall time a self-served gate spends BEFORE its first sample: the child
/// starts a server and loads a checkpoint, and neither is in the measured
/// `frame.elapsed` a unit's estimate comes from. Measured 2026-09-13 on a
/// GB10: ~40 s for a 27B NVFP4 checkpoint already in the page cache, and
/// stack #1073's third campaign killed `video-fidelity` at 52 s — a 17 s bench
/// under a `17 × 3` deadline — while its server was still loading. A cold
/// 35B FP8 load from disk runs to several minutes, hence ten.
pub const SERVE_ALLOWANCE: std::time::Duration = std::time::Duration::from_secs(600);

impl Unit {
    /// When to give up on this unit: the serve allowance, then the estimate
    /// scaled by `timeout_factor` (≥ 1, `CertifyArgs::validate`). The ONE
    /// spelling of the rule — the local loop and every fleet worker read it,
    /// so a unit that fits on one box fits on every equivalent one.
    pub fn deadline(&self, timeout_factor: f64) -> std::time::Duration {
        SERVE_ALLOWANCE
            + std::time::Duration::from_secs((self.secs() as f64 * timeout_factor) as u64)
    }

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

/// Which shards of a group a certification still owes, as `(index, count)`
/// — injected so the plan stays pure; the real one is [`gate::shards_owed`],
/// asked for the count the campaign wants.
pub type Owed<'a> = &'a dyn Fn(&'static gate::group::BenchmarkGroup) -> Vec<(usize, usize)>;

/// A fresh partition of `n` shards: the answer when nothing is banked yet.
pub fn fresh_partition(
    n: usize,
) -> impl Fn(&'static gate::group::BenchmarkGroup) -> Vec<(usize, usize)> {
    move |_| (0..n).map(|i| (i, n)).collect()
}

/// The runs that produce a gate's evidence: the shards a group still owes,
/// the gate itself otherwise.
///
/// A shard the gate would already accept at this commit is not re-measured:
/// stack #1073's third campaign spent 90 node-minutes re-running three shards
/// whose records were already banked, because the plan expanded a group to
/// every member regardless. `owed` is the gate's own answer to "which
/// shards count", so what is skipped here is exactly what the verdict will
/// take — and it also decides the COUNT: a partition started at this commit
/// is finished at its own count, whatever the campaign would pick fresh.
pub fn expand(
    gate_id: &'static str,
    owed: Owed<'_>,
) -> Vec<(&'static str, Option<(usize, usize)>)> {
    match gate::group::find(gate_id) {
        Some(group) => owed(group)
            .into_iter()
            .map(|s| (gate_id, Some(s)))
            .collect(),
        None => vec![(gate_id, None)],
    }
}

/// Units for these gates. `measured(id)` returns `(secs, recorded_at)` of the
/// newest completed run of `id`, when there is one; `owed` says which shards
/// of a group still need a record. A shard's estimate is the group's
/// (declared or measured for the whole draw) divided by its count, floored
/// at [`SHARD_FLOOR_SECS`]: a server start and a warm-up do not shrink with
/// the slice.
pub fn units(
    gates: &[&'static str],
    measured: &dyn Fn(&str) -> Option<(u64, u64)>,
    owed: Owed<'_>,
) -> Result<Vec<Unit>> {
    let mut out = Vec::new();
    for gate_id in gates {
        for (id, shard) in expand(gate_id, owed) {
            let Some(d) = registry::find(id) else {
                bail!("{id} is a required gate but not a registered benchmark");
            };
            let mut estimate = match measured(id) {
                Some((secs, recorded_at)) if secs > 0 => Estimate::Measured { secs, recorded_at },
                _ => Estimate::Declared(d.expected_secs),
            };
            if let Some((_, n)) = shard {
                estimate = match estimate {
                    Estimate::Declared(s) => {
                        Estimate::Declared((s / n as u64).max(SHARD_FLOOR_SECS))
                    }
                    Estimate::Measured { secs, recorded_at } => Estimate::Measured {
                        secs: (secs / n as u64).max(SHARD_FLOOR_SECS),
                        recorded_at,
                    },
                };
            }
            out.push(Unit {
                id,
                group: gate::group::find(id).map(|g| g.id),
                shard,
                class: d.sensitivity,
                estimate,
                needs_confirmation: d.needs_confirmation,
            });
        }
    }
    Ok(out)
}

/// The least a shard is planned at, however thin the slice: a server start,
/// a warm-up and the fixed per-run overhead. Measured 2026-09-14: a quarter
/// of the 27B draw ran 1181-1534 s, an echolp quarter 2516-2698 s.
pub const SHARD_FLOOR_SECS: u64 = 300;

/// The order one box runs its units in: the long correctness legs first (a
/// failure there is the one that must not wait five hours to be seen), then
/// the Speed class shortest-first, then everything else shortest-first.
///
/// Stable, so two runs of the same plan print the same list.
pub fn order_local(mut units: Vec<Unit>) -> Vec<Unit> {
    fn rank(u: &Unit) -> (u8, u64, &'static str, Option<(usize, usize)>) {
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
        (tier, secs, u.id, u.shard)
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
