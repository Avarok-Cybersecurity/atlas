// SPDX-License-Identifier: AGPL-3.0-only
//! Who runs what, and when: a pure scheduler over units and admitted nodes.
//!
//! Two decisions live here and nowhere else:
//!
//! * **Speed mode.** Speed-class units may spread across nodes only when
//!   every pair of nodes is one box by [`equivalent`]; otherwise they are
//!   bundled onto the single node with the most headroom and the campaign
//!   says why. Correctness-class units spread regardless.
//! * **Work-conserving list scheduling, longest first.** Whenever a node is
//!   free it takes the longest eligible unit (LPT — within 4/3 of optimal
//!   makespan, and the same rule at planning time and at run time so the
//!   dry-run estimate is what actually happens). Among equals, a shard whose
//!   group already has a shard on this node yields to one that does not, so
//!   losing a node costs one quarter of a group, not the whole of it.
//!
//! Nothing here starts anything; the driver asks [`next_for`] and runs it.

use atlas_plugin::hardware::equivalence::{SPEED_SPREAD, equivalent};
use atlas_plugin::hardware::policy::Sensitivity;

use super::super::plan::Unit;
use super::node::Node;

/// Where Speed-class units may go.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpeedMode {
    /// Every node is one box: Speed units go anywhere.
    Spread,
    /// Not every pair agrees: Speed units go to `node` only.
    Bundle { node: usize, why: Vec<String> },
}

impl SpeedMode {
    pub fn allows(&self, node: usize) -> bool {
        match self {
            Self::Spread => true,
            Self::Bundle { node: home, .. } => *home == node,
        }
    }
}

/// Decide the Speed mode for these nodes.
///
/// One node is trivially Spread. Bundling picks the node with the most free
/// memory, then the coolest chassis — the box most likely to hold its clock
/// for the whole set.
pub fn speed_mode(nodes: &[Node]) -> SpeedMode {
    let mut why = Vec::new();
    for (i, a) in nodes.iter().enumerate() {
        for b in &nodes[i + 1..] {
            if let Err(m) = equivalent(&a.hardware, &b.hardware, &SPEED_SPREAD) {
                why.push(format!(
                    "{} vs {}: {}",
                    a.addr,
                    b.addr,
                    m.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
    }
    if why.is_empty() {
        return SpeedMode::Spread;
    }
    let node = (0..nodes.len())
        .max_by(|&x, &y| {
            let fx = nodes[x].free_fraction.unwrap_or(0.0);
            let fy = nodes[y].free_fraction.unwrap_or(0.0);
            fx.partial_cmp(&fy)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    // Cooler is better, so compare reversed.
                    let cx = nodes[x].hardware.hottest_chassis_c.unwrap_or(f64::MAX);
                    let cy = nodes[y].hardware.hottest_chassis_c.unwrap_or(f64::MAX);
                    cy.partial_cmp(&cx).unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .unwrap_or(0);
    SpeedMode::Bundle { node, why }
}

/// The unit `node` should take next: the longest pending unit it may run,
/// shards of a group it already hosts yielding to others.
///
/// `pending[i]` says unit `i` is still to run; `placed[i]` is the node a
/// running or finished unit went to (for anti-affinity).
pub fn next_for(
    node: usize,
    units: &[Unit],
    pending: &[bool],
    placed: &[Option<usize>],
    mode: &SpeedMode,
) -> Option<usize> {
    let hosts_shard_of = |group: &str| {
        units
            .iter()
            .enumerate()
            .any(|(j, u)| u.group == Some(group) && placed[j] == Some(node))
    };
    (0..units.len())
        .filter(|&i| pending[i])
        .filter(|&i| units[i].class != Sensitivity::Speed || mode.allows(node))
        .min_by_key(|&i| {
            let crowded = units[i].group.is_some_and(hosts_shard_of);
            // Longest first: negate the duration for min_by_key.
            (crowded, std::cmp::Reverse(units[i].secs()))
        })
}

/// A simulated run of the same rule, for the dry-run estimate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Per node, the units in the order it would take them.
    pub queues: Vec<Vec<usize>>,
    /// Per node, when it goes idle for good, seconds from start.
    pub finish_at: Vec<u64>,
    pub makespan_secs: u64,
}

/// Simulate list scheduling. `build_allowance` is added once to a node that
/// does not have the anchor built.
pub fn simulate(units: &[Unit], nodes: &[Node], mode: &SpeedMode, build_allowance: u64) -> Plan {
    let n = nodes.len();
    let mut pending = vec![true; units.len()];
    let mut placed = vec![None; units.len()];
    let mut queues = vec![Vec::new(); n];
    let mut free_at: Vec<u64> = nodes
        .iter()
        .map(|nd| if nd.built { 0 } else { build_allowance })
        .collect();
    let mut idle = vec![false; n];
    while pending.iter().any(|p| *p) {
        // The earliest-free node that can still take something.
        let Some(node) = (0..n).filter(|&k| !idle[k]).min_by_key(|&k| free_at[k]) else {
            break;
        };
        match next_for(node, units, &pending, &placed, mode) {
            Some(i) => {
                pending[i] = false;
                placed[i] = Some(node);
                queues[node].push(i);
                free_at[node] += units[i].secs();
            }
            None => idle[node] = true,
        }
    }
    let makespan_secs = free_at.iter().copied().max().unwrap_or(0);
    Plan {
        queues,
        finish_at: free_at,
        makespan_secs,
    }
}

#[cfg(test)]
#[path = "schedule_tests.rs"]
mod schedule_tests;
