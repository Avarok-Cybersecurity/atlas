// SPDX-License-Identifier: AGPL-3.0-only

//! Cool-down: a box that warms up mid-campaign is parked until it is back
//! near where it started, and the rest of the fleet keeps working.
//!
//! Every node is admitted with a baseline — the hottest chassis zone at rest,
//! at plan time. Before a worker takes another unit it reads the zone again.
//! Past [`PARK_ABOVE_BASELINE_C`] over its own baseline the node is PARKED:
//! it takes nothing more and re-reads every [`RECHECK`] until the zone is
//! back within [`RESUME_WITHIN_C`] of baseline (hysteresis, so a box does not
//! flap on the threshold). Nothing else waits for it — the scheduler is
//! work-conserving, so pending units go to whichever node is free — and a
//! parked box that hosts the bundled Speed class simply delays that class.
//!
//! Why: the 2026-09-15 campaign spread its Speed class over two boxes that
//! were 43/40 °C at plan time and 55/68 °C in the records — one had run
//! five gates back to back — and `gate::agreement` refused the pair. The
//! equivalence policy judges the RECORDS; this keeps the boxes in the state
//! the policy assumed, and keeps a hot box from being driven hotter. The
//! thresholds are the operator's cut, not a measurement: a GB10 warms 12-28
//! °C over rest under a campaign, and the 0.66 tok/s incident was a box at
//! 89 °C (24 °C over its peer).
//!
//! A reading that cannot be taken parks nothing: the safety net for a wrong
//! reading is the record-level check, and a blind probe must not stop a
//! campaign. It is said, once.

use std::time::Duration;

use super::node::Node;

/// Degrees over its own plan-time baseline at which a node is parked.
pub const PARK_ABOVE_BASELINE_C: f64 = 20.0;
/// Degrees over baseline a parked node must fall back to before it resumes.
pub const RESUME_WITHIN_C: f64 = 5.0;
/// How often a parked node is re-read.
pub const RECHECK: Duration = Duration::from_secs(60);
/// The longest a node stays parked. A box that will not cool (ambient rose,
/// a fan failed) resumes with a warning rather than holding its units
/// forever; the records it then writes are still judged by the equivalence
/// policy, so a bad capture is refused there, never hidden here.
pub const MAX_PARK: Duration = Duration::from_secs(30 * 60);

/// What the zone says about taking another unit.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Take one.
    Ready,
    /// Wait: `now_c` against `baseline_c`, and how far it has to fall.
    Park { now_c: f64, baseline_c: f64 },
    /// No reading (either side); take one, and say so.
    Blind,
}

/// Pure: the hysteresis rule.
#[must_use]
pub fn judge(baseline_c: Option<f64>, now_c: Option<f64>, parked: bool) -> Verdict {
    let (Some(baseline_c), Some(now_c)) = (baseline_c, now_c) else {
        return Verdict::Blind;
    };
    let over = now_c - baseline_c;
    let hold = if parked {
        over > RESUME_WITHIN_C
    } else {
        over > PARK_ABOVE_BASELINE_C
    };
    if hold {
        Verdict::Park { now_c, baseline_c }
    } else {
        Verdict::Ready
    }
}

/// Where a node's live chassis reading comes from.
pub trait Probe: Send + Sync {
    /// The hottest chassis zone now, °C, or `None` when it cannot be read.
    fn hottest_chassis_c(&self, node: &Node) -> Option<f64>;
}

/// The real probe: this box through `HardwareState`, a remote node through
/// `atlasctl bench nodes`.
pub struct FleetProbe {
    pub atlasctl: std::sync::Arc<dyn super::atlasctl::Atlasctl>,
}

impl Probe for FleetProbe {
    fn hottest_chassis_c(&self, node: &Node) -> Option<f64> {
        if node.local {
            return atlas_plugin::hardware::HardwareState::collect().hottest_chassis_c();
        }
        let rows = self.atlasctl.nodes(std::slice::from_ref(&node.addr)).ok()?;
        let row = rows.iter().find(|r| r.node == node.addr)?;
        super::node::fingerprint_of(row.info.as_ref()?).hottest_chassis_c
    }
}

/// One node's cool-down state across a worker's loop.
#[derive(Debug, Default)]
pub struct Gate {
    parked_since: Option<std::time::Instant>,
    said_blind: bool,
}

impl Gate {
    /// Whether the node may take a unit now, reading the probe. `false`
    /// means the caller should sleep [`RECHECK`] and ask again. Every
    /// transition is reported through `say`.
    pub fn may_take(&mut self, node: &Node, probe: &dyn Probe, say: &dyn Fn(&str)) -> bool {
        let now = probe.hottest_chassis_c(node);
        match judge(
            node.hardware.hottest_chassis_c,
            now,
            self.parked_since.is_some(),
        ) {
            Verdict::Ready => {
                if let Some(since) = self.parked_since.take() {
                    say(&format!(
                        "cool-down: {} is back within {RESUME_WITHIN_C:.0} °C of its baseline \
                         ({:.0} °C) after {} s; resuming",
                        node.addr,
                        node.hardware.hottest_chassis_c.unwrap_or(f64::NAN),
                        since.elapsed().as_secs()
                    ));
                }
                true
            }
            Verdict::Blind => {
                if !self.said_blind {
                    self.said_blind = true;
                    say(&format!(
                        "cool-down: {} reports no chassis temperature; it is never parked \
                         (the records' own captures still decide equivalence)",
                        node.addr
                    ));
                }
                self.parked_since = None;
                true
            }
            Verdict::Park { now_c, baseline_c } => {
                let since = *self.parked_since.get_or_insert_with(|| {
                    say(&format!(
                        "cool-down: {} reads {now_c:.0} °C against a baseline of {baseline_c:.0} °C \
                         (> {PARK_ABOVE_BASELINE_C:.0} °C over); parked until it is within \
                         {RESUME_WITHIN_C:.0} °C — the other boxes keep working",
                        node.addr
                    ));
                    std::time::Instant::now()
                });
                if since.elapsed() >= MAX_PARK {
                    say(&format!(
                        "cool-down: {} still reads {now_c:.0} °C after {} s parked; resuming \
                         anyway — its records are judged by the equivalence policy like any other",
                        node.addr,
                        since.elapsed().as_secs()
                    ));
                    self.parked_since = None;
                    return true;
                }
                false
            }
        }
    }
}

#[cfg(test)]
#[path = "thermal_tests.rs"]
mod thermal_tests;
