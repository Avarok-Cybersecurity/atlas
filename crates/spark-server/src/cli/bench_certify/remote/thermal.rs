// SPDX-License-Identifier: AGPL-3.0-only

//! Cool-down: a box that warms up mid-campaign is parked until it is back
//! near where it started, and the rest of the fleet keeps working.
//!
//! Before a worker takes another unit it reads the node's hottest chassis
//! zone and the driver's thermal-throttle flag. At the class's
//! `chassis_park_c` or above, or with the throttle asserted, the node is
//! PARKED: it takes nothing more and re-reads every [`RECHECK`] until the
//! zone is at or below `chassis_resume_c` and the throttle is clear
//! (hysteresis, so a box does not flap on the threshold). The two lines are
//! the target's, from `kernels/<hw>/HARDWARE.toml`
//! `[benchmarks.limits.thermal]` (`hardware::thermal`), never a constant
//! named after one card. Nothing else waits for it — the scheduler is work-conserving,
//! so pending units go to whichever node is free — and a parked box that
//! hosts the bundled Speed class simply delays that class.
//!
//! ★ Absolute, not relative to the box's rest temperature. The first cut
//! parked at +20 °C over the plan-time baseline, and the first campaign under
//! it parked both remote boxes after their first unit: a GB10 rises 26-33 °C
//! over rest under any gate (43 → 76, 39 → 65 on 2026-09-15) and needs a
//! long idle to come back within 5 — "close to baseline" is where a box is
//! only when it is not working. What is abnormal is a box in throttle
//! territory: the 0.66 tok/s incident was a chassis at 89 °C with the driver
//! reporting a thermal slowdown; every healthy loaded box today read 55-76.
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
//!
//! `--dangerous-ignore-thermals` turns every park into a WARNING and lets
//! the box keep taking units: the operator has decided the hardware is
//! theirs to risk. The equivalence policy still judges the records at the
//! end — the flag ignores the security action, never the evidence.

use std::time::Duration;

use super::node::Node;
use avarok_plugin::hardware::limits::ThermalEnvelope;

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
    /// Wait: the reading, and whether the driver reports a thermal throttle.
    Park { now_c: f64, throttled: bool },
    /// No temperature reading; take one, and say so.
    Blind,
}

/// One live reading of a node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reading {
    /// The hottest chassis zone, °C.
    pub chassis_c: Option<f64>,
    /// The driver's thermal-slowdown flag, when it could be read.
    pub throttled: Option<bool>,
}

/// Pure: the hysteresis rule against the class's envelope. A throttle flag
/// parks regardless of the temperature; a missing temperature parks nothing
/// unless the flag is set.
#[must_use]
pub fn judge(r: Reading, parked: bool, env: &ThermalEnvelope) -> Verdict {
    let throttled = r.throttled == Some(true);
    let Some(now_c) = r.chassis_c else {
        return if throttled {
            Verdict::Park {
                now_c: f64::NAN,
                throttled,
            }
        } else {
            Verdict::Blind
        };
    };
    let hold = throttled
        || if parked {
            now_c > env.chassis_resume_c
        } else {
            now_c >= env.chassis_park_c
        };
    if hold {
        Verdict::Park { now_c, throttled }
    } else {
        Verdict::Ready
    }
}

/// Where a node's live chassis reading comes from.
pub trait Probe: Send + Sync {
    /// The node's chassis temperature and throttle flag now.
    fn read(&self, node: &Node) -> Reading;
}

/// The real probe: this box through `HardwareState`, a remote node through
/// `atlasctl bench nodes`.
pub struct FleetProbe {
    pub atlasctl: std::sync::Arc<dyn super::atlasctl::Avarokctl>,
}

impl Probe for FleetProbe {
    fn read(&self, node: &Node) -> Reading {
        if node.local {
            let s = avarok_plugin::hardware::HardwareState::collect();
            return Reading {
                chassis_c: s.hottest_chassis_c(),
                throttled: s.throttle_active.thermal(),
            };
        }
        let blind = Reading {
            chassis_c: None,
            throttled: None,
        };
        let Ok(rows) = self.atlasctl.nodes(std::slice::from_ref(&node.addr)) else {
            return blind;
        };
        let Some(info) = rows
            .iter()
            .find(|r| r.node == node.addr)
            .and_then(|r| r.info.as_ref())
        else {
            return blind;
        };
        let fp = super::node::fingerprint_of(info);
        Reading {
            chassis_c: fp.hottest_chassis_c,
            throttled: fp.thermal_alert,
        }
    }
}

/// One node's cool-down state across a worker's loop.
#[derive(Debug, Default)]
pub struct Gate {
    parked_since: Option<std::time::Instant>,
    said_blind: bool,
    /// Under `ignore`: whether the last reading would have parked, so the
    /// warning is said on the way in and the all-clear on the way out, not
    /// on every unit.
    warned_hot: bool,
}

impl Gate {
    /// Whether the node may take a unit now, reading the probe. `false`
    /// means the caller should sleep [`RECHECK`] and ask again. Every
    /// transition is reported through `say`. With `ignore`
    /// (`--dangerous-ignore-thermals`) the answer is always `true` and a
    /// park becomes a warning.
    pub fn may_take(
        &mut self,
        node: &Node,
        probe: &dyn Probe,
        envelope: Option<ThermalEnvelope>,
        ignore: bool,
        say: &dyn Fn(&str),
    ) -> bool {
        // No envelope is reachable only under `--dangerous-ignore-thermals`
        // (`certify_cmd` refuses otherwise): nothing to judge by, so nothing
        // is parked, and that was said at plan time.
        let Some(envelope) = envelope else {
            return true;
        };
        let r = probe.read(node);
        let park_c = envelope.chassis_park_c;
        let resume_c = envelope.chassis_resume_c;
        let why = |now_c: f64, throttled: bool| {
            if throttled {
                format!("the driver reports a thermal slowdown (chassis {now_c:.0} °C)")
            } else {
                format!("chassis {now_c:.0} °C (park at {park_c:.0}, resume at {resume_c:.0})")
            }
        };
        if ignore {
            match judge(r, self.warned_hot, &envelope) {
                Verdict::Park { now_c, throttled } => {
                    if !self.warned_hot {
                        self.warned_hot = true;
                        say(&format!(
                            "WARNING --dangerous-ignore-thermals: {} would be parked — {}; \
                             continuing on the operator's say-so — its records are still judged \
                             by the equivalence policy",
                            node.addr,
                            why(now_c, throttled)
                        ));
                    }
                }
                Verdict::Ready => {
                    if self.warned_hot {
                        self.warned_hot = false;
                        say(&format!(
                            "{} is back at or below {resume_c:.0} °C",
                            node.addr
                        ));
                    }
                }
                Verdict::Blind => {}
            }
            return true;
        }
        match judge(r, self.parked_since.is_some(), &envelope) {
            Verdict::Ready => {
                if let Some(since) = self.parked_since.take() {
                    say(&format!(
                        "cool-down: {} is back at or below {resume_c:.0} °C after {} s; resuming",
                        node.addr,
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
            Verdict::Park { now_c, throttled } => {
                let since = *self.parked_since.get_or_insert_with(|| {
                    say(&format!(
                        "cool-down: {} parked — {}; nothing more until it is at or below \
                         {resume_c:.0} °C with no throttle — the other boxes keep working",
                        node.addr,
                        why(now_c, throttled)
                    ));
                    std::time::Instant::now()
                });
                if since.elapsed() >= MAX_PARK {
                    say(&format!(
                        "cool-down: {} still {} after {} s parked; resuming anyway — its records \
                         are judged by the equivalence policy like any other",
                        node.addr,
                        why(now_c, throttled),
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
