// SPDX-License-Identifier: AGPL-3.0-only
//! When two boxes may share one Speed-class campaign.
//!
//! The one-signer rule for `Sensitivity::Speed` (see `gate::agreement`) came
//! from two GB10s that read 0.66 tok/s apart on one gate — ten times either
//! box's own sigma — with every static field identical. What separated them
//! was live state: 65 °C against 89 °C in the chassis, and a thermal throttle
//! reason asserted on the hot one. So "the same hardware" is not a model
//! name; it is a model name, a driver line, a clock ceiling, a memory size,
//! AND a thermal state close enough that the number would not move.
//!
//! This module is the ONLY place that decision is made. `spark bench certify`
//! asks it before spreading Speed units across nodes; `gate::agreement` asks
//! it again over the records that came back, so CI decides from what was
//! measured, not from what a scheduler believed. atlasctl reports the facts
//! and decides nothing.
//!
//! A field one side cannot report makes the pair [`Mismatch::Undecidable`],
//! which is not equivalent: the safe answer to "are these the same box?" is
//! never "probably".

use super::state::HardwareState;
use super::{Hardware, HardwareStateReport};
use crate::gate::GateRecord;
use serde::{Deserialize, Serialize};

/// What decides equivalence, read off a record or a live box.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwareFingerprint {
    /// The accelerator name as the driver reports it (`NVIDIA GB10`).
    pub gpu: String,
    /// The driver's major version (`580` of `580.95.05`).
    pub driver_major: Option<u32>,
    /// The box's own SM clock ceiling, MHz.
    pub sm_clock_max_mhz: Option<f64>,
    /// Host memory, kB (unified on GB10, so this is the GPU's too).
    pub mem_total_kb: Option<u64>,
    /// Any thermal throttle reason asserted at capture.
    pub thermal_alert: Option<bool>,
    /// The hottest chassis zone at capture, °C.
    pub hottest_chassis_c: Option<f64>,
    /// For a record: whether its post-run hardware check was valid. `None`
    /// for a live box (nothing has run yet) or a record without a report.
    pub postcheck_valid: Option<bool>,
}

impl HardwareFingerprint {
    /// From a record: static fields off `hardware`, live fields off the
    /// `before` capture, validity off the postcheck.
    pub fn from_record(record: &GateRecord) -> Self {
        let report: Option<&HardwareStateReport> = record.hardware_state.as_ref();
        let before = report.map(|r| &r.before);
        let mut fp = Self::from_parts(&record.hardware, before);
        fp.postcheck_valid = report.and_then(|r| {
            r.postcheck
                .as_ref()
                .map(|p| p.validity == super::policy::Validity::Valid)
        });
        fp
    }

    /// From a live box, before anything runs.
    pub fn from_live(hardware: &Hardware, state: &HardwareState) -> Self {
        Self::from_parts(hardware, Some(state))
    }

    fn from_parts(hardware: &Hardware, state: Option<&HardwareState>) -> Self {
        Self {
            gpu: hardware.gpu.clone(),
            driver_major: driver_major(&hardware.driver),
            sm_clock_max_mhz: state.and_then(|s| s.sm_clock_max_mhz),
            mem_total_kb: state.and_then(|s| s.mem_total_kb),
            thermal_alert: state.and_then(|s| s.throttle_active.thermal()),
            hottest_chassis_c: state.and_then(HardwareState::hottest_chassis_c),
            postcheck_valid: None,
        }
    }
}

/// `580.95.05` → `580`. Empty or non-numeric → `None`.
pub fn driver_major(driver: &str) -> Option<u32> {
    driver.split('.').next()?.trim().parse().ok()
}

/// The tolerances. Named so a record of why they are what they are can sit
/// beside them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquivalencePolicy {
    /// `|a - b| / max(a, b)` on the clock ceiling.
    pub clock_spread: f64,
    /// `|a - b| / max(a, b)` on memory.
    pub mem_spread: f64,
    /// `|a - b|` on the hottest chassis zone, °C.
    pub chassis_delta_c: f64,
}

/// The policy for spreading Speed-class gates.
///
/// * clock ceiling within 1 %: two GB10s report 3003 MHz exactly; anything
///   else is a different part or a different firmware.
/// * memory within 5 %: 128 GB parts read a few hundred MB apart across
///   kernels; a different SKU is 30 % away.
/// * chassis within 15 °C: the 0.66 tok/s incident was a 24 °C gap. The
///   limit was 10 °C until 2026-09-15 — a number with no measurement behind
///   it (owner decision: raise it unless data says otherwise, and none
///   does). What IS measured: one box swings ~8 °C between morning and
///   afternoon, and two equivalent GB10s running one campaign read 55 vs
///   66-68 °C in their records (dgx2 idle-ish, dgx3 after five gates
///   back-to-back) with no number moving — that 11-13 °C pair was refused at
///   10 and is the normal spread of a fleet under load. 15 accepts it and
///   still refuses the incident by a margin; a box that warms past that
///   mid-campaign is parked by the orchestrator's cool-down instead
///   (`bench_certify::remote::thermal`).
pub const SPEED_SPREAD: EquivalencePolicy = EquivalencePolicy {
    clock_spread: 0.01,
    mem_spread: 0.05,
    chassis_delta_c: 15.0,
};

/// Why two fingerprints are not one box.
#[derive(Clone, Debug, PartialEq)]
pub enum Mismatch {
    Gpu(String, String),
    DriverMajor(u32, u32),
    ClockSpread {
        a: f64,
        b: f64,
        limit: f64,
    },
    MemSpread {
        a: u64,
        b: u64,
        limit: f64,
    },
    ChassisDelta {
        a: f64,
        b: f64,
        limit: f64,
    },
    /// One side (or both) had a thermal reason asserted.
    ThermalAlert {
        a: bool,
        b: bool,
    },
    /// A record's post-run check was not valid.
    PostcheckInvalid,
    /// A field one side did not report; named so the operator can fix the
    /// capture rather than guess.
    Undecidable(&'static str),
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu(a, b) => write!(f, "gpu {a:?} vs {b:?}"),
            Self::DriverMajor(a, b) => write!(f, "driver major {a} vs {b}"),
            Self::ClockSpread { a, b, limit } => write!(
                f,
                "clock ceiling {a:.0} vs {b:.0} MHz (limit {:.1} %)",
                limit * 100.0
            ),
            Self::MemSpread { a, b, limit } => write!(
                f,
                "memory {} vs {} MiB (limit {:.0} %)",
                a >> 10,
                b >> 10,
                limit * 100.0
            ),
            Self::ChassisDelta { a, b, limit } => {
                write!(f, "chassis {a:.0} vs {b:.0} °C (limit {limit:.0} °C)")
            }
            Self::ThermalAlert { a, b } => write!(
                f,
                "thermal throttle asserted ({})",
                match (a, b) {
                    (true, true) => "both",
                    (true, false) => "first",
                    _ => "second",
                }
            ),
            Self::PostcheckInvalid => write!(f, "a post-run hardware check was not valid"),
            Self::Undecidable(field) => write!(f, "{field} not reported on one side"),
        }
    }
}

fn spread(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m <= 0.0 { 0.0 } else { (a - b).abs() / m }
}

/// Are `a` and `b` one box, for the purposes of a Speed-class campaign?
///
/// Every mismatch is returned, not the first, so the operator sees the whole
/// reason. A postcheck that is present and invalid on either side is a
/// mismatch; a postcheck that is absent (live box) is not — validity is
/// judged on records, where it exists.
pub fn equivalent(
    a: &HardwareFingerprint,
    b: &HardwareFingerprint,
    p: &EquivalencePolicy,
) -> Result<(), Vec<Mismatch>> {
    let mut out = Vec::new();
    if a.gpu.is_empty() || b.gpu.is_empty() {
        out.push(Mismatch::Undecidable("gpu"));
    } else if a.gpu != b.gpu {
        out.push(Mismatch::Gpu(a.gpu.clone(), b.gpu.clone()));
    }
    match (a.driver_major, b.driver_major) {
        (Some(x), Some(y)) if x != y => out.push(Mismatch::DriverMajor(x, y)),
        (Some(_), Some(_)) => {}
        _ => out.push(Mismatch::Undecidable("driver")),
    }
    match (a.sm_clock_max_mhz, b.sm_clock_max_mhz) {
        (Some(x), Some(y)) => {
            if spread(x, y) > p.clock_spread {
                out.push(Mismatch::ClockSpread {
                    a: x,
                    b: y,
                    limit: p.clock_spread,
                });
            }
        }
        _ => out.push(Mismatch::Undecidable("sm_clock_max_mhz")),
    }
    match (a.mem_total_kb, b.mem_total_kb) {
        (Some(x), Some(y)) => {
            if spread(x as f64, y as f64) > p.mem_spread {
                out.push(Mismatch::MemSpread {
                    a: x,
                    b: y,
                    limit: p.mem_spread,
                });
            }
        }
        _ => out.push(Mismatch::Undecidable("mem_total_kb")),
    }
    match (a.thermal_alert, b.thermal_alert) {
        (Some(x), Some(y)) => {
            if x || y {
                out.push(Mismatch::ThermalAlert { a: x, b: y });
            }
        }
        _ => out.push(Mismatch::Undecidable("throttle reasons")),
    }
    match (a.hottest_chassis_c, b.hottest_chassis_c) {
        (Some(x), Some(y)) => {
            if (x - y).abs() > p.chassis_delta_c {
                out.push(Mismatch::ChassisDelta {
                    a: x,
                    b: y,
                    limit: p.chassis_delta_c,
                });
            }
        }
        _ => out.push(Mismatch::Undecidable("chassis temperature")),
    }
    if a.postcheck_valid == Some(false) || b.postcheck_valid == Some(false) {
        out.push(Mismatch::PostcheckInvalid);
    }
    if out.is_empty() { Ok(()) } else { Err(out) }
}

#[cfg(test)]
#[path = "equivalence_tests.rs"]
mod equivalence_tests;
