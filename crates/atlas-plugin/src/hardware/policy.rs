// SPDX-License-Identifier: AGPL-3.0-only

//! The verdict: may this benchmark start, and did the box stay honest while it
//! ran.
//!
//! Pure functions over `(sensitivity, state_before, state_delta, options)`.
//! No I/O, no env reads except in [`PolicyOptions::from_env`], so every rule
//! below is unit-testable on any box.
//!
//! # Record first, gate second
//!
//! We do not know where the knee is. We have ONE healthy box (chassis
//! 52–66 °C, SW-thermal 0.05% of uptime) and ONE degraded box (chassis
//! 78–89 °C, SW-thermal 5.0%, HW-thermal 228 s) and nothing in between. A
//! temperature ceiling picked from two points is a guess, and a guessed gate
//! that refuses a healthy box is worse than no gate: it teaches operators to
//! set the kill switch permanently.
//!
//! So the absolute-temperature ceilings below ship DISABLED. They are present,
//! documented with the measurements that suggested them, and require an
//! explicit `ATLAS_HW_TEMP_GATE=1` to become gating. Everything they see is
//! recorded either way — that is how the third and fourth data points get
//! collected, and how a defensible threshold eventually gets set.
//!
//! Two checks ship ENABLED, because neither needs a threshold at all:
//!
//! * **Throttle delta.** "A thermal throttle counter advanced while this
//!   benchmark was running" is self-calibrating — it is the hardware's own
//!   report that it could not sustain the clock, in the hardware's own units,
//!   with no number for us to pick.
//! * **Foreign GPU processes.** A box running a benchmark hosts exactly one
//!   model. A second GPU-resident process is contention, and the count is a
//!   count, not a threshold. This is the check that would have caught the
//!   killed `spark serve` whose driver process kept 87 GB.
//!
//! # Sensitivity, not benchmark id
//!
//! The policy keys on [`Sensitivity`], which every benchmark declares on its
//! [`crate::BenchmarkDescriptor`]. There is deliberately no list of benchmark
//! ids here: the registry is the SSOT for what benchmarks exist, and a
//! hand-maintained id list in this module would go stale the first time one is
//! added and would silently treat the new one as correctness-only.

use serde::{Deserialize, Serialize};

use super::state::{HardwareState, HardwareStateDelta};

/// Whether this benchmark's number is a SPEED number.
///
/// The split is not about importance, it is about what thermal state can
/// corrupt. A throttled box produces a slower wall time and the identical
/// tool-call. So a hot box must refuse a speed gate and must NOT block a
/// correctness gate — blocking BFCL because the chassis is warm would stop
/// accuracy work for a reason that cannot affect accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Sensitivity {
    /// Wall time, TTFT, TPOT, tok/s, or a Σwall bound — thermally corruptible.
    Speed,
    /// Accuracy, fidelity, state integrity — recorded, never gated on state.
    Correctness,
}

/// The pre-run decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Decision {
    /// Nothing to say. Only reachable when every gating input was READABLE and
    /// healthy — an unreadable input yields [`Decision::Warn`], never this.
    Proceed,
    /// Recorded, not blocking.
    Warn,
    /// The run must not start.
    Refuse,
}

/// Whether the numbers a completed run produced may be believed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Validity {
    /// Speed-sensitive, and no thermal throttle counter advanced.
    Valid,
    /// Speed-sensitive, and the throttle counters could not be read on both
    /// captures. NOT a pass.
    Unknown,
    /// Speed-sensitive, and the box throttled during the run.
    Invalid,
    /// Correctness gate: thermal state does not bear on the number.
    NotApplicable,
}

/// A verdict plus every concern that fed it, gating or not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Precheck {
    pub decision: Decision,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concerns: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Postcheck {
    pub validity: Validity,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concerns: Vec<String>,
}

/// The two operator switches, read from the environment in ONE place so the
/// rules themselves stay pure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PolicyOptions {
    /// `ATLAS_NO_HW_PRECHECK=1` — never REFUSE. Everything is still collected,
    /// still recorded, and still reported; only the blocking is suppressed,
    /// and [`postcheck`] still marks a throttled run INVALID. An operator can
    /// decide to measure on a hot box; nobody gets to decide the record should
    /// say it was cool.
    pub kill_switch: bool,
    /// `ATLAS_HW_TEMP_GATE=1` — promote the absolute-temperature ceilings from
    /// recorded to gating. Off until there is a third data point.
    pub absolute_temp_gate: bool,
}

/// Env var that suppresses refusal. Named here and read nowhere else.
pub const KILL_SWITCH_ENV: &str = "ATLAS_NO_HW_PRECHECK";
/// Env var that opts in to the absolute-temperature ceilings.
pub const TEMP_GATE_ENV: &str = "ATLAS_HW_TEMP_GATE";

/// GPU die temperature above which a speed number is treated as suspect.
///
/// DISABLED by default. The two measured points on 2026-08-15, both GB10,
/// driver 580.126.09, governor `performance`, persistence on, 3003 MHz
/// ceiling: dgx1 idled 52–66 °C across its chassis zones and spent 0.05% of an
/// 11.2-day uptime in SW thermal slowdown; dgx2 sat at 70–89 °C and spent 5.0%
/// of a 16.1-hour uptime there, plus 228 s of HW thermal slowdown against
/// dgx1's 0.7 s. 75 °C is the midpoint of that gap and nothing more — there is
/// no measurement between 66 and 70 °C, so this constant is a hypothesis, not
/// a finding. Promote it once a third box exists; until then the delta check
/// does the gating and this is recorded so the third point gets collected.
pub const GPU_TEMP_CEILING_C: f64 = 75.0;

/// Hottest chassis zone above which a speed number is treated as suspect.
///
/// DISABLED by default, same reasoning. dgx1's zones read 65/65/62/59/59/58 °C
/// and dgx2's 89/88/82/74/71/70 °C on the same benchmark; 80 °C sits inside
/// dgx2's spread and above all of dgx1's. It is the reading that separated the
/// boxes when driver, governor, persistence and clock ceiling were identical,
/// which is why it is recorded on every run even while it gates on none.
pub const CHASSIS_TEMP_CEILING_C: f64 = 80.0;

/// GPU compute processes, other than this one, that a speed-sensitive run
/// tolerates.
///
/// ENABLED — this is a count, not a calibration. One is the model under test:
/// a `--pull-request-gate` run's self-provisioned serve, or the endpoint a
/// `--url` run was pointed at. Two means something else is resident, which is
/// exactly the shape of the 2026-08-15 incident where a killed `spark serve`
/// left its driver process holding 87 GB and the next run starved at preflight
/// with "box is not free enough".
pub const MAX_FOREIGN_COMPUTE_APPS: usize = 1;

impl PolicyOptions {
    pub fn from_env() -> Self {
        Self {
            kill_switch: flag(KILL_SWITCH_ENV),
            absolute_temp_gate: flag(TEMP_GATE_ENV),
        }
    }
}

/// `1`/`true`/`yes`, trimmed and case-insensitive. Anything else — including
/// an empty value — is off, so `FOO=` cannot silently disable a check.
fn flag(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

/// Worst-wins, so a Refuse cannot be argued down by a later Proceed.
fn worst(a: Decision, b: Decision) -> Decision {
    match (a, b) {
        (Decision::Refuse, _) | (_, Decision::Refuse) => Decision::Refuse,
        (Decision::Warn, _) | (_, Decision::Warn) => Decision::Warn,
        _ => Decision::Proceed,
    }
}

/// May this benchmark start?
///
/// A correctness gate always may — it collects the same state and returns the
/// same concerns, at [`Decision::Warn`] at worst.
pub fn precheck(
    sensitivity: Sensitivity,
    before: &HardwareState,
    options: PolicyOptions,
) -> Precheck {
    let mut concerns = Vec::new();
    let mut decision = Decision::Proceed;
    let mut gate = |d: Decision, why: String| {
        concerns.push(why);
        decision = worst(decision, d);
    };

    // Enabled: the hardware says it is throttling right now.
    match before.throttle_active.thermal() {
        Some(true) => gate(
            Decision::Refuse,
            "a thermal throttle reason is ACTIVE before the run started".to_string(),
        ),
        Some(false) => {}
        None => gate(
            Decision::Warn,
            "throttle reasons could not be read — the box is not known to be unthrottled"
                .to_string(),
        ),
    }

    // Enabled: contention.
    match before.foreign_compute_apps() {
        Some(n) if n > MAX_FOREIGN_COMPUTE_APPS => gate(
            Decision::Refuse,
            format!(
                "{n} GPU compute processes other than this one (at most \
                 {MAX_FOREIGN_COMPUTE_APPS} expected — the model under test)"
            ),
        ),
        Some(_) => {}
        None => gate(
            Decision::Warn,
            "GPU compute processes could not be listed — contention is unknown".to_string(),
        ),
    }

    // Recorded by default; gating only under ATLAS_HW_TEMP_GATE=1.
    let temp_level = if options.absolute_temp_gate {
        Decision::Refuse
    } else {
        Decision::Warn
    };
    if let Some(t) = before.gpu_temp_c.filter(|t| *t > GPU_TEMP_CEILING_C) {
        gate(
            temp_level,
            format!("GPU die {t:.0} °C is above the {GPU_TEMP_CEILING_C:.0} °C ceiling"),
        );
    }
    if let Some(t) = before
        .hottest_chassis_c()
        .filter(|t| *t > CHASSIS_TEMP_CEILING_C)
    {
        gate(
            temp_level,
            format!(
                "hottest chassis zone {t:.0} °C is above the {CHASSIS_TEMP_CEILING_C:.0} °C ceiling"
            ),
        );
    }

    // A correctness gate records everything above and blocks on none of it.
    if sensitivity == Sensitivity::Correctness && decision == Decision::Refuse {
        decision = Decision::Warn;
        concerns.push(
            "correctness gate — recorded and proceeding; accuracy is not thermally sensitive"
                .to_string(),
        );
    }
    if options.kill_switch && decision == Decision::Refuse {
        decision = Decision::Warn;
        concerns.push(format!(
            "{KILL_SWITCH_ENV}=1 — REFUSAL SUPPRESSED BY OPERATOR. The concerns above stand and \
             are recorded with the run; the kill switch only lets it start."
        ));
    }
    Precheck { decision, concerns }
}

/// Did the box stay honest? Read the delta, not the absolute state.
pub fn postcheck(
    sensitivity: Sensitivity,
    delta: &HardwareStateDelta,
    _options: PolicyOptions,
) -> Postcheck {
    let mut concerns = Vec::new();
    if let Some(f) = delta.thermal_throttle_fraction().filter(|f| *f > 0.0) {
        // A partial sum is a floor, never a figure. Printing it bare would put
        // an understated percentage into the record, where it is quoted as the
        // reason a speed number was accepted.
        concerns.push(if delta.thermal_counters_complete() {
            format!("thermally throttled for {:.2}% of the run", f * 100.0)
        } else {
            format!(
                "thermally throttled for AT LEAST {:.2}% of the run — one or more throttle \
                 counters were unreadable, so the true figure can only be higher",
                f * 100.0
            )
        });
    }
    if let Some(d) = delta.hottest_chassis_delta_c {
        concerns.push(format!("hottest chassis zone moved {d:+.0} °C"));
    }

    if sensitivity == Sensitivity::Correctness {
        return Postcheck {
            validity: Validity::NotApplicable,
            concerns,
        };
    }
    let validity = match delta.thermal_throttle_advanced() {
        Some(true) => {
            // ★ `unwrap_or(0)` here wrote "sw 0 µs" for a counter that was
            // never read. This line lands in the record and is the text a
            // human quotes when deciding how bad a throttle was; a counter
            // nobody could read must not appear in it as a measured zero.
            fn counter(v: Option<u64>) -> String {
                v.map_or_else(|| "unreadable".to_string(), |us| format!("{us} µs"))
            }
            concerns.push(format!(
                "a thermal throttle counter advanced during the run (sw {}, hw {}, \
                 brake {}) — this speed number is not comparable",
                counter(delta.sw_thermal_us),
                counter(delta.hw_thermal_us),
                counter(delta.hw_power_brake_us),
            ));
            Validity::Invalid
        }
        Some(false) => Validity::Valid,
        None => {
            concerns.push(
                "throttle counters were unreadable on at least one capture — this run is not \
                 known to have been unthrottled"
                    .to_string(),
            );
            Validity::Unknown
        }
    };
    Postcheck { validity, concerns }
}

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
