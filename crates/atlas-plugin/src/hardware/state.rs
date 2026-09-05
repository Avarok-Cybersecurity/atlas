// SPDX-License-Identifier: AGPL-3.0-only

//! [`HardwareState`] — what the box was DOING while a benchmark ran, as data.
//!
//! [`super::Hardware`] answers "which box"; this answers "what state was that
//! box in". They are different questions and the second one is the one that
//! retracted a regression report on 2026-08-15:
//!
//! | box  | build              | Σwall (agentic-webserver) | correctness    |
//! |------|--------------------|---------------------------|----------------|
//! | dgx1 | ladder stack       | 692 s                     | 10/10 + 10/10  |
//! | dgx1 | pre-stack ref      | 773 s                     | 10/10 + 10/10  |
//! | dgx2 | ladder stack       | 1084 s / 1068 s           | 10/10 + 10/10  |
//! | dgx2 | unmodified main    | 1079 s                    | 10/10 + 10/10  |
//!
//! A "+38% regression" was reported and had to be withdrawn: dgx2 is ~56%
//! slower on that gate for ANY build, unmodified `main` included. The two
//! boxes are configured identically — driver 580.126.09, governor
//! `performance`, persistence on, the same 3003 MHz ceiling — so nothing in
//! [`super::Hardware`] separates them. What separates them is state:
//!
//! * chassis thermal zones 65/65/62/59/59/58 °C (dgx1) vs 89/88/82/74/71/70 °C;
//! * SW Thermal Slowdown 502 s in 11.2 days (0.05% of uptime) vs 2,914 s in
//!   16.1 h (5.0%) — ~96x the rate;
//! * HW Thermal Slowdown 0.7 s vs 228 s — the last-resort path, ~5000x;
//! * SM clock under load 2457–2496 MHz against a 3003 MHz max (82–83%).
//!
//! # Every field is `Option`, and an unknown is never a pass
//!
//! The collector runs on boxes with no `nvidia-smi`, no `/sys/class/thermal`,
//! and no cpufreq. It must never panic and must never substitute a value that
//! reads as healthy — a missing throttle counter is `None`, and
//! [`super::policy`] treats `None` as "not known to be healthy", never as
//! "healthy". Three GB10-specific traps make that concrete:
//!
//! * `nvidia-smi --query-gpu=memory.used` answers `[N/A]` on GB10 (unified
//!   memory). An idle-guard written against it never fires. System memory is
//!   read from `/proc/meminfo` instead, which does answer.
//! * A killed `spark serve` can leave the benchmark DRIVER process holding its
//!   allocation (87 GB observed), so a box that looks free by `nvidia-smi
//!   --query-gpu` is not. `--query-compute-apps` does answer, and is what
//!   [`HardwareState::foreign_compute_apps`] counts.
//! * Page cache held 60.7 GB and prevented a gate from booting at all, which
//!   is why `Cached` is recorded next to `MemAvailable` rather than folded
//!   into it.

use serde::{Deserialize, Serialize};

/// One `/sys/class/thermal/thermal_zone*` reading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThermalZone {
    /// The zone's `type` file, e.g. `"acpitz"`. Empty when unreadable.
    pub name: String,
    pub temp_c: f64,
}

/// Cumulative throttle time per reason, microseconds since driver load.
///
/// These are COUNTERS, not flags: the useful reading is the difference across
/// a run (see [`HardwareStateDelta`]), because a box can start cool and
/// throttle in the middle. `None` means the reason was not reported at all —
/// distinct from `Some(0)`, which means it was reported as never having fired.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThrottleCounters {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_power_cap_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_thermal_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_thermal_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_power_brake_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_boost_us: Option<u64>,
}

/// Throttle reasons asserted at the instant of capture.
///
/// The pre-run check reads these; the post-run check reads the counters above.
/// A reason that is `Active` right now is unambiguous — the box is throttling
/// before the benchmark has issued a single request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThrottleActive {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_power_cap: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_thermal: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_thermal: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_power_brake: Option<bool>,
}

impl ThrottleActive {
    /// True when any THERMAL reason is asserted. `None` when no thermal reason
    /// was reported, so the caller can tell "not throttling" from "no idea".
    ///
    /// SW power cap is excluded on purpose: on GB10 it is asserted for
    /// 16,130 s of an 11.2-day uptime on the HEALTHY box, i.e. it is the
    /// normal steady state of a power-limited part and says nothing about the
    /// thermal fault this check exists to find.
    pub fn thermal(&self) -> Option<bool> {
        match (self.sw_thermal, self.hw_thermal, self.hw_power_brake) {
            (None, None, None) => None,
            (sw, hw, brake) => {
                Some(sw.unwrap_or(false) || hw.unwrap_or(false) || brake.unwrap_or(false))
            }
        }
    }
}

/// One GPU-resident compute process, as `--query-compute-apps` reports it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuComputeApp {
    pub pid: u32,
    pub name: String,
    /// `None` when the driver answers `[N/A]`, which GB10 does for some
    /// queries even while answering this one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used_mib: Option<u64>,
}

/// Who this box is, for a per-class threshold lookup that does not exist yet.
///
/// [`super::Hardware::gate_key`] answers the CLASS (`"gb10"`), which is what
/// baselines are indexed by today. That is not enough: `agentic-webserver`'s
/// `wall_budget_s: 1000` is a dgx1 calibration that unmodified `main` cannot
/// meet on dgx2 (1079 s), and both boxes key to `"gb10"`. Recording the host
/// alongside the class is the prerequisite for resolving a wall bound per
/// PERFORMANCE class rather than per silicon class; this PR only records it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MachineIdentity {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    /// `/etc/machine-id`, which survives a rename where the hostname does not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
}

impl MachineIdentity {
    /// `"gb10@spark-256a"` — silicon class, then the individual box.
    ///
    /// Either half may be missing; an absent half reads `"unknown"` rather
    /// than being dropped, so two records with different missing halves never
    /// collapse to the same string.
    pub fn perf_class(&self) -> String {
        let class = self
            .gpu
            .as_deref()
            .map(|g| super::Hardware {
                gpu: g.to_string(),
                ..super::Hardware::default()
            })
            .map(|h| h.gate_key())
            .unwrap_or_else(|| "unknown".to_string());
        let host = self.hostname.as_deref().unwrap_or("unknown");
        format!("{class}@{host}")
    }
}

/// Everything the collector could read about the box at one instant.
///
/// Captured twice per run — before and after — because the delta is the
/// primary signal (see [`HardwareStateDelta`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HardwareState {
    /// Unix seconds. `0` only for a hand-built value in a test.
    #[serde(default)]
    pub captured_at: u64,
    #[serde(default)]
    pub machine: MachineIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_temp_c: Option<f64>,
    /// Every `/sys/class/thermal` zone, in zone order. `None` means the
    /// directory could not be read at all — NOT "this box has no zones".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chassis_temps_c: Option<Vec<ThermalZone>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_clock_mhz: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_clock_max_mhz: Option<f64>,
    #[serde(default)]
    pub throttle_counters: ThrottleCounters,
    #[serde(default)]
    pub throttle_active: ThrottleActive,
    /// `MemAvailable` from `/proc/meminfo`, kB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_available_kb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_total_kb: Option<u64>,
    /// `Cached`, kB — recorded separately because 60.7 GB of it once stopped a
    /// gate from booting while `MemAvailable` still counted it as available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_cache_kb: Option<u64>,
    /// `None` means the query failed; `Some(vec![])` means the GPU is idle.
    /// Those are different facts and the policy treats them differently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_compute_apps: Option<Vec<GpuComputeApp>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_governor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persistence_mode: Option<bool>,
    /// Which collectors answered, e.g. `["nvidia-smi", "sysfs", "procfs"]`.
    /// Empty means nothing did.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
}

impl HardwareState {
    /// Read the local box. Never fails, never panics; see [`super::collect`].
    pub fn collect() -> Self {
        super::collect::collect()
    }

    /// GPU compute processes that are not THIS process.
    ///
    /// `None` when the query did not answer — which the policy must not read
    /// as zero. The TUI serves in-process, so its own pid appears here and is
    /// excluded; a `--pull-request-gate` run's self-provisioned serve is a
    /// separate pid and counts as one. More than one is the signal: a box
    /// running a benchmark should host exactly one model, and the second
    /// entry is the leftover that starves the next run.
    pub fn foreign_compute_apps(&self) -> Option<usize> {
        let own = std::process::id();
        Some(
            self.gpu_compute_apps
                .as_ref()?
                .iter()
                .filter(|a| a.pid != own)
                .count(),
        )
    }

    /// The hottest chassis zone, which is the reading that separated the two
    /// boxes (65 °C vs 89 °C) when every other field matched.
    pub fn hottest_chassis_c(&self) -> Option<f64> {
        self.chassis_temps_c
            .as_ref()?
            .iter()
            .map(|z| z.temp_c)
            .fold(None::<f64>, |acc, t| Some(acc.map_or(t, |a| a.max(t))))
    }

    /// Current SM clock as a fraction of the box's own maximum, e.g. `0.82`.
    ///
    /// Only meaningful UNDER LOAD — an idle GB10 sits near 200 MHz against the
    /// same 3003 MHz ceiling, so a low value before a run says nothing. That
    /// is why no enabled check reads it; it is recorded for the after-capture,
    /// where the run was still finishing.
    pub fn clock_headroom(&self) -> Option<f64> {
        let max = self.sm_clock_max_mhz?;
        (max > 0.0).then(|| self.sm_clock_mhz.map(|c| c / max))?
    }

    /// One line for a run log.
    pub fn one_line(&self) -> String {
        let mut parts = vec![self.machine.perf_class()];
        if let Some(t) = self.gpu_temp_c {
            parts.push(format!("gpu {t:.0} °C"));
        }
        if let Some(t) = self.hottest_chassis_c() {
            parts.push(format!("chassis max {t:.0} °C"));
        }
        match (self.sm_clock_mhz, self.sm_clock_max_mhz) {
            (Some(c), Some(m)) => parts.push(format!("sm {c:.0}/{m:.0} MHz")),
            (Some(c), None) => parts.push(format!("sm {c:.0} MHz")),
            _ => {}
        }
        match self.foreign_compute_apps() {
            Some(n) => parts.push(format!("{n} foreign gpu proc")),
            None => parts.push("foreign gpu proc unknown".to_string()),
        }
        if let Some(kb) = self.mem_available_kb {
            parts.push(format!("{:.1} GiB avail", kb as f64 / 1_048_576.0));
        }
        parts.join(" · ")
    }
}

/// What changed across the run — the primary signal.
///
/// A box that starts cool and throttles at minute 40 produces a perfectly
/// healthy pre-state and a meaningless number. "Did a throttle counter advance
/// while this benchmark was running" is the measurement-validity question, and
/// it is answerable from two captures and nothing else.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HardwareStateDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_s: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_power_cap_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_thermal_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_thermal_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_power_brake_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_temp_delta_c: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hottest_chassis_delta_c: Option<f64>,
}

/// `after - before`, saturating: a counter that went BACKWARDS means the
/// driver was reloaded mid-run, which is not "zero throttling" — it is an
/// unusable reading, so it yields `None` rather than `0`.
fn advance(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    let (b, a) = (before?, after?);
    // `checked_sub`, not `(a >= b).then_some(a - b)`: the argument to
    // `then_some` is evaluated eagerly, so the backwards case panics in a
    // debug build before the guard ever runs.
    a.checked_sub(b)
}

impl HardwareStateDelta {
    pub fn between(before: &HardwareState, after: &HardwareState) -> Self {
        let (b, a) = (&before.throttle_counters, &after.throttle_counters);
        Self {
            elapsed_s: after.captured_at.checked_sub(before.captured_at),
            sw_power_cap_us: advance(b.sw_power_cap_us, a.sw_power_cap_us),
            sw_thermal_us: advance(b.sw_thermal_us, a.sw_thermal_us),
            hw_thermal_us: advance(b.hw_thermal_us, a.hw_thermal_us),
            hw_power_brake_us: advance(b.hw_power_brake_us, a.hw_power_brake_us),
            gpu_temp_delta_c: Option::zip(before.gpu_temp_c, after.gpu_temp_c).map(|(b, a)| a - b),
            hottest_chassis_delta_c: Option::zip(
                before.hottest_chassis_c(),
                after.hottest_chassis_c(),
            )
            .map(|(b, a)| a - b),
        }
    }

    /// Did any THERMAL throttle reason accumulate time during the run?
    ///
    /// `None` when no thermal counter was readable on both captures — the
    /// policy must render that as "unknown", never as "no".
    ///
    /// SW power cap is excluded for the same reason as in
    /// [`ThrottleActive::thermal`]: on the healthy box it advances constantly
    /// and carries no fault information.
    pub fn thermal_throttle_advanced(&self) -> Option<bool> {
        let counters = [
            self.sw_thermal_us,
            self.hw_thermal_us,
            self.hw_power_brake_us,
        ];
        if counters.into_iter().flatten().any(|value| value > 0) {
            return Some(true);
        }
        counters
            .into_iter()
            .all(|value| value == Some(0))
            .then_some(false)
    }

    /// True when every counter [`Self::thermal_throttle_fraction`] sums was
    /// readable on BOTH captures.
    ///
    /// When false the fraction is a lower bound, and the caller must say so:
    /// an unreadable counter contributes nothing to the sum, which moves the
    /// answer in the reassuring direction.
    pub fn thermal_counters_complete(&self) -> bool {
        self.sw_thermal_us.is_some()
            && self.hw_thermal_us.is_some()
            && self.hw_power_brake_us.is_some()
    }

    /// The throttled fraction of the run, for the record's summary line.
    ///
    /// `None` when the run has no measured duration, or when NOT ONE thermal
    /// counter was readable across both captures. A LOWER BOUND when only some
    /// were — see [`Self::thermal_counters_complete`].
    ///
    /// ★ This used to be `unwrap_or(0)` over all three counters, which is the
    /// same mistake [`Self::thermal_throttle_advanced`] exists to avoid one
    /// method up: an unreadable counter became a measured zero. With nothing
    /// readable it returned `Some(0.0)` — "this run did not throttle" — from a
    /// box that reported nothing at all, and that sentence is persisted in the
    /// record's `hardware_state.postcheck.concerns` and quoted later as
    /// evidence that a speed number is comparable.
    pub fn thermal_throttle_fraction(&self) -> Option<f64> {
        let secs = self.elapsed_s.filter(|s| *s > 0)? as f64;
        let counters = [
            self.sw_thermal_us,
            self.hw_thermal_us,
            self.hw_power_brake_us,
        ];
        if counters.iter().all(Option::is_none) {
            return None;
        }
        let us: u64 = counters.into_iter().flatten().sum();
        Some(us as f64 / 1e6 / secs)
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
