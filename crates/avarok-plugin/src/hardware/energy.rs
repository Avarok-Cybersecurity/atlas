// SPDX-License-Identifier: AGPL-3.0-only

//! In-window GPU energy: the pure half. A background sampler
//! ([`super::energy_sampler`]) streams `(instant, watts, cap flags)` readings;
//! this integrates the ones inside a measured window and turns them into
//! record keys. No I/O here — every rule is provable against hand-built
//! samples on a box with no GPU.
//!
//! # Store joules and tokens, derive the ratios
//!
//! Joules and tokens are additive and survive re-aggregation; a ratio does
//! not. So the record carries `energy_j` and the window's token count, and
//! tokens/J (which IS tokens/W: `(tok/s) / (J/s)`) and `$ per 1M tokens =
//! (J/token) × ($/kWh) / 3.6` are downstream arithmetic.
//!
//! ONE ratio is also written: `gpu_rail_joules_per_token`, because a
//! BENCH.toml ceiling can only bound a key the record carries (owner
//! requirement 2026-09-23: a J/token ceiling per concurrency rung). It rides
//! BESIDE its pair, never instead of it — anything that re-aggregates
//! windows must still add joules and tokens and divide once
//! ([`joules_per_token`] is that one division).
//!
//! # ★ WHICH RAIL — read before comparing this to any other number
//!
//! On GB10 `nvidia-smi` exposes the **GPU rail only**. `power.limit`, the
//! Module Power Readings and GPU Memory Power all answer `N/A`, and
//! `Power Samples` reports `Not Found` (measured 2026-09-20, driver
//! 580.126.09). The module — Grace cores + LPDDR5X — is UNREADABLE, and on
//! a unified-memory part whose decode is bandwidth-bound a dominant share
//! of the real energy is outside this number. Every key this module emits
//! carries `gpu_rail` in its name so no reader later sets it beside a
//! discrete card's board power and calls it a comparison.
//!
//! # `power.draw.average`, not `.instant`
//!
//! The two differed by 22 W in one call on this box. `average` is the
//! driver's own short moving average; sampling it at [`SAMPLE_PERIOD_MS`]
//! and holding each reading forward to the next is the integration rule
//! below, and the record says how many samples it stood on — a joule count
//! without its sample count is unauditable.
//!
//! # SW Power Cap is the NORMAL state
//!
//! This box sits in `SW Power Cap` as its steady state under load, so a
//! watt figure without the cap state is ambiguous. The fraction of samples
//! with the cap and the HW power brake asserted rides beside every window.

use std::collections::BTreeMap;
use std::time::Instant;

/// Sampler cadence. A pinned instrument constant, not a tunable: the
/// integration error at the window edges is bounded by one period, and a
/// recorded joule count is only comparable to another taken at the same
/// cadence.
pub const SAMPLE_PERIOD_MS: u64 = 250;

/// How long the idle baseline is sampled before the first measured window
/// (eight samples at the pinned cadence).
pub const IDLE_BASELINE_SECS: u64 = 2;

/// The one sentence every energy-carrying record logs beside its numbers.
pub const RAIL_NOTE: &str = "energy: GPU RAIL ONLY (nvidia-smi power.draw.average, 250 ms cadence). \
     On GB10 the module rail — Grace cores + LPDDR5X — reads N/A and is NOT in this number; on a \
     bandwidth-bound unified-memory decode a dominant share of real energy is outside it. Do not \
     compare to a discrete card's board power.";

/// One reading from the sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerSample {
    pub at: Instant,
    pub power_w: f64,
    /// `clocks_event_reasons.sw_power_cap`; `None` when the driver did not
    /// report it (never "not active").
    pub sw_power_cap: Option<bool>,
    /// `clocks_event_reasons.hw_power_brake_slowdown`.
    pub hw_power_brake: Option<bool>,
}

/// One CSV row of
/// `--query-gpu=power.draw.average,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_power_brake_slowdown`
/// with `--format=csv,noheader,nounits`, e.g. `4.76, Not Active, Active`.
///
/// `None` when the watt cell is not a finite number — a `[N/A]` rail must
/// not integrate as zero watts. A missing or unrecognised flag cell leaves
/// that flag unknown rather than "not active".
pub fn parse_line(line: &str, at: Instant) -> Option<PowerSample> {
    let cells: Vec<&str> = line.split(',').map(str::trim).collect();
    let power_w = cells
        .first()?
        .parse::<f64>()
        .ok()
        .filter(|w| w.is_finite() && *w >= 0.0)?;
    let flag = |i: usize| match cells.get(i).copied() {
        Some("Active") => Some(true),
        Some("Not Active") => Some(false),
        _ => None,
    };
    Some(PowerSample {
        at,
        power_w,
        sw_power_cap: flag(1),
        hw_power_brake: flag(2),
    })
}

/// The energy integral over one measured window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnergyWindow {
    /// Window length the joules span, seconds.
    pub window_s: f64,
    /// Readings the integral stood on.
    pub samples: usize,
    /// `Σ P_i · dt_i` — each reading held forward to the next, the first
    /// back-filled to the window start, the last held to the window end, so
    /// `Σ dt_i` is exactly `window_s`.
    pub energy_j: f64,
    /// `energy_j / window_s`.
    pub mean_power_w: f64,
    pub max_power_w: f64,
    /// Fraction of readings (by count, not time) with the SW power cap
    /// asserted; `None` when no reading reported the flag.
    pub sw_power_cap_frac: Option<f64>,
    /// Same, for the HW power brake.
    pub hw_power_brake_frac: Option<f64>,
}

/// Integrate the readings inside `[start, end]`.
///
/// `None` when the window is empty or no reading falls inside it — a
/// window with no evidence must not report zero joules.
pub fn integrate(samples: &[PowerSample], start: Instant, end: Instant) -> Option<EnergyWindow> {
    if end <= start {
        return None;
    }
    let inside: Vec<&PowerSample> = samples
        .iter()
        .filter(|s| s.at >= start && s.at <= end)
        .collect();
    if inside.is_empty() {
        return None;
    }
    let window_s = end.duration_since(start).as_secs_f64();
    let mut energy_j = 0.0;
    let mut max_power_w = 0.0f64;
    for (i, s) in inside.iter().enumerate() {
        let from = if i == 0 { start } else { s.at };
        let to = inside.get(i + 1).map_or(end, |n| n.at);
        energy_j += s.power_w * to.duration_since(from).as_secs_f64();
        max_power_w = max_power_w.max(s.power_w);
    }
    let frac = |pick: fn(&PowerSample) -> Option<bool>| {
        let reported: Vec<bool> = inside.iter().filter_map(|s| pick(s)).collect();
        (!reported.is_empty())
            .then(|| reported.iter().filter(|b| **b).count() as f64 / reported.len() as f64)
    };
    Some(EnergyWindow {
        window_s,
        samples: inside.len(),
        energy_j,
        mean_power_w: energy_j / window_s,
        max_power_w,
        sw_power_cap_frac: frac(|s| s.sw_power_cap),
        hw_power_brake_frac: frac(|s| s.hw_power_brake),
    })
}

/// GPU-rail joules per delivered output token, or `None` when the pair
/// cannot support a ratio: no tokens (the division is undefined), or a joule
/// count that is not a finite positive number (a delivered token never costs
/// zero energy, so 0 J means the window measured nothing). `None` rather than
/// 0.0 or ∞ because this value is bounded by a ceiling, and every sentinel a
/// ceiling can compare against would read as a pass or as a real reading.
pub fn joules_per_token(energy_j: f64, tokens: usize) -> Option<f64> {
    (tokens > 0 && energy_j.is_finite() && energy_j > 0.0).then(|| energy_j / tokens as f64)
}

impl EnergyWindow {
    /// Joules above what the box would have drawn idle for the same
    /// duration: `energy_j − idle.mean_power_w × window_s`. Derived, so it
    /// is defined once here; can be negative when the window drew less
    /// than the baseline, and that is reported rather than clamped.
    pub fn above_idle_j(&self, idle: &EnergyWindow) -> f64 {
        self.energy_j - idle.mean_power_w * self.window_s
    }

    /// Several windows as one: joules, seconds and samples add; the mean
    /// is re-derived; the flag fractions are sample-weighted. `None` for
    /// no windows.
    pub fn sum(windows: &[EnergyWindow]) -> Option<EnergyWindow> {
        if windows.is_empty() {
            return None;
        }
        let window_s: f64 = windows.iter().map(|w| w.window_s).sum();
        let energy_j: f64 = windows.iter().map(|w| w.energy_j).sum();
        let samples: usize = windows.iter().map(|w| w.samples).sum();
        let weighted = |pick: fn(&EnergyWindow) -> Option<f64>| {
            let (num, den) = windows
                .iter()
                .fold((0.0, 0usize), |(n, d), w| match pick(w) {
                    Some(f) => (n + f * w.samples as f64, d + w.samples),
                    None => (n, d),
                });
            (den > 0).then(|| num / den as f64)
        };
        Some(EnergyWindow {
            window_s,
            samples,
            energy_j,
            mean_power_w: if window_s > 0.0 {
                energy_j / window_s
            } else {
                0.0
            },
            max_power_w: windows.iter().map(|w| w.max_power_w).fold(0.0, f64::max),
            sw_power_cap_frac: weighted(|w| w.sw_power_cap_frac),
            hw_power_brake_frac: weighted(|w| w.hw_power_brake_frac),
        })
    }

    /// The record keys for one window, under `prefix` (`"c8_"` or `""`).
    /// `gpu_rail` is in every name on purpose — see the module docs.
    /// `tokens` is the output-token count delivered INSIDE this window: it
    /// rides beside the joules because a joule count with no denominator is
    /// not usable, and J/token is written from the pair (see the module docs
    /// for why that one ratio is stored).
    pub fn metrics(
        &self,
        prefix: &str,
        tokens: usize,
        idle: Option<&EnergyWindow>,
        m: &mut BTreeMap<String, f64>,
    ) {
        let put = |m: &mut BTreeMap<String, f64>, k: &str, v: f64| {
            m.insert(format!("{prefix}{k}"), v);
        };
        put(m, "gpu_rail_energy_j", self.energy_j);
        put(m, "gpu_rail_energy_window_tokens", tokens as f64);
        // Absent — never 0, inf or NaN — when the window cannot support the
        // ratio, so a J/token ceiling on it fails as "missing from the
        // record" instead of passing on an unmeasured value.
        if let Some(r) = joules_per_token(self.energy_j, tokens) {
            put(m, "gpu_rail_joules_per_token", r);
        }
        put(m, "gpu_rail_mean_power_w", self.mean_power_w);
        put(m, "gpu_rail_max_power_w", self.max_power_w);
        put(m, "gpu_rail_power_samples", self.samples as f64);
        put(m, "gpu_rail_energy_window_s", self.window_s);
        if let Some(f) = self.sw_power_cap_frac {
            put(m, "gpu_rail_sw_power_cap_frac", f);
        }
        if let Some(f) = self.hw_power_brake_frac {
            put(m, "gpu_rail_hw_power_brake_frac", f);
        }
        if let Some(idle) = idle {
            put(m, "gpu_rail_energy_above_idle_j", self.above_idle_j(idle));
        }
    }

    /// One log line: total and above-idle joules, mean watts, samples, cap
    /// state.
    pub fn one_line(&self, idle: Option<&EnergyWindow>) -> String {
        let mut s = format!(
            "gpu rail {:.1} J over {:.1} s ({:.1} W mean, {:.1} W max, {} samples)",
            self.energy_j, self.window_s, self.mean_power_w, self.max_power_w, self.samples
        );
        if let Some(idle) = idle {
            s.push_str(&format!(
                " · {:.1} J above the {:.1} W idle baseline",
                self.above_idle_j(idle),
                idle.mean_power_w
            ));
        }
        match self.sw_power_cap_frac {
            Some(f) => s.push_str(&format!(" · sw power cap {:.0}% of samples", f * 100.0)),
            None => s.push_str(" · sw power cap unreported"),
        }
        if let Some(f) = self.hw_power_brake_frac.filter(|f| *f > 0.0) {
            s.push_str(&format!(" · HW POWER BRAKE {:.0}% of samples", f * 100.0));
        }
        s
    }
}

/// What the sampler cost, for the record: the instrument must not perturb
/// the measurement it annotates, and the cost is reported rather than
/// assumed negligible.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplerCost {
    /// CPU seconds the sampler child consumed over its lifetime
    /// (`/proc/<pid>/schedstat`, nanosecond-resolution, no tick assumption).
    /// `None` when procfs could not be read.
    pub cpu_s: Option<f64>,
    /// Wall seconds it ran.
    pub wall_s: f64,
    /// Readings it produced.
    pub samples: usize,
    /// Lines it emitted that did not parse as a reading — reported, never
    /// swallowed; a non-zero count means the rail was intermittently N/A.
    pub rejected_lines: u64,
}

impl SamplerCost {
    pub fn metrics(&self, m: &mut BTreeMap<String, f64>) {
        m.insert(
            "gpu_rail_sample_period_ms".to_string(),
            SAMPLE_PERIOD_MS as f64,
        );
        if let Some(cpu) = self.cpu_s {
            m.insert("gpu_rail_sampler_cpu_s".to_string(), cpu);
        }
        m.insert("gpu_rail_sampler_wall_s".to_string(), self.wall_s);
        if self.rejected_lines > 0 {
            m.insert(
                "gpu_rail_sampler_rejected_lines".to_string(),
                self.rejected_lines as f64,
            );
        }
    }

    pub fn one_line(&self) -> String {
        format!(
            "energy sampler cost: {} CPU over {:.0} s wall ({} samples at {SAMPLE_PERIOD_MS} ms{})",
            self.cpu_s
                .map(|c| format!("{c:.3} s"))
                .unwrap_or_else(|| "unmeasured".into()),
            self.wall_s,
            self.samples,
            if self.rejected_lines > 0 {
                format!(", {} unparseable lines", self.rejected_lines)
            } else {
                String::new()
            }
        )
    }
}

#[cfg(test)]
#[path = "energy_tests.rs"]
mod tests;
