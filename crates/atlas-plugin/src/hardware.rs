// SPDX-License-Identifier: AGPL-3.0-only

//! Hardware fingerprint — what box a benchmark number was measured on.
//!
//! Results committed from different boxes are not comparable without it: a
//! TTFT median from a cold GB10 at idle clocks and the same model on a warm
//! desktop card differ for reasons the model never sees. Every gate record
//! carries one, fetched from the serving endpoint's `/hardware` so it always
//! describes the box that did the work, even when the benchmark CLI runs
//! somewhere else.
//!
//! Field names are deliberately vendor-neutral — the "sm-clock" reading is
//! nvidia-smi's `clocks.sm` on NVIDIA and the equivalent core-clock reading
//! anywhere else.

//! # Two layers
//!
//! * [`Hardware`] — WHICH box, as a fingerprint. Stable across a run, fetched
//!   from the serving endpoint, and the key a gate baseline is indexed by.
//! * [`state::HardwareState`] — WHAT STATE that box was in, captured before
//!   and after every benchmark. Added 2026-08-15, after two boxes with
//!   byte-identical [`Hardware`] fingerprints produced 692 s and 1079 s on the
//!   same gate and a "+38% regression" had to be retracted. See
//!   [`state`] for the incident and [`policy`] for what is gated on it.

use serde::{Deserialize, Serialize};

pub mod collect;
pub mod parse;
pub mod policy;
pub mod report;
pub mod state;
pub mod throttle_monitor;

pub use policy::{Decision, Sensitivity, Validity};
pub use report::HardwareStateReport;
pub use state::{HardwareState, HardwareStateDelta};

/// What the serving box reported about itself.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Hardware {
    /// Device model, e.g. "NVIDIA GB10". Empty when unknown.
    #[serde(default)]
    pub gpu: String,
    /// Driver version, e.g. "580.126.09". Empty when unknown.
    #[serde(default)]
    pub driver: String,
    /// Measured GPU sm-clock in MHz at probe time. `None` when the box cannot
    /// report one (some unified-memory parts and non-NVIDIA stacks answer
    /// nothing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_clock_mhz: Option<f64>,
    /// Where the reading came from ("nvidia-smi", "rocm-smi", "sysfs"), so a
    /// future reader knows how much to trust it. Empty for records written
    /// before the fingerprint existed.
    #[serde(default)]
    pub source: String,
}

impl Hardware {
    pub fn unknown() -> Self {
        Self::default()
    }

    /// The box-class key a gate baseline is indexed by, e.g. `"gb10"`.
    ///
    /// Derived from the reported GPU model, lowercased with the vendor prefix
    /// and separators dropped: `"NVIDIA GB10"` → `"gb10"`. A CLASS, not a host
    /// — two GB10 boxes share thresholds; a GB10 and an MI300 do not.
    ///
    /// An unknown fingerprint yields `"unknown"` rather than a guess or an
    /// empty string, so a baseline lookup for it FAILS with a name instead of
    /// silently matching some other box's entry. That matters because
    /// `fetch_hardware` degrades to `Hardware::unknown()` on every error path
    /// without surfacing one.
    pub fn gate_key(&self) -> String {
        if self.gpu.is_empty() {
            return "unknown".to_string();
        }
        let key: String = self
            .gpu
            .to_lowercase()
            .replace("nvidia", "")
            .replace("amd", "")
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect();
        if key.is_empty() {
            "unknown".to_string()
        } else {
            key
        }
    }

    /// True when no field carries information.
    pub fn is_unknown(&self) -> bool {
        self.gpu.is_empty() && self.driver.is_empty() && self.sm_clock_mhz.is_none()
    }

    /// One-line summary for reports: "NVIDIA GB10 · driver 580.126.09 · sm 208 MHz".
    pub fn one_line(&self) -> String {
        if self.is_unknown() {
            return "unknown hardware".to_string();
        }
        let mut parts = Vec::new();
        if !self.gpu.is_empty() {
            parts.push(self.gpu.clone());
        }
        if !self.driver.is_empty() {
            parts.push(format!("driver {}", self.driver));
        }
        if let Some(clock) = self.sm_clock_mhz {
            parts.push(format!("sm {clock:.0} MHz"));
        }
        parts.join(" · ")
    }

    /// Probe the local box. Tries each vendor tool in turn — `nvidia-smi`,
    /// then `rocm-smi`, then sysfs — and returns the first that answers.
    ///
    /// Never fails: a missing tool yields [`Hardware::unknown`], because a
    /// fingerprint is provenance, not a gate — a run must not be unrecordable
    /// on a box that simply lacks the reporting tool.
    pub fn probe() -> Self {
        nvidia_smi()
            .or_else(rocm_smi)
            .or_else(sysfs)
            .unwrap_or_else(Self::unknown)
    }
}

/// One tool's output, or nothing.
fn run(tool: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(tool)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// nvidia-smi answers all three fields in one CSV call; `[N/A]` cells (common
/// on unified-memory parts) simply leave that field empty.
fn nvidia_smi() -> Option<Hardware> {
    let line = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,clocks.sm",
            "--format=csv,noheader,nounits",
        ],
    )?
    .lines()
    .next()?
    .to_string();
    let mut cells = line.split(',').map(str::trim);
    let gpu = cells.next().unwrap_or_default().to_string();
    let driver = cells.next().unwrap_or_default().to_string();
    let sm_clock_mhz = cells
        .next()
        .filter(|v| *v != "[N/A]")
        .and_then(|v| v.parse().ok());
    Some(Hardware {
        gpu,
        driver,
        sm_clock_mhz,
        source: "nvidia-smi".into(),
    })
}

/// rocm-smi: two calls — one for the card + driver, one for the core clock
/// (the AMD equivalent of the sm-clock reading).
fn rocm_smi() -> Option<Hardware> {
    let card = run(
        "rocm-smi",
        &["--showproductname", "--showdriverversion", "--csv"],
    )?;
    let mut gpu = String::new();
    let mut driver = String::new();
    for line in card.lines().skip(1) {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        if cells.len() >= 3 {
            gpu = cells[1].to_string();
            driver = cells[2].to_string();
            break;
        }
    }
    let sm_clock_mhz = run("rocm-smi", &["--showclocks", "--csv"]).and_then(|out| {
        out.lines()
            .find_map(|l| l.split(',').nth(2))
            .and_then(|v| v.trim().parse().ok())
    });
    Some(Hardware {
        gpu,
        driver,
        sm_clock_mhz,
        source: "rocm-smi".into(),
    })
}

/// Last resort: the PCI class + IDs from sysfs. No driver version or clock —
/// those are vendor-tool territory — but the device alone still separates the
/// boxes (`0x10de:0x2e12` is unambiguous to anyone comparing two records).
fn sysfs() -> Option<Hardware> {
    for entry in std::fs::read_dir("/sys/bus/pci/devices").ok()?.flatten() {
        let path = entry.path();
        let Ok(class) = std::fs::read_to_string(path.join("class")) else {
            continue;
        };
        // 0x03 = display controller family.
        if !class.trim().starts_with("0x03") {
            continue;
        }
        let vendor = std::fs::read_to_string(path.join("vendor")).unwrap_or_default();
        let device = std::fs::read_to_string(path.join("device")).unwrap_or_default();
        return Some(Hardware {
            gpu: format!("pci:{}:{}", vendor.trim(), device.trim()),
            source: "sysfs".into(),
            ..Hardware::default()
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_every_field() {
        let hw = Hardware {
            gpu: "NVIDIA GB10".into(),
            driver: "580.126.09".into(),
            sm_clock_mhz: Some(208.0),
            source: "nvidia-smi".into(),
        };
        let back: Hardware = serde_json::from_str(&serde_json::to_string(&hw).unwrap()).unwrap();
        assert_eq!(hw, back);
    }

    #[test]
    fn missing_fields_default_to_unknown() {
        let hw: Hardware = serde_json::from_str("{}").unwrap();
        assert_eq!(hw, Hardware::unknown());
        assert_eq!(hw.one_line(), "unknown hardware");
    }

    #[test]
    fn one_line_lists_each_reported_measurement() {
        let hw = Hardware {
            gpu: "NVIDIA GB10".into(),
            driver: "580.126.09".into(),
            sm_clock_mhz: Some(208.0),
            source: "nvidia-smi".into(),
        };
        assert_eq!(
            hw.one_line(),
            "NVIDIA GB10 · driver 580.126.09 · sm 208 MHz"
        );
        assert_eq!(
            Hardware {
                gpu: hw.gpu.clone(),
                ..Hardware::default()
            }
            .one_line(),
            "NVIDIA GB10"
        );
        assert_eq!(
            Hardware {
                driver: hw.driver.clone(),
                ..Hardware::default()
            }
            .one_line(),
            "driver 580.126.09"
        );
        assert_eq!(
            Hardware {
                sm_clock_mhz: hw.sm_clock_mhz,
                ..Hardware::default()
            }
            .one_line(),
            "sm 208 MHz"
        );
    }
}

#[cfg(test)]
#[path = "hardware_gate_key_tests.rs"]
mod gate_key_tests;
