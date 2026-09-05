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
pub mod ids;
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
    /// How many GPUs the box exposed at probe time.
    ///
    /// The fingerprint was single-GPU by construction: `gpu` names ONE part,
    /// and every committed record was measured on a one-card GB10 where that
    /// was the whole truth. A multi-GPU A/B breaks it. The same recipe served
    /// at `--tp-size 2` and at `--tp-size 8` produces fingerprint-identical
    /// records whose numbers differ by the width of the node, and nothing in
    /// the file could tell a reader which one they were holding — the same
    /// class of defect as the two GB10 boxes that returned 692 s and 1079 s
    /// with byte-identical fingerprints (see [`HardwareStateReport`]).
    ///
    /// The topology FLAGS (`--tp-size`, `--ep-size`, `--world-size`) are
    /// already recorded — they are recipe keys, so they arrive through
    /// `GateRecord::served_by` and `serve_overrides` and are replayed in
    /// `command`. This is the other half: what the box actually offered, read
    /// from the box rather than from what was asked of it. A run pinned to
    /// `--tp-size 2` on an 8-GPU node is a different measurement from the same
    /// flags on a 2-GPU node, and only the pair says which.
    ///
    /// Additive and optional; the schema stays 1 and older records simply lack
    /// it, exactly as `GateRecord::dataset_fingerprint` did. `None` means
    /// UNMEASURED, never one: `Some(1)` on a box whose nvidia-smi was missing
    /// would claim a single-GPU topology for a run that may have used eight.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_count: Option<u32>,
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
        // A registered SKU answers for itself. Stripping punctuation is all
        // this could do before, and it turns "NVIDIA H100 80GB HBM3" into
        // `h10080gbhbm3` — a key no baseline defines, on a box whose baseline
        // slot is called `h100`. See [`ids::hardware_id_from_gpu_name`]; parts
        // the table has no entry for fall through to the normalisation below,
        // which is why `gb10` and every A100 capacity are unaffected.
        if let Some(id) = ids::hardware_id_from_gpu_name(&self.gpu) {
            return id.to_string();
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
            parts.push(match self.gpu_count {
                // A width worth naming goes in the FIRST field, beside the
                // part: a reader comparing two report lines has to see "×8"
                // before they compare the numbers, not after.
                Some(n) if n > 1 => format!("{} ×{n}", self.gpu),
                _ => self.gpu.clone(),
            });
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
    // A second call, because `--query-gpu` answers one ROW per GPU and this
    // query already took only the first: counting its rows would work, but it
    // would also silently become wrong the day someone adds a `head -1` to the
    // query above. `-L` asks the question directly.
    let gpu_count = run("nvidia-smi", &["-L"]).and_then(|out| parse::gpu_count(&out));
    Some(Hardware {
        gpu,
        driver,
        sm_clock_mhz,
        gpu_count,
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
        // Left UNMEASURED rather than assumed 1. `rocm-smi --showproductname`
        // does list a row per card and could be counted, but no AMD box has
        // ever produced a committed record, so an untested count here would be
        // a claim nobody has checked. None is the honest answer.
        gpu_count: None,
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
            gpu_count: Some(1),
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
            gpu_count: None,
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
    /// Oracle: the report line a reader compares two runs by.
    ///
    /// An 8-way node and a single card serving the same recipe produce report
    /// lines that were previously identical. The width has to be in the line,
    /// beside the part, or the comparison is between two things the reader
    /// believes are one thing.
    ///
    /// One GPU adds nothing: "×1" on every GB10 line would train readers to
    /// skip the field on the record where it matters. `None` (unmeasured) also
    /// adds nothing — it must not read as a width.
    #[test]
    fn a_multi_gpu_box_names_its_width_and_a_single_one_stays_quiet() {
        let node = |n| Hardware {
            gpu: "NVIDIA H100 80GB HBM3".into(),
            driver: "580.126.09".into(),
            gpu_count: n,
            ..Hardware::default()
        };
        assert_eq!(
            node(Some(8)).one_line(),
            "NVIDIA H100 80GB HBM3 ×8 · driver 580.126.09"
        );
        assert_eq!(
            node(Some(1)).one_line(),
            "NVIDIA H100 80GB HBM3 · driver 580.126.09"
        );
        assert_eq!(
            node(None).one_line(),
            "NVIDIA H100 80GB HBM3 · driver 580.126.09"
        );
    }

    /// Oracle: serde's own round trip, over the widened struct. The field is
    /// additive, so it has to survive the trip AND be absent from the JSON
    /// when unmeasured — a `"gpu_count": null` on every pre-existing record
    /// would be a diff across the whole `.benchmarks/` corpus for no reading.
    #[test]
    fn the_gpu_count_round_trips_and_stays_out_of_the_json_when_unmeasured() {
        let eight = Hardware {
            gpu: "NVIDIA H200".into(),
            gpu_count: Some(8),
            ..Hardware::default()
        };
        let json = serde_json::to_string(&eight).unwrap();
        assert_eq!(
            serde_json::from_str::<Hardware>(&json).unwrap().gpu_count,
            Some(8)
        );
        let unmeasured = Hardware {
            gpu: "NVIDIA GB10".into(),
            ..Hardware::default()
        };
        let json = serde_json::to_string(&unmeasured).unwrap();
        assert!(!json.contains("gpu_count"), "{json}");
        assert_eq!(
            serde_json::from_str::<Hardware>(&json).unwrap().gpu_count,
            None
        );
    }
}

#[cfg(test)]
#[path = "hardware_gate_key_tests.rs"]
mod gate_key_tests;
