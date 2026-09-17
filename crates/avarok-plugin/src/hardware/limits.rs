// SPDX-License-Identifier: AGPL-3.0-only
//! A target's benchmark limits: the numbers a certification campaign and the
//! record policies judge a box class by, declared per hardware class in
//! `kernels/<hw>/HARDWARE.toml` under `[benchmarks.limits]`.
//!
//! Every number here is a fact about a box class, not about Avarok — a GB10's
//! `acpitz` zones read 55-76 °C under load and 89 °C in the throttled
//! 0.66 tok/s incident; its unified 121 GB pool freezes under co-tenancy at
//! ~15 % held by others; a 27B NVFP4 checkpoint loads in ~40 s from its page
//! cache. An H100 SXM has other sensors, another memory model and another
//! disk. So they live beside the target's other facts (arch, memory, serving
//! defaults), not in Rust constants named after one card. A target declares
//! the tables when someone has measured them; a target without them has NO
//! limits, and the campaign says so rather than borrowing another card's
//! (PCND). Every sub-table is required once `[benchmarks.limits]` exists: a
//! reader of the file sees every number the campaign will use.
//!
//! `HARDWARE.toml` is a closure input, so changing a limit re-opens the
//! target's gates: the bar and the measurement move together.

use std::path::Path;

use anyhow::{Context, Result, bail};

/// `[benchmarks.limits]` of one target.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub thermal: ThermalEnvelope,
    pub memory: MemoryLimits,
    pub timing: TimingLimits,
    pub equivalence: EquivalenceLimits,
}

/// `[benchmarks.limits.thermal]`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ThermalEnvelope {
    /// The hottest chassis zone at which the orchestrator parks a box, °C —
    /// and above which a Speed record's pre-run capture is suspect.
    pub chassis_park_c: f64,
    /// The zone a parked box must fall back to before it resumes, °C.
    pub chassis_resume_c: f64,
    /// The largest chassis difference at which two boxes of this class are
    /// still "one box" for a Speed-class measurement, °C.
    pub chassis_equivalence_delta_c: f64,
    /// GPU die temperature above which a Speed record's pre-run capture is
    /// suspect, °C.
    pub gpu_ceiling_c: f64,
}

/// `[benchmarks.limits.memory]`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryLimits {
    /// `MemAvailable / MemTotal` a gate needs before its server may start.
    pub min_free_fraction: f64,
}

/// `[benchmarks.limits.timing]`, seconds.
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimingLimits {
    /// Wall time a self-served gate may spend before its first sample —
    /// server start plus checkpoint load — before its deadline counts.
    pub serve_allowance_s: u64,
    /// How long a started server may take to name its model.
    pub boot_timeout_s: u64,
    /// The fixed cost every shard pays whatever its slice (start, warm-up,
    /// scoring), added to its share of the draw when planning.
    pub shard_overhead_s: u64,
    /// The least a shard is planned at, however thin the slice.
    pub shard_floor_s: u64,
    /// Time a remote node may spend building the anchor before its unit's
    /// deadline counts.
    pub build_allowance_s: u64,
}

/// `[benchmarks.limits.equivalence]`: the non-thermal half of "one box".
#[derive(Clone, Copy, Debug, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquivalenceLimits {
    /// `|a - b| / max(a, b)` on the clock ceiling.
    pub clock_spread: f64,
    /// `|a - b| / max(a, b)` on memory.
    pub mem_spread: f64,
}

#[derive(serde::Deserialize)]
struct HardwareToml {
    benchmarks: Option<Benchmarks>,
}

#[derive(serde::Deserialize)]
struct Benchmarks {
    limits: Option<Limits>,
}

/// The limits `hardware` declares, `None` when its `HARDWARE.toml` has no
/// `[benchmarks.limits]`.
///
/// # Errors
/// A missing or unparseable `HARDWARE.toml`, a `[benchmarks.limits]` missing
/// one of its tables or carrying a stray key, or values that contradict
/// themselves (resume above park, a non-positive tolerance, a fraction
/// outside `(0, 1)`).
pub fn limits(root: &Path, hardware: &str) -> Result<Option<Limits>> {
    let path = root.join("kernels").join(hardware).join("HARDWARE.toml");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let parsed: HardwareToml =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let Some(l) = parsed.benchmarks.and_then(|b| b.limits) else {
        return Ok(None);
    };
    let at = |what: &str| format!("{}: [benchmarks.limits] {what}", path.display());
    let t = l.thermal;
    for (name, v) in [
        ("thermal.chassis_park_c", t.chassis_park_c),
        ("thermal.chassis_resume_c", t.chassis_resume_c),
        (
            "thermal.chassis_equivalence_delta_c",
            t.chassis_equivalence_delta_c,
        ),
        ("thermal.gpu_ceiling_c", t.gpu_ceiling_c),
        ("memory.min_free_fraction", l.memory.min_free_fraction),
        ("equivalence.clock_spread", l.equivalence.clock_spread),
        ("equivalence.mem_spread", l.equivalence.mem_spread),
    ] {
        if !v.is_finite() {
            bail!("{}", at(&format!("{name} must be finite")));
        }
    }
    if t.chassis_resume_c >= t.chassis_park_c {
        bail!(
            "{}",
            at(&format!(
                "thermal.chassis_resume_c ({}) must be below chassis_park_c ({}) — without \
                 hysteresis a box flaps on the line",
                t.chassis_resume_c, t.chassis_park_c
            ))
        );
    }
    if t.chassis_equivalence_delta_c <= 0.0 {
        bail!(
            "{}",
            at("thermal.chassis_equivalence_delta_c must be positive")
        );
    }
    let f = l.memory.min_free_fraction;
    if !(f > 0.0 && f < 1.0) {
        bail!("{}", at("memory.min_free_fraction must be inside (0, 1)"));
    }
    if l.equivalence.clock_spread <= 0.0 || l.equivalence.mem_spread <= 0.0 {
        bail!("{}", at("equivalence spreads must be positive"));
    }
    let ti = l.timing;
    if ti.serve_allowance_s == 0 || ti.boot_timeout_s == 0 || ti.build_allowance_s == 0 {
        bail!("{}", at("timing allowances must be positive"));
    }
    Ok(Some(l))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB10: &str = "[benchmarks.limits.thermal]\nchassis_park_c = 80\nchassis_resume_c = 70\n\
        chassis_equivalence_delta_c = 15\ngpu_ceiling_c = 75\n[benchmarks.limits.memory]\n\
        min_free_fraction = 0.85\n[benchmarks.limits.timing]\nserve_allowance_s = 600\n\
        boot_timeout_s = 900\nshard_overhead_s = 420\nshard_floor_s = 300\nbuild_allowance_s = 1800\n\
        [benchmarks.limits.equivalence]\nclock_spread = 0.01\nmem_spread = 0.05\n";

    fn root_with(hw: &str, body: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("limits-{}-{}", std::process::id(), hw));
        std::fs::create_dir_all(d.join("kernels").join(hw)).unwrap();
        std::fs::write(d.join("kernels").join(hw).join("HARDWARE.toml"), body).unwrap();
        d
    }

    #[test]
    fn declared_limits_are_read_and_absent_ones_are_none() {
        let r = root_with("x1", &format!("[hardware]\nname = \"x1\"\n{GB10}"));
        let l = limits(&r, "x1").unwrap().unwrap();
        assert_eq!(
            (l.thermal.chassis_park_c, l.thermal.chassis_resume_c),
            (80.0, 70.0)
        );
        assert_eq!(l.memory.min_free_fraction, 0.85);
        assert_eq!(l.timing.serve_allowance_s, 600);
        assert_eq!(l.equivalence.clock_spread, 0.01);
        let r = root_with("x2", "[hardware]\nname = \"x2\"\n");
        assert_eq!(limits(&r, "x2").unwrap(), None);
        assert!(limits(&r, "nope").is_err(), "no file is an error, not None");
    }

    /// NEGATIVE CONTROLS: every way a table can be wrong is refused by name —
    /// a missing sub-table, a stray key, a contradiction.
    #[test]
    fn a_wrong_table_is_refused_by_name() {
        let r = root_with(
            "y1",
            &GB10.replace("[benchmarks.limits.memory]\nmin_free_fraction = 0.85\n", ""),
        );
        assert!(format!("{:#}", limits(&r, "y1").unwrap_err()).contains("memory"));
        let r = root_with("y2", &format!("{GB10}chassis_typo_c = 1\n"));
        assert!(limits(&r, "y2").is_err());
        let r = root_with(
            "y3",
            &GB10.replace("chassis_resume_c = 70", "chassis_resume_c = 85"),
        );
        assert!(
            limits(&r, "y3")
                .unwrap_err()
                .to_string()
                .contains("hysteresis")
        );
        let r = root_with(
            "y4",
            &GB10.replace("min_free_fraction = 0.85", "min_free_fraction = 1.5"),
        );
        assert!(
            limits(&r, "y4")
                .unwrap_err()
                .to_string()
                .contains("min_free_fraction")
        );
        let r = root_with(
            "y5",
            &GB10.replace(
                "chassis_equivalence_delta_c = 15",
                "chassis_equivalence_delta_c = 0",
            ),
        );
        assert!(limits(&r, "y5").is_err());
        let r = root_with(
            "y6",
            &GB10.replace("serve_allowance_s = 600", "serve_allowance_s = 0"),
        );
        assert!(
            limits(&r, "y6")
                .unwrap_err()
                .to_string()
                .contains("allowances")
        );
    }

    /// The committed GB10 table is the one the doctrine cites; targets nobody
    /// has measured declare nothing, and say so rather than borrowing it.
    #[test]
    fn gb10_declares_the_measured_limits_and_others_declare_none() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let l = limits(root, "gb10")
            .unwrap()
            .expect("gb10 declares [benchmarks.limits]");
        assert_eq!(
            (
                l.thermal.chassis_park_c,
                l.thermal.chassis_resume_c,
                l.thermal.chassis_equivalence_delta_c,
                l.thermal.gpu_ceiling_c
            ),
            (80.0, 70.0, 15.0, 75.0)
        );
        assert_eq!(l.memory.min_free_fraction, 0.85);
        assert_eq!(
            (
                l.timing.serve_allowance_s,
                l.timing.boot_timeout_s,
                l.timing.shard_overhead_s,
                l.timing.shard_floor_s,
                l.timing.build_allowance_s
            ),
            (600, 900, 420, 300, 1800)
        );
        assert_eq!(
            (l.equivalence.clock_spread, l.equivalence.mem_spread),
            (0.01, 0.05)
        );
        for hw in ["hopper", "b200", "strix", "strix-hip", "metal"] {
            assert_eq!(limits(root, hw).unwrap(), None, "{hw}");
        }
    }
}
