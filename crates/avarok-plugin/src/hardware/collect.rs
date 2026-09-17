// SPDX-License-Identifier: AGPL-3.0-only

//! The I/O half of the collector: run the tools, read the files, hand the text
//! to [`super::parse`].
//!
//! Nothing here decides anything. Every function is "get the bytes or don't",
//! so the only behaviour that needs a GPU to exercise is byte retrieval, and
//! the interpretation of those bytes is unit-tested against fixtures.
//!
//! **Never panics and never fails.** A missing tool, a permissions error, a
//! non-zero exit and a garbage answer all land as `None` on the field they
//! would have filled. A benchmark must stay runnable on a box that cannot
//! report its own state; what it must NOT do is record that box as healthy,
//! which is [`super::policy`]'s job to prevent.
//!
//! `nvidia-smi` normally answers in tens of milliseconds but can block for
//! seconds on a wedged driver. `std::process` has no timeout, so [`collect`]
//! is a BLOCKING call by contract — every async caller runs it through
//! `spawn_blocking`, exactly as `GET /hardware` already does for
//! [`super::Hardware::probe`].

use std::time::{SystemTime, UNIX_EPOCH};

use super::parse;
use super::state::{HardwareState, MachineIdentity, ThermalZone};

/// Run a tool and return its stdout, or nothing.
fn run(tool: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(tool)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).to_string())
        .filter(|s| !s.trim().is_empty())
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Every `/sys/class/thermal/thermal_zone*`, in numeric zone order.
///
/// `None` when the directory itself cannot be read — a box with no thermal
/// sysfs is a different fact from a box reporting zero zones, and the six
/// chassis zones are what separated 65 °C from 89 °C when nothing else did.
fn thermal_zones() -> Option<Vec<ThermalZone>> {
    let mut entries: Vec<_> = std::fs::read_dir("/sys/class/thermal")
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("thermal_zone"))
        })
        .collect();
    // Lexical order would put zone10 before zone2. The zones are reported
    // positionally in the incident table, so the order has to be stable.
    entries.sort_by_key(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.trim_start_matches("thermal_zone").parse::<u32>().ok())
            .unwrap_or(u32::MAX)
    });
    Some(
        entries
            .iter()
            .filter_map(|p| {
                let temp = std::fs::read_to_string(p.join("temp")).ok()?;
                let kind = std::fs::read_to_string(p.join("type")).ok();
                parse::thermal_zone(kind.as_deref(), &temp)
            })
            .collect(),
    )
}

/// The governor of cpu0. One core stands for the box: a mixed-governor machine
/// is a misconfiguration this field exists to make visible, and cpu0 is the
/// one that is always present.
fn cpu_governor() -> Option<String> {
    read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").map(|s| s.trim().to_string())
}

fn machine() -> MachineIdentity {
    MachineIdentity {
        // Read from procfs rather than shelling out to `hostname(1)`, which is
        // not installed everywhere and costs a process.
        hostname: read("/proc/sys/kernel/hostname")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        machine_id: read("/etc/machine-id")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
        gpu: None,
        driver: None,
    }
}

/// Capture the local box's state.
///
/// Field order below matches [`HardwareState`]; every source is independent,
/// so one tool being absent costs exactly the fields it owns.
pub fn collect() -> HardwareState {
    let mut sources = Vec::new();
    let mut state = HardwareState {
        captured_at: now_secs(),
        machine: machine(),
        ..HardwareState::default()
    };

    if let Some(text) = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,persistence_mode",
            "--format=csv,noheader,nounits",
        ],
    ) {
        let q = parse::gpu_query(&text);
        state.machine.gpu = q.name;
        state.machine.driver = q.driver;
        state.sm_clock_mhz = q.sm_clock_mhz;
        state.sm_clock_max_mhz = q.sm_clock_max_mhz;
        state.gpu_temp_c = q.gpu_temp_c;
        state.persistence_mode = q.persistence_mode;
        sources.push("nvidia-smi".to_string());
    }

    if let Some(text) = run("nvidia-smi", &["-q", "-d", "PERFORMANCE"]) {
        let (counters, active) = parse::performance(&text);
        state.throttle_counters = counters;
        state.throttle_active = active;
        if !sources.iter().any(|s| s == "nvidia-smi") {
            sources.push("nvidia-smi".to_string());
        }
    }

    // Deliberately NOT `--query-gpu=memory.used`: GB10 answers `[N/A]` there
    // (host and device share one pool), and the compute-apps query is the one
    // that reports the 87 GB a killed serve can leave behind.
    if let Some(text) = run(
        "nvidia-smi",
        &[
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    ) {
        state.gpu_compute_apps = Some(parse::compute_apps(&text));
    } else if run("nvidia-smi", &["--list-gpus"]).is_some() {
        // The tool works and the query answered nothing: the GPU is idle, and
        // an empty list is a real reading. Without this, an idle box would be
        // indistinguishable from an unreadable one and would never pass.
        state.gpu_compute_apps = Some(Vec::new());
    }

    if let Some(text) = read("/proc/meminfo") {
        let m = parse::meminfo(&text);
        state.mem_total_kb = m.total_kb;
        state.mem_available_kb = m.available_kb;
        state.page_cache_kb = m.cached_kb;
        sources.push("procfs".to_string());
    }

    state.chassis_temps_c = thermal_zones();
    if state.chassis_temps_c.is_some() {
        sources.push("sysfs".to_string());
    }
    state.cpu_governor = cpu_governor();

    state.sources = sources;
    state
}
