// SPDX-License-Identifier: AGPL-3.0-only
//! Which nodes may take part, and what each is for the scheduler.
//!
//! Admission is strict on purpose: a node is a machine that will sign
//! records this repository certifies with, so everything that would make a
//! record unusable afterwards — a signer nobody committed, a box class other
//! than the one being certified, a busy or unhealthy box — is refused HERE,
//! before any unit is sent, with the reason. A fact a node cannot report is
//! a refusal, not a pass.

use atlas_plugin::hardware::equivalence::{HardwareFingerprint, driver_major};
use atlas_plugin::hardware::{Hardware, HardwareState};

use super::atlasctl::{NodeInfo, NodeRow};

/// A node the campaign may run on.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// The address as given to `--with-nodes`, or `local`.
    pub addr: String,
    /// The node's display name.
    pub name: String,
    /// Fingerprint of the atlasctl node (empty for local).
    pub node_id: String,
    /// The signing key its records will carry.
    pub signer: String,
    pub hardware: HardwareFingerprint,
    /// `MemAvailable / MemTotal` at admission; the tie-breaker for the box
    /// a bundled Speed set goes to.
    pub free_fraction: Option<f64>,
    /// Whether the anchor is already built there (no build allowance needed).
    pub built: bool,
    pub local: bool,
}

/// Why a node was not admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejection {
    pub addr: String,
    pub why: String,
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.addr, self.why)
    }
}

/// The fingerprint a node's report implies. Every absent fact is `None`,
/// which `equivalent` treats as undecidable.
pub fn fingerprint_of(info: &NodeInfo) -> HardwareFingerprint {
    let t = info.thermal.as_ref();
    HardwareFingerprint {
        gpu: info
            .gpu
            .as_ref()
            .map(|g| g.name.clone())
            .unwrap_or_default(),
        driver_major: info
            .gpu
            .as_ref()
            .and_then(|g| driver_major(&g.driver_version)),
        sm_clock_max_mhz: t.and_then(|t| t.sm_clock_max_mhz),
        mem_total_kb: t.and_then(|t| t.mem_total_kb),
        thermal_alert: t.and_then(|t| t.throttle_thermal),
        hottest_chassis_c: t.and_then(|t| {
            t.chassis_temps_c
                .iter()
                .copied()
                .fold(None, |m: Option<f64>, x| Some(m.map_or(x, |m| m.max(x))))
        }),
        postcheck_valid: None,
    }
}

/// What admission needs to know about this side.
pub struct Wanted<'a> {
    /// The box class being certified (`--hardware`).
    pub hardware: &'a str,
    /// Fingerprints committed in `.github/record-signers/`.
    pub committed_signers: &'a [String],
    /// The anchor, to tell whether a node has it built.
    pub anchor: &'a str,
    /// The free-memory floor a gate needs to start.
    pub min_free_fraction: f64,
}

/// Judge one row. Every reason is collected, not just the first.
pub fn admit(row: &NodeRow, wanted: &Wanted) -> Result<Node, Rejection> {
    let reject = |why: String| Rejection {
        addr: row.node.clone(),
        why,
    };
    let Some(info) = &row.info else {
        return Err(reject(match &row.error {
            Some(e) => e.to_string(),
            None => "no report and no error — atlasctl said nothing about it".into(),
        }));
    };
    let mut why = Vec::new();
    if !info.bench_enabled {
        why.push(format!(
            "bench is off there ({})",
            info.disabled_reason.as_deref().unwrap_or("no reason given")
        ));
    }
    match info.hardware_class.as_deref() {
        Some(c) if c == wanted.hardware => {}
        Some(c) => why.push(format!(
            "its box class is {c}, this campaign certifies {}",
            wanted.hardware
        )),
        None => why.push("it reports no box class".into()),
    }
    match &info.signer_fp {
        Some(fp) if wanted.committed_signers.iter().any(|s| s == fp) => {}
        Some(fp) => why.push(format!(
            "its signer {fp} is not committed in .github/record-signers/ (commit {fp}.pub first)"
        )),
        None => why.push("it reports no signing identity (no ATLAS_HOME identity there)".into()),
    }
    if info.busy {
        why.push(format!(
            "it is busy ({})",
            info.busy_reason.as_deref().unwrap_or("no reason given")
        ));
    }
    if info.queued > 0 {
        why.push(format!("it already has {} job(s) queued", info.queued));
    }
    let free = info.host_free_fraction.value();
    match free {
        Some(f) if f < wanted.min_free_fraction => why.push(format!(
            "only {:.0} % of host memory is free; a gate needs {:.0} %",
            f * 100.0,
            wanted.min_free_fraction * 100.0
        )),
        Some(_) => {}
        None => why.push("it cannot report free memory".into()),
    }
    if let Some(d) = info.disk_free_bytes.value()
        && (d as u64) < info.min_free_disk_bytes
    {
        why.push(format!(
            "only {} MiB free on its bench cache; it requires {} MiB",
            (d as u64) >> 20,
            info.min_free_disk_bytes >> 20
        ));
    }
    let hw = fingerprint_of(info);
    if hw.gpu.is_empty() {
        why.push("it reports no GPU".into());
    }
    if !why.is_empty() {
        return Err(reject(why.join("; ")));
    }
    Ok(Node {
        addr: row.node.clone(),
        name: info.name.clone(),
        node_id: info.node.clone(),
        signer: info.signer_fp.clone().unwrap_or_default(),
        hardware: hw,
        free_fraction: free,
        built: info.built_shas.iter().any(|b| b.sha == wanted.anchor),
        local: false,
    })
}

/// This machine as a node.
pub fn local(signer: &str, hardware: &Hardware, state: &HardwareState) -> Node {
    Node {
        addr: "local".into(),
        name: state
            .machine
            .hostname
            .clone()
            .unwrap_or_else(|| "local".into()),
        node_id: String::new(),
        signer: signer.to_owned(),
        hardware: HardwareFingerprint::from_live(hardware, state),
        free_fraction: match (state.mem_available_kb, state.mem_total_kb) {
            (Some(a), Some(t)) if t > 0 => Some(a as f64 / t as f64),
            _ => None,
        },
        built: true,
        local: true,
    }
}

#[cfg(test)]
#[path = "node_tests.rs"]
mod node_tests;
