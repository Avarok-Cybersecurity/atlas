// SPDX-License-Identifier: AGPL-3.0-only
//! The shapes `atlasctl bench … --json` writes, as this side reads them
//! (atlas-recipes `docs/BENCH.md`). Every optional fact defaults, so an
//! atlasctl one version away still parses — and admission treats an absent
//! fact as a refusal, never as a pass.

use std::path::PathBuf;

use serde::Deserialize;

/// atlasctl's exit codes, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    Done,
    Usage,
    Unreachable,
    NotPaired,
    Refused,
    JobFailed,
    Cancelled,
    StreamLost,
    Unsupported,
    /// Killed, or a code this build does not know.
    Other(Option<i32>),
}

impl Exit {
    pub fn from_code(code: Option<i32>) -> Self {
        match code {
            Some(0) => Self::Done,
            Some(1) => Self::Usage,
            Some(2) => Self::Unreachable,
            Some(3) => Self::NotPaired,
            Some(4) => Self::Refused,
            Some(5) => Self::JobFailed,
            Some(6) => Self::Cancelled,
            Some(7) => Self::StreamLost,
            Some(8) => Self::Unsupported,
            other => Self::Other(other),
        }
    }
}

/// atlasctl's error document.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Default)]
pub struct ErrorObj {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub node: Option<String>,
    #[serde(default)]
    pub node_id: Option<String>,
    #[serde(default)]
    pub fix: Option<String>,
    #[serde(default)]
    pub retryable: bool,
}

impl std::fmt::Display for ErrorObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)?;
        if let Some(fix) = &self.fix {
            write!(f, " (fix: {fix})")?;
        }
        Ok(())
    }
}

/// A `{"state":"reading","value":x}` / `{"state":"unsupported"}` metric.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Default)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Metric {
    Reading {
        value: f64,
    },
    #[default]
    Unsupported,
}

impl Metric {
    pub fn value(self) -> Option<f64> {
        match self {
            Self::Reading { value } => Some(value),
            Self::Unsupported => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct GpuInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub count: u32,
    #[serde(default)]
    pub driver_version: String,
    #[serde(default)]
    pub cuda_version: String,
    #[serde(default)]
    pub sm_clock_mhz: Metric,
    #[serde(default)]
    pub temperature_c: Metric,
    #[serde(default)]
    pub memory_total_bytes: Metric,
    #[serde(default)]
    pub memory_used_frac: Metric,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct HostThermal {
    #[serde(default)]
    pub chassis_temps_c: Vec<f64>,
    #[serde(default)]
    pub throttle_thermal: Option<bool>,
    #[serde(default)]
    pub sm_clock_max_mhz: Option<f64>,
    #[serde(default)]
    pub mem_total_kb: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct RepoInfo {
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub remote_name: String,
    #[serde(default)]
    pub remote_url: String,
    #[serde(default)]
    pub head_sha: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct BuiltSha {
    pub sha: String,
}

/// What a node says about itself. Every field defaults, so a newer or older
/// atlasctl that adds or drops one still parses — and the admission rules
/// treat an absent fact as a refusal, never as a pass.
#[derive(Clone, Debug, Deserialize, PartialEq, Default)]
pub struct NodeInfo {
    #[serde(default)]
    pub node: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub agent_version: String,
    #[serde(default)]
    pub bench_enabled: bool,
    #[serde(default)]
    pub disabled_reason: Option<String>,
    #[serde(default)]
    pub gpu: Option<GpuInfo>,
    #[serde(default)]
    pub thermal: Option<HostThermal>,
    #[serde(default)]
    pub hardware_class: Option<String>,
    #[serde(default)]
    pub atlas_repo: Option<RepoInfo>,
    #[serde(default)]
    pub atlas_home: Option<String>,
    #[serde(default)]
    pub signer_fp: Option<String>,
    #[serde(default)]
    pub built_shas: Vec<BuiltSha>,
    #[serde(default)]
    pub busy: bool,
    #[serde(default)]
    pub busy_reason: Option<String>,
    #[serde(default)]
    pub queued: u32,
    #[serde(default)]
    pub queue_depth: u32,
    #[serde(default)]
    pub host_free_fraction: Metric,
    #[serde(default)]
    pub disk_free_bytes: Metric,
    #[serde(default)]
    pub min_free_disk_bytes: u64,
    #[serde(default)]
    pub max_run_s: u32,
}

/// One row of `bench nodes --json`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct NodeRow {
    pub node: String,
    pub ok: bool,
    #[serde(default)]
    pub info: Option<NodeInfo>,
    #[serde(default)]
    pub error: Option<ErrorObj>,
}

/// What `submit` sends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubmitSpec {
    pub job_key: String,
    pub sha: String,
    pub gate: String,
    pub hardware: String,
    pub max_run_s: Option<u32>,
    pub note: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct Submitted {
    pub node_id: String,
    pub job_id: String,
    pub existing: bool,
}

/// One line of an attached stream. Only the fields the driver acts on are
/// named; the rest ride along in `rest` for the log.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct StreamEvent {
    pub seq: u64,
    pub kind: String,
    #[serde(default)]
    pub phase: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub lines: Vec<String>,
    #[serde(default)]
    pub outcome: Option<String>,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub verdict: Option<serde_json::Value>,
    #[serde(default)]
    pub cached: Option<bool>,
}

/// How an attach ended.
#[derive(Clone, Debug, PartialEq)]
pub enum AttachEnd {
    /// Exit 0: the job passed. `exit_code` is the child's.
    Passed { exit_code: Option<i32> },
    /// Exit 5: the job ended without a pass; the `done` event says how.
    JobFailed {
        outcome: Option<String>,
        exit_code: Option<i32>,
        detail: String,
    },
    /// Exit 6.
    Cancelled,
    /// Exit 7: the stream could not be re-established; resume from here.
    StreamLost { last_seq: u64 },
    /// atlasctl refused, or could not reach or authenticate.
    Failed { exit: Exit, error: Option<ErrorObj> },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct FetchedFile {
    pub name: String,
    pub relative_path: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}
