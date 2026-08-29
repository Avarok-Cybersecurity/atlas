// SPDX-License-Identifier: AGPL-3.0-only

//! The committed shape of one gate run, and how it is written and read.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::gate_dir;
use crate::hardware::Hardware;
use crate::history::RunRecord;
use crate::result::{RunStatus, VerdictKind};

pub use super::record_path::{date_of, record_path, record_path_for, variant_slug};

/// One run record, as committed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateRecord {
    pub schema: u32,
    pub benchmark_id: String,
    pub benchmark_name: String,
    /// Commit the measured binary was built from. A record that cannot name
    /// its commit cannot be traced, so the writer refuses one without it.
    pub git_sha: String,
    /// The uncommitted invalidation-set files present when the run started —
    /// the ones that make `git_sha` above an incomplete description of the
    /// binary. Empty (and absent from the JSON) is the normal case.
    ///
    /// ★ Recorded, not just warned about, because the console warning is
    /// ephemeral and the record is what survives. A reader six weeks later
    /// asking "does this number belong to that commit?" has to be able to
    /// answer it from the file, without having watched the run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dirty_paths: Vec<String>,
    pub recorded_at: u64,
    pub target_model: String,
    /// Every parameter of the run, defaults included — the exact inputs of
    /// the command below.
    pub params: BTreeMap<String, String>,
    /// The exact CLI invocation, reconstructed from the recorded inputs, so
    /// the run can be reproduced without interpretation.
    pub command: Vec<String>,
    /// The recipe that served this run, when the gate provisioned its own
    /// server (`<family>/<stem>`). `None` means an endpoint the operator was
    /// already running.
    ///
    /// This is the honest half of `command` for a self-provisioned run: the
    /// URL such a run used names an ephemeral port that no longer exists, so
    /// what actually determined the config is the recipe, not the flags.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub served_by: Option<String>,
    /// Recipe keys the operator changed on the command line for this run.
    ///
    /// Empty (and absent from the JSON) means the recipe served exactly as
    /// pinned. Non-empty means `served_by` alone OVERSTATES the provenance —
    /// the numbers describe a config that exists in no file, and a reader who
    /// opened that recipe would be reading the wrong one. That is the same
    /// failure `served_by` was added to prevent, one level in.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_overrides: BTreeMap<String, String>,
    pub atlas_version: String,
    /// The box that served the model during the run.
    pub hardware: Hardware,
    /// What STATE that box was in, captured before and after the run, with the
    /// delta and both verdicts.
    ///
    /// [`Hardware`] above names the box; this says whether the box was in a
    /// condition to produce a number worth reading. They are different
    /// questions: on 2026-08-15 two boxes with byte-identical fingerprints
    /// (NVIDIA GB10, driver 580.126.09) returned 692 s and 1079 s on
    /// `agentic-webserver` for the SAME code, and a "+38% regression" was
    /// filed and retracted because the record could not tell them apart.
    ///
    /// Absent from the JSON for every record written before this existed, and
    /// for any run whose state could not be captured. Absent means UNMEASURED
    /// — it must never be read as "the box was fine".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_state: Option<crate::hardware::HardwareStateReport>,
    /// Raw headline numbers, keyed by stable metric name.
    pub metrics: BTreeMap<String, f64>,
    /// Content identity of the dataset the run scored against, from the
    /// terminal frame (e.g. `file-sha256:…;draw-sha256:…`). Additive and
    /// optional — schema stays 1, older records simply lack it. It exists
    /// because `metrics` is f64-only: an exact `samples`/`trajectories` pin
    /// catches a draw whose SIZE drifted, and nothing before this could catch
    /// a draw whose CONTENT did. BFCL computes exactly this digest during
    /// provisioning and drops it; the MLPerf agentic leg is the first to
    /// record it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_fingerprint: Option<String>,
    /// The run's terminal status. A `Failed` frame never passes the gate,
    /// whatever its numbers look like.
    pub frame_status: RunStatus,
    /// PASS / FAIL / info, and the reason the verdict carries.
    pub verdict: Option<String>,
    pub verdict_reason: String,
    /// One line a future reader scans before the numbers: what was measured,
    /// what it hit, and anything the verdict or log makes noteworthy.
    pub summary: String,
    /// The scheduler performance controls this run actually resolved.
    ///
    /// `--pull-request-gate` starts the server as a task INSIDE this process
    /// (`bench_selfstart::serve_for`), so the scheduler reads these from the
    /// inherited process environment. Nothing pinned them and nothing recorded
    /// them, which means two records could share a tree, a recipe and a full
    /// set of serve overrides and still have executed different admission
    /// behaviour — the confirmed provenance defect in atlas#812, and the one
    /// remaining candidate for the C=4 bimodality that the stored fields could
    /// not separate.
    ///
    /// Values are RESOLVED, not raw: an unset variable is recorded as the
    /// default the scheduler would apply, because "absent" and "set to the
    /// default" are the same run and must read the same in the record.
    ///
    /// Disclosure only — `check_record` does not demand a match. Demanding one
    /// would invalidate every record written before this field existed, for a
    /// value none of them could have carried.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub perf_env: BTreeMap<String, String>,
    /// What each kernel target compiled to when this was measured.
    ///
    /// Lets a later `kernels/`-only diff keep this record for the targets whose
    /// device code did not change — see [`super::closure`]. Empty (and absent
    /// from the JSON) is the pre-attestation case and excuses nothing, so
    /// records written before this existed behave exactly as they did.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub closure: super::closure::Attestation,
}

/// The scheduler performance controls a gate record discloses, each with the
/// default the scheduler applies when it is unset.
///
/// Kept beside the record rather than imported from the scheduler because
/// `atlas-plugin` does not depend on `spark-server`. That is a real duplication
/// and `perf_env_defaults_match_the_scheduler` pins it: if a default moves in
/// `scheduler::mod_helpers`, that test is what fails.
const PERF_CONTROLS: [(&str, &str); 3] = [
    ("ATLAS_PREFILL_CODISPATCH", "0"),
    ("ATLAS_PREFILL_CODISPATCH_WINDOW_MS", "100"),
    ("ATLAS_PREFILL_CODISPATCH_SETTLE_MS", "10"),
];

/// Resolve [`PERF_CONTROLS`] through `lookup`, substituting each default for an
/// unset or empty variable.
///
/// Pure over the lookup so it is testable without mutating the process
/// environment — `set_var` is unsafe and process-global, and a test that raced
/// another test's read would be exactly the kind of intermittent this file
/// exists to make impossible.
pub fn resolve_perf_env(lookup: impl Fn(&str) -> Option<String>) -> BTreeMap<String, String> {
    PERF_CONTROLS
        .iter()
        .map(|(key, default)| {
            let value = lookup(key).filter(|v| !v.trim().is_empty());
            (
                (*key).to_string(),
                value.unwrap_or_else(|| (*default).to_string()),
            )
        })
        .collect()
}

/// Comparison against one metric's threshold. `min` fails below (scores),
/// `max` fails above (latencies, wall time) — the two are mutually exclusive
/// per metric.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Bound {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Points of slack the gate allows beyond the bound — measurement noise,
    /// e.g. MTP's sub-noise BFCL dips. Default 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise: Option<f64>,
}

/// One (hardware, model) pair's thresholds, and the recipe that produces them.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelBaseline {
    /// The recipe that serves this model, as `<family>/<stem>` — e.g.
    /// `qwen3.6/qwen3.6-27b-nvfp4-unsloth`. This is the ONLY machine-readable
    /// binding from a benchmark to its serve config; without it a gate can be
    /// run against hand-typed flags that differ from the ones the thresholds
    /// were measured under, which is the failure this whole file exists to
    /// stop. `None` means the gate cannot self-provision and must be told a
    /// live endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
    /// Human name for this variant, for the TUI's variant list. Empty falls
    /// back to the checkpoint id.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    /// Why these are the thresholds — the source run the numbers come from.
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub metrics: BTreeMap<String, Bound>,
    /// Recipe keys self-start applies for this gate. Empty (and omitted) means
    /// the recipe serves exactly as pinned. See `bench::BenchEntry` for the
    /// full contract (values are strings matching `--serve-override KEY=VALUE`;
    /// `port` is refused at parse).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_overrides: BTreeMap<String, String>,
    /// Benchmark PARAMETER keys the gate pins for this entry — the request
    /// side of what `serve_overrides` is for the serve side. Empty (and
    /// omitted) means the benchmark's schema defaults ARE the gate's shape.
    /// Non-empty means the thresholds were calibrated on a different
    /// instrument than the schema default (the concurrency gate's ladder is
    /// C=1/4/8/16 at isl 512 / osl 320, where the schema sweeps 1..32 at
    /// osl 128), and a gate run must reproduce that instrument or its
    /// numbers are comparable to nothing. Values are strings routed through
    /// each parameter's own `ParamKind::parse`, exactly like a typed
    /// `--param`; an explicit `--param` still wins. See `bench::BenchEntry`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub param_overrides: BTreeMap<String, String>,
}

/// Baseline pins first; the operator's `--serve-override` wins on a clash.
///
/// Declaration under operator: the baseline states what the gate NEEDS to be
/// meaningful, but an operator typing a key at the command line is stating
/// intent for this run, and both end up disclosed in the record either way.
pub fn merge_serve_overrides(
    baseline: BTreeMap<String, String>,
    requested: BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut out = baseline;
    out.extend(requested);
    out
}

/// Every model measured on one box class, and which one to serve by default.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HardwareBaseline {
    /// The model to use when the caller does not name one.
    ///
    /// Explicit rather than "the only entry" or "the first key": a second model
    /// added later must not silently move the gate's subject.
    pub default: String,
    #[serde(default)]
    pub models: BTreeMap<String, ModelBaseline>,
}

/// The thresholds a benchmark's gate records must meet.
///
/// Assembled at read time from every `kernels/<hw>/<model>/BENCH.toml`; see
/// [`super::bench`] for why they live beside the model rather than in one file
/// per gate. `.benchmarks/<id>/` still holds the RECORDS.
///
/// Keyed **hardware → model → thresholds** because both axes genuinely move the
/// numbers. TTFT is box-local by construction — a ceiling measured on one box
/// says nothing about another — and a BFCL score is checkpoint-specific, so a
/// single flat threshold set could only ever be right for one combination and
/// silently wrong for the rest.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GateBaseline {
    /// Schema version. 2 introduced the hardware/model nesting.
    #[serde(default)]
    pub schema: u32,
    pub hardware: BTreeMap<String, HardwareBaseline>,
}

impl GateBaseline {
    /// Resolve one (hardware, model) entry. `model: None` takes the hardware's
    /// declared default.
    ///
    /// Every failure names both what was asked for and what exists — an
    /// unresolved baseline must never read as "nothing to check".
    pub fn resolve(&self, hardware: &str, model: Option<&str>) -> Result<(String, &ModelBaseline)> {
        let hw = self.hardware.get(hardware).ok_or_else(|| {
            anyhow::anyhow!(
                "no baseline for hardware {hardware:?}; this benchmark has entries for [{}]",
                self.hardware.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;
        let want = model.unwrap_or(&hw.default);
        let entry = hw.models.get(want).ok_or_else(|| {
            anyhow::anyhow!(
                "no baseline for model {want:?} on {hardware:?}; it has [{}]",
                hw.models.keys().cloned().collect::<Vec<_>>().join(", ")
            )
        })?;
        Ok((want.to_string(), entry))
    }
}

/// Write one gate record. Returns the path; the parent directory is created,
/// but never committed on the writer's behalf — that stays the caller's
/// explicit act.
pub fn write_record(root: &Path, record: &GateRecord) -> Result<PathBuf> {
    let path = record_path_for(root, record);
    std::fs::create_dir_all(path.parent().expect("record path has a parent")).with_context(
        || {
            format!(
                "creating {}",
                gate_dir(root, &record.benchmark_id).display()
            )
        },
    )?;
    let json = serde_json::to_string_pretty(record).context("serializing the gate record")?;
    std::fs::write(&path, json + "\n").with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Read one committed record.
pub fn read_record(path: &Path) -> Result<GateRecord> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Read one committed baseline.
pub fn read_baseline(root: &Path, benchmark_id: &str) -> Result<GateBaseline> {
    // Assembled from every model's `kernels/<hw>/<model>/BENCH.toml` rather
    // than read from one file. `.benchmarks/<id>/` still holds the RECORDS —
    // only the thresholds moved, to sit beside the model they describe.
    super::bench::baseline_for(root, benchmark_id)
}

impl GateRecord {
    /// Build a gate record from what a finished run leaves behind. The
    /// hardware fingerprint comes from the serving endpoint's `/hardware` —
    /// the box that did the inference, not the box running this CLI.
    /// `served_by` names the recipe when the gate provisioned its own server.
    /// It changes the reconstructed command, because the two modes are
    /// reproduced differently: a self-provisioned run is replayed by asking for
    /// the same benchmark again (the recipe re-derives the endpoint), whereas
    /// naming its `--url` would point at an ephemeral port that no longer
    /// exists and a `--model` nobody typed.
    ///
    /// `dirty_paths` is the invalidation-set dirt that was in the tree when the
    /// run started (see [`super::dirty_perf_paths`]); it is a parameter rather
    /// than a setter so that a caller cannot produce a record that quietly
    /// omits it.
    pub fn from_run(
        record: &RunRecord,
        hardware: Hardware,
        git_sha: String,
        dirty_paths: Vec<String>,
        served_by: Option<String>,
        serve_overrides: BTreeMap<String, String>,
    ) -> Result<Self> {
        if git_sha.trim().is_empty() {
            bail!("a gate record needs the commit sha it was measured from");
        }
        let frame = &record.frame;
        if frame.status == RunStatus::Running {
            bail!("the run never reached a terminal frame — nothing to gate");
        }
        let mut params = Vec::new();
        if served_by.is_none() {
            if !record.target_url.is_empty() {
                params.push(("--url".to_string(), record.target_url.clone()));
            }
            if !record.target_model.is_empty() {
                params.push(("--model".to_string(), record.target_model.clone()));
            }
        }
        for (k, v) in &record.params {
            params.push(("--param".to_string(), format!("{k}={v}")));
        }
        // Reconstructed alongside the rest so `command` stays REPLAYABLE, not
        // merely descriptive: a self-provisioned run's config is the recipe
        // plus these, and a command that omitted them would rerun a different
        // server and quietly disagree with the record it came from.
        for (k, v) in &serve_overrides {
            params.push(("--serve-override".to_string(), format!("{k}={v}")));
        }
        if record.benchmark_id == "agentic-webserver" {
            params.push(("--yes".to_string(), String::new()));
        }
        let mut command: Vec<String> = vec![
            "spark".into(),
            "benchmark".into(),
            "run".into(),
            record.benchmark_id.clone(),
        ];
        for (flag, value) in &params {
            command.push(flag.clone());
            if !value.is_empty() {
                command.push(value.clone());
            }
        }
        command.push("--pull-request-gate".into());

        let verdict = frame.verdict.as_ref().map(|v| match v.kind {
            VerdictKind::Pass => "PASS".to_string(),
            VerdictKind::Fail => "FAIL".to_string(),
            VerdictKind::Info => "info".to_string(),
        });
        let verdict_reason = frame
            .verdict
            .as_ref()
            .map(|v| v.reason.clone())
            .unwrap_or_default();
        Ok(Self {
            schema: 1,
            // Attached afterwards by `with_closure`, unlike `dirty_paths`.
            //
            // The asymmetry is deliberate and rests on which direction an
            // omission fails in. A record missing its dirt OVERSTATES its
            // provenance — it claims to describe a commit it does not — so the
            // constructor refuses to build one. A record missing its
            // attestation merely forfeits the savings: it excuses no future
            // diff and behaves exactly like every record written before this
            // existed. Requiring it here would mean threading a repo root
            // through every caller to buy nothing.
            closure: Default::default(),
            benchmark_id: record.benchmark_id.clone(),
            benchmark_name: record.benchmark_name.clone(),
            git_sha,
            dirty_paths,
            recorded_at: record.recorded_at,
            target_model: record.target_model.clone(),
            params: record.params.clone(),
            command,
            served_by,
            serve_overrides,
            atlas_version: record.atlas_version.clone(),
            hardware,
            // Carried through from the terminal frame the executor stamped it
            // on, rather than probed here: this record is written minutes to
            // hours after the run, and a state captured now would describe an
            // idle box instead of the one that did the work.
            hardware_state: frame.hardware_state.clone(),
            metrics: frame.metrics.clone(),
            dataset_fingerprint: frame.dataset_fingerprint.clone(),
            frame_status: frame.status,
            verdict,
            verdict_reason,
            summary: summarize(record),
            // Read here rather than at run start because the gate serves in
            // THIS process: the environment the scheduler read is still the
            // environment this call sees, and there is no second process whose
            // state could have diverged in between.
            perf_env: resolve_perf_env(|k| std::env::var(k).ok()),
        })
    }

    /// Attach what each kernel target in the measuring BINARY compiled from.
    ///
    /// `baked` is `atlas_kernels::TARGET_CLOSURES` — computed by the build
    /// script at the moment the kernels were compiled. It deliberately does not
    /// recompute from the working tree: doing so would attest to sources that
    /// may never have been built, which is the staleness this exists to catch.
    ///
    /// Unparseable or empty input attaches nothing, which excuses no future
    /// diff. That is the correct failure: it costs re-runs, not soundness.
    #[must_use]
    pub fn with_closure(mut self, baked: &str) -> Self {
        self.closure = serde_json::from_str(baked).unwrap_or_default();
        self
    }

    /// True when the run's verdict is a PASS. Anything else — FAIL, info, or
    /// no verdict at all — has not proven its bar.
    pub fn verdict_passes(&self) -> bool {
        self.verdict.as_deref() == Some("PASS")
    }

    /// True when the run's own frame says it never completed.
    pub fn frame_status_failed(&self) -> bool {
        self.frame_status == RunStatus::Failed
    }
}

/// The one line a future reader sees first. States the headline numbers and,
/// when the frame logged warnings, the first one — those are the observations
/// worth carrying into the next run's context.
fn summarize(record: &RunRecord) -> String {
    let frame = &record.frame;
    let numbers: Vec<String> = frame
        .metrics
        .iter()
        .map(|(k, v)| format!("{k}={v:.2}"))
        .collect();
    let numbers = if numbers.is_empty() {
        "no metrics".to_string()
    } else {
        numbers.join(", ")
    };
    let warning = frame
        .log
        .iter()
        .find(|l| {
            matches!(
                l.level,
                crate::result::LogLevel::Warn | crate::result::LogLevel::Error
            )
        })
        .map(|l| format!(" · warning: {}", l.text));
    let verdict = frame
        .verdict
        .as_ref()
        .map(|v| format!("{:?}: {}", v.kind, v.reason))
        .unwrap_or_else(|| "no verdict".into());
    format!(
        "{} · {} · {}{}",
        record.target_model,
        numbers,
        verdict,
        warning.unwrap_or_default()
    )
}

/// Unix seconds now.
pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}
