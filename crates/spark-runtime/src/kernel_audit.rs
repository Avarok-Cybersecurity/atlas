// SPDX-License-Identifier: AGPL-3.0-only

//! Startup kernel-resolution audit + embedded-kernel-set table.
//!
//! Two halves, both printed once at model-load time:
//!  1. The EMBEDDED kernel set — every `(module, ptx)` compiled into this
//!     binary, with a per-kernel PTX content hash and the overall kernel-set
//!     hash. The count here is ground truth (e.g. 98 vs 99 modules), and the
//!     hashes pin exactly which kernel binary is loaded — so a stale/dropped
//!     kernel from a build-codegen regression is visible at a glance.
//!  2. The RESOLUTION audit — every `GpuBackend::kernel(module, func)` lookup,
//!     whether it resolved, and WHERE it was issued from. A MISSING optional
//!     kernel (`try_kernel` → handle 0) silently falls back to a slower
//!     dispatch path with no error; this surfaces it (see the 2026-06-04
//!     pipelined-GEMM regression where `w8a16_gemm_pipelined` resolved to 0
//!     and QKVZ fell back to the ~4.6× slower `w8a16_gemm`).
//!
//! Every kernel lookup in Atlas is EAGER: each one sits in a constructor on the
//! `serve_phases::build_model` path, so by the time the model is built the
//! audit holds the COMPLETE `(module, func)` set this model asks for. That is
//! what makes [`seal`] meaningful — after it, a lookup is by definition a late
//! one, and a late MISS is a silent slow path nobody would ever see. Sealing
//! turns the invariant from a belief into an assertion.

mod report;

pub use report::{render_kernel_table, unresolved_report};

use std::collections::BTreeMap;
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// The audit vector is a field of the single run mailbox,
// `crate::run_metrics::RunMetrics`. It is per-model in the sharpest way —
// it lists which of THIS model's registry modules resolved — so without
// the run-start clear a swap would leave the dashboard's kernel table
// showing both models' modules with no way to tell them apart.

/// True once the boot gate has run and passed. See [`seal`].
static SEALED: AtomicBool = AtomicBool::new(false);
/// `--dangerously-allow-unresolved-kernel-lookups`, as handed to [`seal`].
static ALLOW_UNRESOLVED: AtomicBool = AtomicBool::new(false);
/// Unresolved lookups for the live model: the gate's count, plus any late
/// miss recorded after the seal. Exported as `atlas_kernel_lookups_unresolved`.
static UNRESOLVED: AtomicU64 = AtomicU64::new(0);
/// One-shot latch so a late miss inside a hot loop warns once, not per token.
static LATE_WARNED: AtomicBool = AtomicBool::new(false);

/// One deduped kernel-resolution row.
///
/// `site` is the DISPATCH SITE — `file:line` of the `.kernel(…)` /
/// `try_kernel(…)` call, captured through `#[track_caller]`. A bare
/// `module::func` list is not actionable: the same module name is looked up
/// from a dozen constructors, and the fix is always "go to that line".
#[derive(Clone, Debug)]
pub struct AuditRow {
    pub module: String,
    pub func: String,
    /// True if ANY lookup of this `(module, func)` resolved.
    pub loaded: bool,
    /// Dispatch site of the first lookup of this pair.
    pub site: &'static Location<'static>,
}

impl AuditRow {
    /// `module::func` — the name the log table and the TUI both print.
    pub fn name(&self) -> String {
        format!("{}::{}", self.module, self.func)
    }
}

/// Record one kernel lookup. Cheap; called from `GpuBackend::kernel`.
///
/// `site` is the caller's `Location`, which the backend obtains from its own
/// `#[track_caller]` frame — this function cannot take it implicitly, because
/// its own caller is the backend, not the dispatch site.
pub fn record(module: &str, func: &str, loaded: bool, site: &'static Location<'static>) {
    if !loaded && SEALED.load(Ordering::Acquire) {
        late_miss(module, func, site);
    }
    if let Ok(mut v) = crate::run_metrics::metrics().kernel_audit.lock() {
        v.push((module.to_string(), func.to_string(), loaded, site));
    }
}

/// A kernel lookup that FAILED after the boot gate had already passed.
///
/// This can only happen if a lookup is not eager — i.e. some dispatch path
/// resolves a kernel lazily, on the first request that needs it. That is
/// precisely the case the boot gate cannot see, so it must be loud here or it
/// is invisible forever: the caller takes a silent slow path and the only
/// symptom is a throughput number nobody has a baseline for.
fn late_miss(module: &str, func: &str, site: &'static Location<'static>) {
    UNRESOLVED.fetch_add(1, Ordering::Relaxed);
    if ALLOW_UNRESOLVED.load(Ordering::Relaxed) {
        if !LATE_WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "kernel lookup {module}::{func} at {}:{} failed AFTER the boot audit sealed. \
                 This lookup is not eager, so the boot gate could not see it. Continuing \
                 because --dangerously-allow-unresolved-kernel-lookups was passed. \
                 Performance may be seriously degraded. We recommend you open a GitHub issue \
                 and/or open a PR to solve this issue. \
                 (atlas_kernel_lookups_unresolved counts every occurrence.)",
                site.file(),
                site.line(),
            );
        }
        return;
    }
    // Abort, not panic: a panic here unwinds one scheduler/request thread and
    // leaves a half-serving process behind, which is the same silent
    // degradation this gate exists to stop. FAIL LOUDLY.
    tracing::error!(
        "kernel lookup {module}::{func} at {}:{} failed AFTER the boot audit sealed — a \
         non-eager lookup that the boot gate could not see. The dispatch that asked for it is \
         now on a silent fallback path. Aborting. Pass \
         --dangerously-allow-unresolved-kernel-lookups to downgrade this to a warning.",
        site.file(),
        site.line(),
    );
    std::process::abort();
}

/// Close the audit for this run: the boot gate has run and every eager lookup
/// has been made. `unresolved` is the gate's own count (rows that failed and
/// were not classified expected-absent); `allow` is
/// `--dangerously-allow-unresolved-kernel-lookups`.
pub fn seal(unresolved: u64, allow: bool) {
    UNRESOLVED.store(unresolved, Ordering::Relaxed);
    ALLOW_UNRESOLVED.store(allow, Ordering::Relaxed);
    LATE_WARNED.store(false, Ordering::Relaxed);
    SEALED.store(true, Ordering::Release);
}

/// Re-open the audit for a new model load. Called from
/// [`crate::run_metrics::reset_for_new_run`], which is where a run begins —
/// the next model runs its own eager lookups and gets its own gate.
pub fn unseal() {
    SEALED.store(false, Ordering::Release);
    UNRESOLVED.store(0, Ordering::Relaxed);
    LATE_WARNED.store(false, Ordering::Relaxed);
}

/// Unresolved kernel lookups for the live model. Exported on `/metrics` as
/// `atlas_kernel_lookups_unresolved` so a gate can assert `== 0` without
/// parsing logs.
pub fn unresolved_lookups() -> u64 {
    UNRESOLVED.load(Ordering::Relaxed)
}

/// Structured resolution rows for observers (log table, TUI kernel table, the
/// boot gate): deduped `(module, func)`, sorted, `loaded` true if ANY lookup of
/// that pair resolved.
pub fn audit_rows() -> Vec<AuditRow> {
    let mut resolved: BTreeMap<(String, String), (bool, &'static Location<'static>)> =
        BTreeMap::new();
    if let Ok(v) = crate::run_metrics::metrics().kernel_audit.lock() {
        for (m, f, ok, site) in v.iter() {
            let e = resolved
                .entry((m.clone(), f.clone()))
                .or_insert((false, *site));
            e.0 = e.0 || *ok;
        }
    }
    resolved
        .into_iter()
        .map(|((module, func), (loaded, site))| AuditRow {
            module,
            func,
            loaded,
            site,
        })
        .collect()
}

/// The failed lookups, split by whether the operator must act.
///
/// SSOT: the log table, the TUI kernel table and the boot gate all read this
/// one function. Reporting the two classes as a single list is what let the
/// 27B ship with concurrent decode silently disabled — the four dropped GDN
/// kernels sat among ~26 entries for architectures the model does not have, so
/// the whole warning read as benign and everyone learned to skip it. A warning
/// that is almost always noise trains people to ignore the one time it is not.
#[derive(Clone, Debug, Default)]
pub struct FailureSplit {
    /// Actionable. Nothing declared these absent, so either the model's
    /// dispatch should not have asked (gate it on config, see
    /// `qwen3_attention::init`) or the kernel should have been compiled.
    pub required: Vec<AuditRow>,
    /// Declared in this target's MODEL.toml `[expected_absent]`, each with a
    /// stated reason. Informational; never fatal.
    pub expected: Vec<AuditRow>,
}

/// Split [`audit_rows`]'s failures against a target's `[expected_absent]`
/// declaration (`TargetPtxSet::expected_absent`).
pub fn classify_failures(expected_absent: &[(&str, &str)]) -> FailureSplit {
    split_failures(&audit_rows(), expected_absent)
}

/// [`classify_failures`] over rows the caller already has.
pub fn split_failures(rows: &[AuditRow], expected_absent: &[(&str, &str)]) -> FailureSplit {
    let (expected, required): (Vec<AuditRow>, Vec<AuditRow>) =
        rows.iter().filter(|r| !r.loaded).cloned().partition(|r| {
            expected_absent
                .iter()
                .any(|(em, ef)| *em == r.module.as_str() && *ef == r.func.as_str())
        });
    FailureSplit { required, expected }
}
