// SPDX-License-Identifier: AGPL-3.0-only

//! Startup kernel-resolution audit + embedded-kernel-set table.
//!
//! Two halves, both printed once at model-load time:
//!  1. The EMBEDDED kernel set — every `(module, ptx)` compiled into this
//!     binary, with a per-kernel PTX content hash and the overall kernel-set
//!     hash. The count here is ground truth (e.g. 98 vs 99 modules), and the
//!     hashes pin exactly which kernel binary is loaded — so a stale/dropped
//!     kernel from a build-codegen regression is visible at a glance.
//!  2. The RESOLUTION audit — every `GpuBackend::kernel(module, func)` lookup
//!     and whether it resolved. A MISSING optional kernel (`try_kernel` →
//!     handle 0) silently falls back to a slower dispatch path with no error;
//!     this surfaces it (see the 2026-06-04 pipelined-GEMM regression where
//!     `w8a16_gemm_pipelined` resolved to 0 and QKVZ fell back to the ~4.6×
//!     slower `w8a16_gemm`).

use std::collections::BTreeMap;

// The audit vector is a field of the single run mailbox,
// `crate::run_metrics::RunMetrics`. It is per-model in the sharpest way —
// it lists which of THIS model's registry modules resolved — so without
// the run-start clear a swap would leave the dashboard's kernel table
// showing both models' modules with no way to tell them apart.

/// Record one kernel lookup. Cheap; called from `GpuBackend::kernel`.
pub fn record(module: &str, func: &str, loaded: bool) {
    if let Ok(mut v) = crate::run_metrics::metrics().kernel_audit.lock() {
        v.push((module.to_string(), func.to_string(), loaded));
    }
}

/// FNV-1a 64-bit content fingerprint → 12 hex chars (matches build.rs).
fn ptx_hash(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

/// Structured resolution rows for observers (e.g. the TUI kernel table):
/// deduped `(module, func, loaded)`, sorted. `loaded` is true if ANY lookup
/// of that (module, func) resolved.
pub fn audit_rows() -> Vec<(String, String, bool)> {
    let mut resolved: BTreeMap<(String, String), bool> = BTreeMap::new();
    if let Ok(v) = crate::run_metrics::metrics().kernel_audit.lock() {
        for (m, f, ok) in v.iter() {
            let e = resolved.entry((m.clone(), f.clone())).or_insert(false);
            *e = *e || *ok;
        }
    }
    resolved
        .into_iter()
        .map(|((m, f), ok)| (m, f, ok))
        .collect()
}

/// A `(module, func)` pair borrowed from the resolution audit.
type KernelRef<'a> = &'a (String, String);

/// `(module, func)` lookups that FAILED and are kernels this target dropped by
/// shadowing `common/` — i.e. the kernel exists upstream, this model's fork of
/// the file omitted it, and the model's own dispatch then asked for it.
///
/// That conjunction is the actionable one: the kernel would have been there but
/// for the fork, and something in this model actively wanted it. A failed
/// lookup NOT in this set is a kernel that was never compiled for this target
/// at all (typically another architecture's — MLA, hyper-connection, CSA);
/// those are expected and are not returned here.
pub fn fatal_missing(shadowed_dropped: &[(&str, &str)]) -> Vec<(String, String)> {
    // SSOT: the same deduped resolution rows the TUI kernel table renders,
    // read off the run mailbox rather than a second copy of the fold.
    audit_rows()
        .into_iter()
        .filter(|(_, _, ok)| !*ok)
        .map(|(m, f, _)| (m, f))
        .filter(|(m, f)| {
            shadowed_dropped
                .iter()
                .any(|(sm, sf)| *sm == m.as_str() && *sf == f.as_str())
        })
        .collect()
}

/// Render the embedded kernel set (`embedded` = the binary's `ptx_modules()`,
/// passed in since spark-runtime doesn't depend on atlas-kernels) plus the
/// runtime resolution overlay. `set_hash` is `atlas_kernels::KERNEL_SET_HASH`.
/// `shadowed_dropped` is `TargetPtxSet::shadowed_dropped`, which drives the
/// SHADOWED column and splits the failed lookups into required-vs-expected.
pub fn render_kernel_table(
    embedded: &[(&str, &[u8])],
    set_hash: &str,
    shadowed_dropped: &[(&str, &str)],
) -> String {
    // Dedup resolution audit: (module, func) → loaded (true if ever true).
    let mut resolved: BTreeMap<(String, String), bool> = BTreeMap::new();
    if let Ok(v) = crate::run_metrics::metrics().kernel_audit.lock() {
        for (m, f, ok) in v.iter() {
            let e = resolved.entry((m.clone(), f.clone())).or_insert(false);
            *e = *e || *ok;
        }
    }
    // Per-module resolution rollup: any-loaded / any-requested.
    let mut mod_resolved: BTreeMap<&str, (bool, bool)> = BTreeMap::new(); // (requested, loaded)
    for ((m, _f), ok) in &resolved {
        let e = mod_resolved.entry(m.as_str()).or_insert((false, false));
        e.0 = true;
        e.1 = e.1 || *ok;
    }

    let mut out = String::new();
    out.push_str(&format!(
        "\n┌─ Kernel load audit ─ {} kernels embedded · set-hash {} ─\n",
        embedded.len(),
        set_hash
    ));
    out.push_str(&format!(
        "│ {:<34} {:<14} {:<20} {}\n",
        "MODULE (operation)", "PTX-HASH", "RESOLUTION", "SHADOWED"
    ));
    out.push_str(&format!("│ {}\n", "─".repeat(84)));
    let mut sorted: Vec<&(&str, &[u8])> = embedded.iter().collect();
    sorted.sort_by_key(|(m, _)| *m);
    for (m, blob) in sorted {
        // Blob is the raw kernel bytes (PTX text or AMD/Metal binary);
        // FNV-1a over the bytes directly — matches build.rs's set hash.
        let h = ptx_hash(blob);
        let res = match mod_resolved.get(m) {
            Some((_req, true)) => "used",
            Some((_req, false)) => "** lookup FAILED **",
            None => "-", // embedded but not requested by this model's dispatch
        };
        // Y when this model's fork of the file dropped one or more kernels that
        // `common/` defines — the module compiled, but not everything in it.
        let n_dropped = shadowed_dropped.iter().filter(|(sm, _)| sm == m).count();
        let shadow = if n_dropped > 0 {
            format!("Y ({n_dropped} dropped)")
        } else {
            "N".to_string()
        };
        out.push_str(&format!("│ {m:<34} {h:<14} {res:<20} {shadow}\n"));
    }
    out.push_str("└─");

    // Split the failed lookups by CAUSE. Reporting them as one list is what let
    // the 27B ship with concurrent decode silently disabled: the four dropped
    // GDN kernels sat among ~26 entries for architectures the model does not
    // have (MLA, hyper-connection, CSA), so the whole warning read as benign
    // and everyone learned to skip it. A warning that is almost always noise
    // trains people to ignore the one time it is not.
    let missing: Vec<&(String, String)> = resolved
        .iter()
        .filter(|(_, ok)| !**ok)
        .map(|(k, _)| k)
        .collect();
    let is_dropped =
        |m: &str, f: &str| shadowed_dropped.iter().any(|(sm, sf)| *sm == m && *sf == f);
    let (dropped, absent): (Vec<KernelRef>, Vec<KernelRef>) = missing
        .into_iter()
        .partition(|(m, f)| is_dropped(m.as_str(), f.as_str()));

    if !dropped.is_empty() {
        out.push_str(&format!(
            "\n\u{2718} {} REQUIRED kernel(s) MISSING — this model's dispatch asked for them and \
             `common/` defines them, but this target's kernel file shadows `common/` WITHOUT \
             them, so they were never compiled:\n",
            dropped.len()
        ));
        for (m, f) in &dropped {
            out.push_str(&format!("    - {m}::{f}\n"));
        }
        out.push_str(
            "  Port them into this model's kernel file (exact piecewise copy from common/).\n",
        );
    }
    if !absent.is_empty() {
        // Informational only: never compiled for this target, and the model
        // asking is just an unconditional `try_kernel` probe.
        out.push_str(&format!(
            "\n\u{2139} {} optional kernel(s) not built for this target (other architectures — \
             expected, no action):\n    {}\n",
            absent.len(),
            absent
                .iter()
                .map(|(m, f)| format!("{m}::{f}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    out
}
