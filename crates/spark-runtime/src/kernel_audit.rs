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
use std::sync::Mutex;

/// (module, func, loaded). Appended on every `kernel()` lookup.
static AUDIT: Mutex<Vec<(String, String, bool)>> = Mutex::new(Vec::new());

/// Record one kernel lookup. Cheap; called from `GpuBackend::kernel`.
pub fn record(module: &str, func: &str, loaded: bool) {
    if let Ok(mut v) = AUDIT.lock() {
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

/// Render the embedded kernel set (`embedded` = the binary's `ptx_modules()`,
/// passed in since spark-runtime doesn't depend on atlas-kernels) plus the
/// runtime resolution overlay. `set_hash` is `atlas_kernels::KERNEL_SET_HASH`.
pub fn render_kernel_table(embedded: &[(&str, &str)], set_hash: &str) -> String {
    // Dedup resolution audit: (module, func) → loaded (true if ever true).
    let mut resolved: BTreeMap<(String, String), bool> = BTreeMap::new();
    if let Ok(v) = AUDIT.lock() {
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
        "│ {:<34} {:<14} {}\n",
        "MODULE (operation)", "PTX-HASH", "RESOLUTION"
    ));
    out.push_str(&format!("│ {}\n", "─".repeat(74)));
    let mut sorted: Vec<&(&str, &str)> = embedded.iter().collect();
    sorted.sort_by_key(|(m, _)| *m);
    for (m, ptx) in sorted {
        let h = ptx_hash(ptx.as_bytes());
        let res = match mod_resolved.get(m) {
            Some((_req, true)) => "used",
            Some((_req, false)) => "** lookup FAILED **",
            None => "-", // embedded but not requested by this model's dispatch
        };
        out.push_str(&format!("│ {m:<34} {h:<14} {res}\n"));
    }
    out.push_str("└─");

    // Explicit MISSING list: (module, func) requested but never resolved →
    // silent slower-fallback dispatch. The actionable debug signal.
    let missing: Vec<&(String, String)> =
        resolved.iter().filter(|(_, ok)| !**ok).map(|(k, _)| k).collect();
    if !missing.is_empty() {
        out.push_str(&format!(
            "\n⚠ {} kernel lookup(s) MISSING — slower fallback dispatch in use:\n",
            missing.len()
        ));
        for (m, f) in &missing {
            out.push_str(&format!("    - {m}::{f}\n"));
        }
    }
    out
}
