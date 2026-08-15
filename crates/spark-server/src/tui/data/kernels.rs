// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-table model for Main ▸ Kernels: the LOADED target's embedded module
//! set joined with the runtime resolution audit — the same data
//! `render_kernel_table` prints, as rows a real `Table` widget can sort/filter.
//!
//! The module set has to come from `ptx_for_shape`, exactly as
//! `super::library` does it. `atlas_kernels::ptx_modules()` is emitted as a
//! plain alias of TARGET 0 in a multi-target build (`build_codegen.rs`), and
//! targets are sorted by directory name — so on every model except
//! `deepseek-v4-flash` it rendered another architecture's modules: phantom
//! rows, no row at all for the modules the live model actually uses, and every
//! PTX hash wrong. The table is the instrument this whole audit is read
//! through; while it named the wrong target, it was worse than absent.

use std::sync::Mutex;

use spark_runtime::kernel_audit::AuditRow;

/// One row of the kernel table.
#[derive(Clone, Debug)]
pub struct KernelRow {
    pub module: String,
    pub ptx_hash: String,
    /// None = embedded but never requested ("-"); Some(true) = used;
    /// Some(false) = lookup FAILED.
    pub resolution: Option<bool>,
}

/// A `(module, func)` lookup that failed — the "missing" list under the table.
#[derive(Clone, Debug)]
pub struct MissingKernel {
    pub module: String,
    pub func: String,
    /// `file:line` of the dispatch site, from the audit's `#[track_caller]`
    /// capture. Without it the operator has a name and nowhere to go.
    pub site: String,
}

impl MissingKernel {
    fn from_row(r: &AuditRow) -> Self {
        Self {
            module: r.module.clone(),
            func: r.func.clone(),
            site: format!("{}:{}", r.site.file(), r.site.line()),
        }
    }
}

#[derive(Default)]
pub struct KernelTableModel {
    pub rows: Vec<KernelRow>,
    /// ACTIONABLE failures: nothing declared these absent. This is the only
    /// list that may raise an alarm.
    pub missing_required: Vec<MissingKernel>,
    /// Failures the target's MODEL.toml `[expected_absent]` declares, each
    /// with a stated reason. Shown, never alarmed on — a warning that is
    /// almost always noise trains people to ignore the one time it is not.
    pub missing_expected: Vec<MissingKernel>,
}

/// `(model_type, hidden_size, mtp_layers)` of the model currently loaded.
///
/// Published by the serve path at the moment it resolves the target, so the
/// dashboard re-resolves the SAME `TargetPtxSet` the model was built against
/// rather than guessing from the served-model name (which is an HF id, not a
/// kernel-target directory name). `mtp_layers` is part of the key because it
/// is part of the selection: drop it and the table shows the sibling
/// target's modules for two models that share `(model_type, hidden_size)`.
static LOADED_SHAPE: Mutex<Option<(String, usize, usize)>> = Mutex::new(None);

/// Record which model config the live target was resolved from.
pub fn publish_loaded_shape(model_type: &str, hidden_size: usize, mtp_layers: usize) {
    if let Ok(mut g) = LOADED_SHAPE.lock() {
        *g = Some((model_type.to_string(), hidden_size, mtp_layers));
    }
}

fn loaded_shape() -> Option<(String, usize, usize)> {
    let guard = LOADED_SHAPE.lock().ok()?;
    guard.clone()
}

/// FNV-1a 12-hex content hash — matches `kernel_audit`'s `ptx_hash`.
fn ptx_hash(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

/// Build the table from the LOADED target's module set + the audit rows.
/// Cheap enough to refresh on demand (kernel resolution only happens at
/// startup, so callers refresh once after `ready`).
pub fn build() -> KernelTableModel {
    let audit = spark_runtime::kernel_audit::audit_rows();
    // No model loaded yet, or this build has no matching compiled target: an
    // EMPTY table is the honest answer. Falling back to some other target's
    // module list is the bug this function was rewritten to fix.
    let Some(ptx) = loaded_shape().and_then(|(model_type, hidden_size, mtp_layers)| {
        atlas_kernels::ptx_for_shape(atlas_kernels::ModelShape {
            model_type: &model_type,
            hidden_size,
            mtp_layers,
        })
    }) else {
        return KernelTableModel::default();
    };
    let mut rows: Vec<KernelRow> = ptx
        .modules
        .iter()
        .map(|(module, blob)| {
            let mut resolution = None;
            for r in &audit {
                if r.module == *module {
                    resolution = Some(resolution.unwrap_or(false) || r.loaded);
                }
            }
            KernelRow {
                module: (*module).to_string(),
                ptx_hash: ptx_hash(blob),
                resolution,
            }
        })
        .collect();
    rows.sort_by(|a, b| a.module.cmp(&b.module));
    // SSOT: the same classification the log table and the boot gate use. A
    // second copy of this rule here is how the TUI came to report 51 failures
    // where the log reported 4 actionable ones.
    let split = spark_runtime::kernel_audit::split_failures(&audit, ptx.expected_absent);
    KernelTableModel {
        rows,
        missing_required: split.required.iter().map(MissingKernel::from_row).collect(),
        missing_expected: split.expected.iter().map(MissingKernel::from_row).collect(),
    }
}
