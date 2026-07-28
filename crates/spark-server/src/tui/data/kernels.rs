// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel-table model for Main ▸ Kernels: the embedded module set joined with
//! the runtime resolution audit — the same data `render_kernel_table` prints,
//! as rows a real `Table` widget can sort/filter.

/// One row of the kernel table.
#[derive(Clone, Debug)]
pub struct KernelRow {
    pub module: String,
    pub ptx_hash: String,
    /// None = embedded but never requested ("-"); Some(true) = used;
    /// Some(false) = lookup FAILED.
    pub resolution: Option<bool>,
}

/// A (module, func) lookup that failed — the "missing" list under the table.
#[derive(Clone, Debug)]
pub struct MissingKernel {
    pub module: String,
    pub func: String,
}

pub struct KernelTableModel {
    pub rows: Vec<KernelRow>,
    pub missing: Vec<MissingKernel>,
}

/// FNV-1a 12-hex content hash — matches `kernel_audit::ptx_hash`.
fn ptx_hash(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:012x}", h & 0xffff_ffff_ffff)
}

/// Build the table from the binary's embedded module set + the audit rows.
/// Cheap enough to refresh on demand (kernel resolution only happens at
/// startup, so callers refresh once after `ready`).
pub fn build() -> KernelTableModel {
    let audit = spark_runtime::kernel_audit::audit_rows();
    let mut rows: Vec<KernelRow> = atlas_kernels::ptx_modules()
        .iter()
        .map(|(module, blob)| {
            let mut resolution = None;
            for (m, _f, ok) in &audit {
                if m == module {
                    resolution = Some(resolution.unwrap_or(false) || *ok);
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
    let missing = audit
        .into_iter()
        .filter(|(_, _, ok)| !ok)
        .map(|(module, func, _)| MissingKernel { module, func })
        .collect();
    KernelTableModel { rows, missing }
}
