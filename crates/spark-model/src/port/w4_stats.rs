// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the process-wide record of what the 4-bit (`W4Linear`) routes installed at load, and its
//! `/metrics` text: `atlas_w4_weight_bytes{role,format}`, `atlas_w4_selfcheck_failures_total` and
//! `atlas_affine4_prefill_int8_fallbacks_total` (lane L23b's namespace, master plan 4.2).
//!
//! The loader (`weight_loader/qwen35_dense/w4_install.rs`) [`publish`]es one entry per layer type at the end
//! of every load, replacing the previous load's entries (a model swap reloads). With every 4-bit row off,
//! nothing is published and every series renders as zero, on every target.
//!
//! spark-model cannot reach spark-server's `api::port_metrics` helpers, so [`render_prometheus`] writes the
//! same `# HELP` / `# TYPE` lines itself; spark-server calls it from `port_metrics::render`.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::port::w4::W4Format;

/// The layer type a 4-bit weight serves (the `role` label).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum W4Role {
    /// Dense-FFN gate, up and down.
    Ffn,
    /// GDN `in_proj_qkvz` and `out_proj`.
    Gdn,
    /// Full-attention q, k, v and o.
    Attn,
    /// The LM head (a packed GGUF head, lane L24).
    LmHead,
}

impl W4Role {
    /// Every role, in label order.
    pub const ALL: [W4Role; 4] = [W4Role::Ffn, W4Role::Gdn, W4Role::Attn, W4Role::LmHead];

    /// The `role` label and the word the boot lines use.
    pub fn label(self) -> &'static str {
        match self {
            Self::Ffn => "ffn",
            Self::Gdn => "gdn",
            Self::Attn => "attn",
            Self::LmHead => "lm_head",
        }
    }
}

/// What one layer type ended up with after a load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoleRecord {
    pub role: W4Role,
    /// `None` when nothing of this role is 4-bit.
    pub format: Option<W4Format>,
    /// Layers that took the route.
    pub layers: u32,
    /// Resident device bytes of the installed 4-bit weights.
    pub bytes: u64,
    /// The self-check failed and the role's batched rows run one row at a time.
    pub per_row: bool,
}

impl RoleRecord {
    /// An empty record (nothing installed).
    pub const fn none(role: W4Role) -> Self {
        Self {
            role,
            format: None,
            layers: 0,
            bytes: 0,
            per_row: false,
        }
    }
}

fn records() -> &'static Mutex<Vec<RoleRecord>> {
    static R: std::sync::OnceLock<Mutex<Vec<RoleRecord>>> = std::sync::OnceLock::new();
    R.get_or_init(|| Mutex::new(Vec::new()))
}

static SELFCHECK_FAILURES: AtomicU64 = AtomicU64::new(0);

/// Replace the published records with this load's (one per role that installed anything).
pub fn publish(entries: &[RoleRecord]) {
    let mut r = records().lock();
    r.clear();
    r.extend(entries.iter().copied().filter(|e| e.format.is_some()));
}

/// The records of the last load.
pub fn snapshot() -> Vec<RoleRecord> {
    records().lock().clone()
}

/// Count one failed load-time self-check (monotonic over the process).
pub fn note_selfcheck_failure() {
    SELFCHECK_FAILURES.fetch_add(1, Ordering::Relaxed);
}

/// Failed self-checks so far.
pub fn selfcheck_failures() -> u64 {
    SELFCHECK_FAILURES.load(Ordering::Relaxed)
}

/// The `/metrics` text of the three series, appended to `out`.
pub fn render_prometheus(out: &mut String) {
    render_from(
        out,
        &snapshot(),
        selfcheck_failures(),
        crate::port::w4::prefill_declines(),
    );
}

/// [`render_prometheus`] over explicit inputs (pure, for the tests).
pub fn render_from(out: &mut String, recs: &[RoleRecord], failures: u64, declines: u64) {
    let name = "atlas_w4_weight_bytes";
    let _ = writeln!(
        out,
        "# HELP {name} Resident device bytes of the 4-bit weights installed at load, by layer type and format.\n\
         # TYPE {name} gauge"
    );
    for role in W4Role::ALL {
        match recs.iter().find(|r| r.role == role && r.format.is_some()) {
            Some(r) => {
                let f = r.format.map(|f| f.to_string()).unwrap_or_default();
                let _ = writeln!(
                    out,
                    "{name}{{role=\"{}\",format=\"{f}\"}} {}",
                    role.label(),
                    r.bytes
                );
            }
            None => {
                let _ = writeln!(out, "{name}{{role=\"{}\",format=\"none\"}} 0", role.label());
            }
        }
    }
    let name = "atlas_w4_selfcheck_failures_total";
    let _ = writeln!(
        out,
        "# HELP {name} Load-time 4-bit self-checks (M=1 rows against one M=4 decode) that were not bit-equal; \
         the layer type then runs its batched rows one at a time.\n# TYPE {name} counter\n{name} {failures}"
    );
    let name = "atlas_affine4_prefill_int8_fallbacks_total";
    let _ = writeln!(
        out,
        "# HELP {name} 4-bit prompt GEMM calls that declined (an int8 exceptional group, or a format without a \
         prompt GEMM) and ran the BF16 dequant fallback.\n# TYPE {name} counter\n{name} {declines}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_off_renders_zeros_for_every_role() {
        let mut s = String::new();
        render_from(&mut s, &[], 0, 0);
        for role in ["ffn", "gdn", "attn", "lm_head"] {
            assert!(
                s.contains(&format!(
                    "atlas_w4_weight_bytes{{role=\"{role}\",format=\"none\"}} 0\n"
                )),
                "{s}"
            );
        }
        assert!(s.contains("\natlas_w4_selfcheck_failures_total 0\n"), "{s}");
        assert!(
            s.contains("\natlas_affine4_prefill_int8_fallbacks_total 0\n"),
            "{s}"
        );
        assert_eq!(s.matches("# TYPE ").count(), 3);
        assert!(!s.contains('\u{2014}'), "no em dashes in metric help");
    }

    #[test]
    fn an_installed_role_reports_its_format_and_bytes() {
        let recs = [RoleRecord {
            role: W4Role::Ffn,
            format: Some(W4Format::Nvfp4),
            layers: 32,
            bytes: 1_274_019_840,
            per_row: false,
        }];
        let mut s = String::new();
        render_from(&mut s, &recs, 2, 5);
        assert!(
            s.contains("atlas_w4_weight_bytes{role=\"ffn\",format=\"nvfp4\"} 1274019840\n"),
            "{s}"
        );
        assert!(
            s.contains("atlas_w4_weight_bytes{role=\"gdn\",format=\"none\"} 0\n"),
            "{s}"
        );
        assert!(s.contains("\natlas_w4_selfcheck_failures_total 2\n"), "{s}");
        assert!(
            s.contains("\natlas_affine4_prefill_int8_fallbacks_total 5\n"),
            "{s}"
        );
    }

    #[test]
    fn publish_keeps_only_installed_roles_and_replaces_the_last_load() {
        // One test owns the global records, so the publish/replace sequence is not raced by another test.
        publish(&[
            RoleRecord {
                format: Some(W4Format::Affine { bits: 4, group: 64 }),
                layers: 2,
                bytes: 10,
                ..RoleRecord::none(W4Role::Gdn)
            },
            RoleRecord::none(W4Role::Ffn),
        ]);
        let s = snapshot();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].role, W4Role::Gdn);
        publish(&[]);
        assert!(
            snapshot().is_empty(),
            "a reload with every row off clears the gauge"
        );
    }

    #[test]
    fn labels_are_stable() {
        let l: Vec<_> = W4Role::ALL.iter().map(|r| r.label()).collect();
        assert_eq!(l, ["ffn", "gdn", "attn", "lm_head"]);
    }
}
