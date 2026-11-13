// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the K-generic half of GDN write-on-accept for the batched
//! verify (port lane L10, row `gdn_woa_mtp`): the Ada twin
//! `gated_delta_rule_wyn_woa_<K>` (K = 2, 3, 5..16) and its fold
//! `gated_delta_rule_wyn_fold` (`kernels/ada/common/gated_delta_rule_wyn_woa.cu`),
//! their handles, and the two pure choices the layer makes with them: which
//! twin a width launches and which fold kernel commits it.
//!
//! K = 4 is NOT here: it keeps the gb10-common pair (`woa.rs`), so the
//! shipped K = 4 path is unchanged on every target. Everything in this file
//! is reachable only with the row on, on a target whose kernel set carries
//! the Ada module: the handles are resolved with `try_target_kernel` behind
//! the row, so a row-off serve (and every non-Ada target) issues no lookup at
//! all, leaves no failed row in the startup kernel audit, and every handle
//! stays `KernelHandle(0)`.
//!
//! Design after the ReplaySSM record-and-fold idea (Cinference / NInfer,
//! `docs/maintainer/replayssm-gdn.md`, Apache-2.0). Idea only: no upstream
//! code is copied.

use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// The Ada module (file stem of the kernel source).
pub(super) const WYN_WOA_MODULE: &str = "gated_delta_rule_wyn_woa";
/// The K-generic fold entry point.
pub(super) const WYN_FOLD_NAME: &str = "gated_delta_rule_wyn_fold";

/// Twin entry points, index = K - 2 for K in 2..=16. The K = 4 slot is empty
/// on purpose: that width is served by the gb10-common `gated_delta_rule_wy4_woa`.
pub(super) const WYN_WOA_NAMES: [Option<&str>; 15] = [
    Some("gated_delta_rule_wyn_woa_2"),
    Some("gated_delta_rule_wyn_woa_3"),
    None,
    Some("gated_delta_rule_wyn_woa_5"),
    Some("gated_delta_rule_wyn_woa_6"),
    Some("gated_delta_rule_wyn_woa_7"),
    Some("gated_delta_rule_wyn_woa_8"),
    Some("gated_delta_rule_wyn_woa_9"),
    Some("gated_delta_rule_wyn_woa_10"),
    Some("gated_delta_rule_wyn_woa_11"),
    Some("gated_delta_rule_wyn_woa_12"),
    Some("gated_delta_rule_wyn_woa_13"),
    Some("gated_delta_rule_wyn_woa_14"),
    Some("gated_delta_rule_wyn_woa_15"),
    Some("gated_delta_rule_wyn_woa_16"),
];

/// Resolved K-generic write-on-accept handles. All zero when the row is off
/// or the module is absent.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WynWoaKernels {
    /// Index = K - 2 (K = 2..=16); the K = 4 entry is always zero.
    pub twin: [KernelHandle; 15],
    pub fold: KernelHandle,
}

impl Default for WynWoaKernels {
    /// Nothing linked (`KernelHandle` carries no `Default` of its own).
    fn default() -> Self {
        Self {
            twin: [KernelHandle(0); 15],
            fold: KernelHandle(0),
        }
    }
}

impl WynWoaKernels {
    /// The twin for verify width `k`, when it and the fold both resolved
    /// (a twin whose fold is missing could never be committed).
    pub(super) fn twin_for(&self, k: usize) -> Option<KernelHandle> {
        twin_index(k)
            .map(|i| self.twin[i])
            .filter(|h| h.0 != 0 && self.fold.0 != 0)
    }

    /// The K-generic fold, when resolved.
    pub(super) fn fold_kernel(&self) -> Option<KernelHandle> {
        (self.fold.0 != 0).then_some(self.fold)
    }
}

/// `WYN_WOA_NAMES` index for width `k`, `None` outside 2..=16 and at 4.
pub(super) const fn twin_index(k: usize) -> Option<usize> {
    if k < 2 || k > 16 || k == 4 {
        None
    } else {
        Some(k - 2)
    }
}

/// Resolve the handles once, at layer construction (never inside a capture).
/// `row_on` is the `gdn_woa_mtp` row: off means no lookup is issued.
pub(super) fn resolve(gpu: &dyn GpuBackend, row_on: bool) -> WynWoaKernels {
    if !row_on || !gpu.has_module(WYN_WOA_MODULE) {
        return WynWoaKernels::default();
    }
    let lookup = |name: &str| crate::layers::try_target_kernel(gpu, WYN_WOA_MODULE, name);
    let mut out = WynWoaKernels {
        fold: lookup(WYN_FOLD_NAME),
        ..WynWoaKernels::default()
    };
    for (slot, name) in out.twin.iter_mut().zip(WYN_WOA_NAMES) {
        if let Some(name) = name {
            *slot = lookup(name);
        }
    }
    // Every GDN layer resolves the same module: say it once per process.
    static LOGGED: std::sync::Once = std::sync::Once::new();
    LOGGED.call_once(|| {
        let linked = out.twin.iter().filter(|h| h.0 != 0).count();
        tracing::info!(
            "GDN write-on-accept (gdn_woa_mtp): K-generic twin module resolved: {linked}/14 \
             widths, fold {}",
            if out.fold.0 != 0 {
                "linked"
            } else {
                "MISSING (twins unused)"
            }
        );
    });
    out
}

/// Which fold kernel commits a batched verify of width `k_rows`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FoldKernel {
    /// The gb10-common `gated_delta_rule_wy4_fold` (stash at K = 4 layout;
    /// its flag-0 arm restores from the Hi tables for any width).
    Wy4,
    /// The Ada `gated_delta_rule_wyn_fold` (stash at the `k_rows` layout;
    /// the same flag-0 arm, byte for byte).
    Wyn,
}

/// The fold choice as a pure function. K = 4 always takes the wy4 fold (the
/// shipped path). Any other width takes the K-generic fold only when it is
/// linked (it is resolved only with the row on); otherwise the wy4 fold's
/// flag-0 restore runs as before, which is correct because without the
/// K-generic fold no K != 4 twin can have engaged (`twin_for`).
pub(super) const fn fold_choice(k_rows: usize, wyn_fold_linked: bool) -> FoldKernel {
    if k_rows != 4 && wyn_fold_linked {
        FoldKernel::Wyn
    } else {
        FoldKernel::Wy4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_contract_skips_k4_and_covers_2_to_16() {
        assert_eq!(twin_index(1), None);
        assert_eq!(twin_index(2), Some(0));
        assert_eq!(twin_index(3), Some(1));
        assert_eq!(twin_index(4), None);
        assert_eq!(twin_index(5), Some(3));
        assert_eq!(twin_index(16), Some(14));
        assert_eq!(twin_index(17), None);
        for k in 2..=16usize {
            match twin_index(k) {
                Some(i) => {
                    let name = WYN_WOA_NAMES[i].expect("named");
                    assert_eq!(name, format!("gated_delta_rule_wyn_woa_{k}"));
                }
                None => assert_eq!(k, 4),
            }
        }
        assert!(WYN_WOA_NAMES[2].is_none(), "the K=4 slot stays empty");
    }

    #[test]
    fn a_twin_without_its_fold_is_never_offered() {
        let raw = |h: Option<KernelHandle>| h.map(|h| h.0);
        let mut k = WynWoaKernels::default();
        k.twin[0] = KernelHandle(7);
        assert_eq!(raw(k.twin_for(2)), None, "fold missing");
        k.fold = KernelHandle(9);
        assert_eq!(raw(k.twin_for(2)), Some(7));
        assert_eq!(raw(k.twin_for(3)), None, "K=3 twin missing");
        assert_eq!(raw(k.twin_for(4)), None, "K=4 is the wy4 pair's");
        assert_eq!(raw(k.fold_kernel()), Some(9));
        assert_eq!(raw(WynWoaKernels::default().fold_kernel()), None);
    }

    #[test]
    fn k4_always_folds_through_the_shipped_kernel() {
        assert_eq!(fold_choice(4, false), FoldKernel::Wy4);
        assert_eq!(fold_choice(4, true), FoldKernel::Wy4);
        for k in [2, 3, 5, 8, 16] {
            assert_eq!(fold_choice(k, false), FoldKernel::Wy4, "k={k}");
            assert_eq!(fold_choice(k, true), FoldKernel::Wyn, "k={k}");
        }
    }

    /// The source declares exactly the entry points this file looks up.
    #[test]
    fn the_kernel_source_declares_every_looked_up_name() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/ada/common/gated_delta_rule_wyn_woa.cu"
        ))
        .expect("kernel source");
        for k in [2usize, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] {
            assert!(
                src.contains(&format!("AVAROK_WYN_WOA_INSTANTIATE({k})")),
                "K={k} not instantiated"
            );
        }
        assert!(!src.contains("AVAROK_WYN_WOA_INSTANTIATE(4)"));
        assert!(src.contains("void gated_delta_rule_wyn_woa_##K("));
        assert!(src.contains(&format!("void {WYN_FOLD_NAME}(")));
        assert!(!src.contains('\u{2014}'), "no em dashes in new text");
    }
}
