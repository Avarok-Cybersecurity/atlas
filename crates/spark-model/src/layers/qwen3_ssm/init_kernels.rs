// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel lookups the SSM layer's constructor makes conditionally.
//!
//! Split from `init.rs` for the 500-LoC cap, which the file crossed by one
//! line when the LoRA `out_proj` slot joined the layer. Exact piecewise copy.

use super::*;

/// Resolve one `hyper_connection` entry point, but ONLY for a model that
/// carries the highway. Skipping the lookup rather than discarding its result
/// is the point: an un-issued lookup leaves no failed row in the fail-closed
/// startup audit, so what remains there is what someone has to act on.
#[track_caller]
pub(super) fn hc_kernel(
    config: &atlas_core::config::ModelConfig,
    gpu: &dyn GpuBackend,
    func: &str,
) -> KernelHandle {
    if config.hc_mult > 0 {
        crate::layers::try_kernel(gpu, "hyper_connection", func)
    } else {
        KernelHandle(0)
    }
}

/// Chain-verify K=5..16 WY kernels (one templated gb10-common module;
/// K=9..16 arrived 2026-08-29 with the gamma>8 window). Index = K-5; a NULL
/// handle means the target lacks the module, in which case that width keeps
/// the sequential per-token path.
///
/// Split out of `init.rs` with its FP16 twin for the 500-LoC cap. Exact
/// piecewise copy — the index contract is the load-bearing part and is
/// unchanged.
pub(super) fn wyn_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    [
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy5"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy6"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy7"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy8"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy9"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy10"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy11"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy12"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy13"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy14"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy15"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy16"),
    ]
}

/// FP16 h-state twins (K=5..16), same module and the SAME index contract as
/// [`wyn_kernels`] — a mismatch between the two would silently pair a width
/// with another width's twin.
pub(super) fn wyn_f16_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    [
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy5_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy6_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy7_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy8_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy9_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy10_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy11_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy12_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy13_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy14_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy15_f16"),
        crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", "gated_delta_rule_wy16_f16"),
    ]
}

/// Cross-sequence batched-verify pointer-table twins of [`wyn_kernels`]
/// (`state_is_table` compiled in as 1). ADDITIVE symbols: the contiguous
/// forms above keep their exact signatures, so the shared
/// `gdn_decode_wyn` launch (also used by the per-target wy17) is never
/// perturbed. Same index contract as `wyn_kernels`.
// provenance-id: 526f6e616c6420522e205374657369616b
/// Exported so the unit tests can hold the index contract (index = K - 5)
/// against the literal symbol list — the same trap the #831 lever test
/// closed: test the thing the server actually loads, not a copy.
pub(super) const WYN_TABLE_NAMES: [&str; 12] = [
    "gated_delta_rule_wy5_table",
    "gated_delta_rule_wy6_table",
    "gated_delta_rule_wy7_table",
    "gated_delta_rule_wy8_table",
    "gated_delta_rule_wy9_table",
    "gated_delta_rule_wy10_table",
    "gated_delta_rule_wy11_table",
    "gated_delta_rule_wy12_table",
    "gated_delta_rule_wy13_table",
    "gated_delta_rule_wy14_table",
    "gated_delta_rule_wy15_table",
    "gated_delta_rule_wy16_table",
];

pub(super) fn wyn_table_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    WYN_TABLE_NAMES.map(|n| crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", n))
}

/// FP16 twins of [`wyn_table_kernels`]; same index contract.
/// FP16 twin names; same index contract as [`WYN_TABLE_NAMES`].
pub(super) const WYN_F16_TABLE_NAMES: [&str; 12] = [
    "gated_delta_rule_wy5_f16_table",
    "gated_delta_rule_wy6_f16_table",
    "gated_delta_rule_wy7_f16_table",
    "gated_delta_rule_wy8_f16_table",
    "gated_delta_rule_wy9_f16_table",
    "gated_delta_rule_wy10_f16_table",
    "gated_delta_rule_wy11_f16_table",
    "gated_delta_rule_wy12_f16_table",
    "gated_delta_rule_wy13_f16_table",
    "gated_delta_rule_wy14_f16_table",
    "gated_delta_rule_wy15_f16_table",
    "gated_delta_rule_wy16_f16_table",
];

pub(super) fn wyn_f16_table_kernels(gpu: &dyn GpuBackend) -> [KernelHandle; 12] {
    WYN_F16_TABLE_NAMES.map(|n| crate::layers::try_kernel(gpu, "gated_delta_rule_wyn", n))
}

#[cfg(test)]
mod table_registry_tests {
    use super::{WYN_F16_TABLE_NAMES, WYN_TABLE_NAMES};

    /// Index contract: entry i must be the K = i+5 symbol, for both dtypes.
    /// A mismatch would silently pair a verify width with another width's
    /// kernel — the exact failure the registry comments warn about.
    #[test]
    fn table_names_hold_the_index_contract() {
        for (i, (n, f)) in WYN_TABLE_NAMES.iter().zip(WYN_F16_TABLE_NAMES).enumerate() {
            let k = i + 5;
            assert_eq!(*n, format!("gated_delta_rule_wy{k}_table"), "fp32 index {i}");
            assert_eq!(f, format!("gated_delta_rule_wy{k}_f16_table"), "f16 index {i}");
        }
    }

    /// The dispatcher's widest arm is K=16; the registries must cover it
    /// and start exactly at the first non-wy4 width.
    #[test]
    fn table_registry_covers_k5_through_k16() {
        assert_eq!(WYN_TABLE_NAMES.len(), 12);
        assert!(WYN_TABLE_NAMES[0].contains("wy5_"));
        assert!(WYN_TABLE_NAMES[11].contains("wy16_"));
    }
}
