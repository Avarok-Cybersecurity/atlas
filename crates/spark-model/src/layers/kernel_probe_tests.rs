// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

/// A module the build does not carry is never looked up — the boot audit
/// would otherwise record a silent-fallback site the target does not have.
/// A module it does carry is looked up exactly as `try_kernel` does.
#[test]
fn a_module_the_target_never_built_is_not_looked_up() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent("gdn_fwd_o_hopper");
    let h = try_target_kernel(
        &gpu,
        "gdn_fwd_o_hopper",
        "gated_delta_rule_chunk_fwd_o_hopper",
    );
    assert_eq!(h.0, 0);
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "no lookup may reach the backend for an absent module"
    );
    // NEGATIVE CONTROL: the plain probe DOES issue the lookup, which is the
    // audit entry this helper exists to avoid.
    let _ = try_kernel(
        &gpu,
        "gdn_fwd_o_hopper",
        "gated_delta_rule_chunk_fwd_o_hopper",
    );
    assert_eq!(gpu.kernel_lookups_snapshot().len(), 1);
    // A compiled module: looked up, handle returned.
    let h = try_target_kernel(&gpu, "ssm_preprocess", "dense_gemm_ba_gates_prefill");
    assert_ne!(h.0, 0);
    assert_eq!(gpu.kernel_lookups_snapshot().len(), 2);
}
