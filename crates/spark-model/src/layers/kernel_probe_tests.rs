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

/// A module the target DID build, holding a name its code object does not
/// define. `avarok_core::registry` refuses that lookup itself rather than
/// trusting the driver: observed on SCALE 1.7.1 / gfx1201, `cuModuleGetFunction`
/// on the compiled-out `nvfp4_mmq` returns SUCCESS with a handle backed by no
/// code, and the first launch through it dies with CUDA_ERROR_INVALID_IMAGE
/// (200) on `avarok_nvfp4_repack`. The refusal is an ordinary `Err`, so the
/// optional-kernel probe degrades to handle 0 exactly as it does on NVIDIA,
/// where the same lookup fails with "not found".
#[test]
fn a_refused_lookup_degrades_to_handle_zero() {
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel("nvfp4_mmq", "avarok_nvfp4_repack");
    // The module IS present in the build, so the target-scoped probe does not
    // short-circuit: the lookup below is issued and the error is what turns it
    // into handle 0 (the snapshot assertion is what proves both).
    let h = try_target_kernel(&gpu, "nvfp4_mmq", "avarok_nvfp4_repack");
    assert_eq!(h.0, 0);
    assert_eq!(
        gpu.kernel_lookups_snapshot(),
        vec![("nvfp4_mmq".to_string(), "avarok_nvfp4_repack".to_string())]
    );
}
