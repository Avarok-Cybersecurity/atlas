// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

fn lookups(gpu: &MockGpuBackend) -> Vec<(String, String)> {
    gpu.kernel_lookups_snapshot()
}

fn pair(e: Entry) -> (String, String) {
    (e.0.to_owned(), e.1.to_owned())
}

#[test]
fn a_target_that_compiles_the_sm89_modules_takes_the_drop_ins_and_nothing_else() {
    let gpu = MockGpuBackend::new();
    let k = Fp8FfnKernels::resolve(&gpu);
    assert!(k.gemv.0 != 0 && k.gemv_dual.0 != 0 && k.gemv_silu_input.0 != 0 && k.gemm_w8a8.0 != 0);
    let seen = lookups(&gpu);
    for (drop_in, base) in FP8_FFN_KERNELS {
        assert!(seen.contains(&pair(drop_in)), "{drop_in:?} not looked up");
        assert!(
            !seen.contains(&pair(base)),
            "{base:?} looked up although its drop-in resolved"
        );
    }
    assert!(k.swiglu_quant.0 != 0);
    assert!(seen.contains(&pair(SWIGLU_QUANT_KERNEL)));
}

#[test]
fn a_target_without_the_sm89_modules_issues_no_lookup_for_them() {
    // gb10, Hopper, Blackwell: the modules were never built, so the boot
    // audit must not see a failed lookup for them (it fails CLOSED on one).
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent("fp8_gemv_ada");
    gpu.mark_module_absent("fp8_gemm_ada");
    gpu.mark_module_absent(SWIGLU_QUANT_KERNEL.0);
    let k = Fp8FfnKernels::resolve(&gpu);
    assert!(k.gemv.0 != 0 && k.gemm_w8a8.0 != 0);
    assert_eq!(k.swiglu_quant.0, 0, "no fused SwiGLU quant off sm_89");
    let seen = lookups(&gpu);
    assert!(!seen.contains(&pair(SWIGLU_QUANT_KERNEL)));
    for (drop_in, base) in FP8_FFN_KERNELS {
        assert!(
            !seen.contains(&pair(drop_in)),
            "{drop_in:?} looked up on a target without it"
        );
        assert!(seen.contains(&pair(base)), "{base:?} not looked up");
    }
}

#[test]
fn a_drop_in_whose_entry_fails_to_resolve_falls_back_to_the_kernel_it_replaces() {
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel("fp8_gemm_ada", "fp8_gemm_ada_w8a8");
    let k = Fp8FfnKernels::resolve(&gpu);
    assert!(k.gemm_w8a8.0 != 0);
    assert!(lookups(&gpu).contains(&pair(("fp8_gemm_t_blockscaled", "fp8_gemm_t_blockscaled"))));
}

#[test]
fn the_drop_ins_are_the_entry_points_the_ada_sources_declare() {
    // The module name is the .cu stem and the entry is an `extern "C"` name in
    // it. This catches a rename on either side without needing nvcc.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/ada/common");
    let drop_ins = FP8_FFN_KERNELS.map(|(d, _)| d);
    for drop_in in drop_ins.into_iter().chain([SWIGLU_QUANT_KERNEL]) {
        let src = std::fs::read_to_string(root.join(format!("{}.cu", drop_in.0)))
            .unwrap_or_else(|e| panic!("kernels/ada/common/{}.cu: {e}", drop_in.0));
        assert!(
            src.contains(&format!(" {}(", drop_in.1)),
            "kernels/ada/common/{}.cu declares no `{}`",
            drop_in.0,
            drop_in.1
        );
    }
}
