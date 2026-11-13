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

const ALL_ROWS: AdaFp8Rows = AdaFp8Rows {
    l2_evict_first: true,
    verify_gemv: true,
    rmsnorm_quant: true,
    decode_gemv: true,
};

fn module_lookups(gpu: &MockGpuBackend, module: &str) -> usize {
    lookups(gpu).iter().filter(|(m, _)| m == module).count()
}

#[test]
fn with_every_port_row_off_the_lookups_are_the_pre_sprint_ones() {
    let gpu = MockGpuBackend::new();
    let k = Fp8FfnKernels::resolve_with(&gpu, AdaFp8Rows::default());
    assert!(k.gemv.0 != 0);
    let seen = lookups(&gpu);
    for twin in FP8_FFN_L2EF_KERNELS {
        assert!(
            !seen.contains(&pair(twin)),
            "{twin:?} looked up with the row off"
        );
    }
    assert_eq!(module_lookups(&gpu, verify::BATCHM_MODULE), 0);
    assert!(!seen.contains(&pair(RMSNORM_QUANT_KERNEL)));
    let mut expected: Vec<(String, String)> =
        FP8_FFN_KERNELS.iter().map(|(d, _)| pair(*d)).collect();
    expected.push(pair(SWIGLU_QUANT_KERNEL));
    assert_eq!(seen, expected, "the exact pre-sprint lookup sequence");
}

#[test]
fn the_l2_row_takes_each_decode_twin_and_skips_the_plain_drop_in() {
    let gpu = MockGpuBackend::new();
    let rows = AdaFp8Rows {
        l2_evict_first: true,
        ..AdaFp8Rows::default()
    };
    let k = Fp8FfnKernels::resolve_with(&gpu, rows);
    assert!(k.gemv.0 != 0 && k.gemv_dual.0 != 0 && k.gemv_silu_input.0 != 0);
    let seen = lookups(&gpu);
    for (role, twin) in FP8_FFN_L2EF_KERNELS.into_iter().enumerate() {
        assert!(seen.contains(&pair(twin)), "{twin:?} not looked up");
        let (drop_in, _) = FP8_FFN_KERNELS[role];
        assert!(
            !seen.contains(&pair(drop_in)),
            "{drop_in:?} looked up although its twin resolved"
        );
    }
    // The prefill GEMM has no twin.
    assert!(seen.contains(&pair(FP8_FFN_KERNELS[3].0)));
}

#[test]
fn the_l2_row_on_a_target_without_the_module_issues_no_twin_lookup() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent("fp8_gemv_ada");
    let k = Fp8FfnKernels::resolve_with(
        &gpu,
        AdaFp8Rows {
            l2_evict_first: true,
            ..AdaFp8Rows::default()
        },
    );
    assert!(k.gemv.0 != 0);
    let seen = lookups(&gpu);
    for twin in FP8_FFN_L2EF_KERNELS {
        assert!(!seen.contains(&pair(twin)));
    }
    assert!(
        seen.contains(&pair(FP8_FFN_KERNELS[0].1)),
        "gb10's kernel, as before"
    );
}

#[test]
fn a_twin_that_fails_to_resolve_falls_back_to_the_plain_drop_in() {
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel("fp8_gemv_ada", "fp8_gemv_ada_l2ef");
    let rows = AdaFp8Rows {
        l2_evict_first: true,
        ..AdaFp8Rows::default()
    };
    let h = Fp8FfnKernels::m1_gemv_with(&gpu, rows);
    assert!(h.0 != 0);
    assert!(lookups(&gpu).contains(&pair(FP8_FFN_KERNELS[0].0)));
}

#[test]
fn the_verify_row_primes_every_batchm_entry_once_through_the_op_cache() {
    let gpu = MockGpuBackend::new();
    let rows = AdaFp8Rows {
        verify_gemv: true,
        ..AdaFp8Rows::default()
    };
    Fp8FfnKernels::resolve_with(&gpu, rows);
    let primed = module_lookups(&gpu, verify::BATCHM_MODULE);
    assert_eq!(primed, 3 * (verify::MAX_ROWS - verify::MIN_ROWS + 1));
    // A second layer (or the forward-time read) hits the cache.
    Fp8FfnKernels::resolve_with(&gpu, rows);
    for m in verify::MIN_ROWS..=verify::MAX_ROWS {
        assert!(
            verify::RowKernels::cached(&gpu, m, false).is_some(),
            "m={m}"
        );
    }
    assert_eq!(
        module_lookups(&gpu, verify::BATCHM_MODULE),
        primed,
        "no lookup after the first"
    );
    // The plain names, not the twins.
    let seen = lookups(&gpu);
    assert!(
        seen.iter()
            .all(|(m, f)| m != verify::BATCHM_MODULE || !f.contains("l2ef"))
    );
}

#[test]
fn the_decode_row_alone_primes_the_same_batchm_entries() {
    // `ffn_fp8_decode_gemv` reads the verify row's handles at forward time, so
    // it must prime them at construction whether or not the verify row is on.
    let gpu = MockGpuBackend::new();
    let rows = AdaFp8Rows {
        decode_gemv: true,
        ..AdaFp8Rows::default()
    };
    Fp8FfnKernels::resolve_with(&gpu, rows);
    let primed = module_lookups(&gpu, verify::BATCHM_MODULE);
    assert_eq!(primed, 3 * (verify::MAX_ROWS - verify::MIN_ROWS + 1));
    // Both rows on prime once, not twice.
    let both = MockGpuBackend::new();
    Fp8FfnKernels::resolve_with(
        &both,
        AdaFp8Rows {
            decode_gemv: true,
            verify_gemv: true,
            ..AdaFp8Rows::default()
        },
    );
    assert_eq!(module_lookups(&both, verify::BATCHM_MODULE), primed);
}

#[test]
fn the_verify_row_with_the_l2_row_primes_the_twins_instead() {
    let gpu = MockGpuBackend::new();
    Fp8FfnKernels::resolve_with(&gpu, ALL_ROWS);
    let seen = lookups(&gpu);
    let batchm: Vec<_> = seen
        .iter()
        .filter(|(m, _)| m == verify::BATCHM_MODULE)
        .collect();
    assert_eq!(batchm.len(), 21);
    assert!(
        batchm.iter().all(|(_, f)| f.contains("_l2ef_")),
        "{batchm:?}"
    );
}

#[test]
fn the_port_rows_on_a_target_without_the_modules_issue_no_lookup() {
    let gpu = MockGpuBackend::new();
    for m in [
        "fp8_gemv_ada",
        "fp8_gemm_ada",
        verify::BATCHM_MODULE,
        RMSNORM_QUANT_KERNEL.0,
    ] {
        gpu.mark_module_absent(m);
    }
    gpu.mark_module_absent(SWIGLU_QUANT_KERNEL.0);
    Fp8FfnKernels::resolve_with(&gpu, ALL_ROWS);
    assert_eq!(module_lookups(&gpu, verify::BATCHM_MODULE), 0);
    assert_eq!(module_lookups(&gpu, RMSNORM_QUANT_KERNEL.0), 0);
    assert_eq!(rmsnorm_quant_kernel(&gpu).0, 0);
    assert!(verify::RowKernels::cached(&gpu, 4, false).is_none());
}

#[test]
fn the_norm_row_primes_the_fused_quantizer() {
    let gpu = MockGpuBackend::new();
    Fp8FfnKernels::resolve_with(
        &gpu,
        AdaFp8Rows {
            rmsnorm_quant: true,
            ..AdaFp8Rows::default()
        },
    );
    assert!(lookups(&gpu).contains(&pair(RMSNORM_QUANT_KERNEL)));
    let before = lookups(&gpu).len();
    assert!(rmsnorm_quant_kernel(&gpu).0 != 0);
    assert_eq!(
        lookups(&gpu).len(),
        before,
        "the forward-time read is a cache hit"
    );
}

#[test]
fn every_port_entry_is_declared_by_its_ada_source() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/ada/common");
    let mut entries: Vec<Entry> = FP8_FFN_L2EF_KERNELS.to_vec();
    entries.push(RMSNORM_QUANT_KERNEL);
    for m in verify::MIN_ROWS..=verify::MAX_ROWS {
        for l2ef in [false, true] {
            for name in verify::RowKernels::names(m, l2ef).expect("in range") {
                entries.push((verify::BATCHM_MODULE, name));
            }
        }
    }
    for (module, entry) in entries {
        let src = std::fs::read_to_string(root.join(format!("{module}.cu")))
            .unwrap_or_else(|e| panic!("kernels/ada/common/{module}.cu: {e}"));
        assert!(
            src.contains(&format!(" {entry}(")),
            "kernels/ada/common/{module}.cu declares no `{entry}`"
        );
    }
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
