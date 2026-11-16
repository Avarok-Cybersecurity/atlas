// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::layers::dense_ffn::ada_kernels::SWIGLU_QUANT_KERNEL;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use std::path::{Path, PathBuf};

const K: KernelHandle = KernelHandle(0x5A1);

fn repo(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
}

#[test]
fn every_clause_of_the_rule_is_load_bearing() {
    let silu = FfnActivation::SiLU;
    assert!(swiglu_quant_selected(K, silu, true, 9216));
    assert!(swiglu_quant_selected(K, silu, true, 3584));
    // Off sm_89 the module is never built.
    assert!(!swiglu_quant_selected(KernelHandle(0), silu, true, 9216));
    // GeLU's `act_mul` is `gelu_mul`, which the fused kernel does not compute.
    assert!(!swiglu_quant_selected(K, FfnActivation::GeLU, true, 9216));
    // A W8A16 down projection reads the BF16 product the fused arm never writes.
    assert!(!swiglu_quant_selected(K, silu, false, 9216));
    // One scale per 128-wide group.
    assert!(!swiglu_quant_selected(K, silu, true, 9216 + 64));
}

/// The kernel's per-CTA span, mirrored from the `.cu` (`gpc`, `g0`, `g_end`,
/// eight groups per iteration).
fn groups_covered(k: u32, grid_y: u32) -> Vec<u32> {
    let l = k / 128;
    let gpc = l.div_ceil(grid_y);
    let mut seen = Vec::new();
    for by in 0..grid_y {
        let g0 = by * gpc;
        if g0 >= l {
            continue;
        }
        let g_end = (g0 + gpc).min(l);
        for it in 0..gpc.div_ceil(ops::SWIGLU_QUANT_GROUPS_PER_CTA) {
            for sub in 0..ops::SWIGLU_QUANT_GROUPS_PER_CTA {
                let g = g0 + it * ops::SWIGLU_QUANT_GROUPS_PER_CTA + sub;
                if g < g_end {
                    seen.push(g);
                }
            }
        }
    }
    seen
}

#[test]
fn the_launch_grid_covers_every_group_of_a_row_exactly_once() {
    for k in [128, 1024, 2560, 3584, 9216, 128 * 9] {
        let [m, gy, gz] = ops::swiglu_quant_grid(3921, k);
        assert_eq!((m, gz), (3921, 1));
        let mut seen = groups_covered(k, gy);
        seen.sort_unstable();
        assert_eq!(seen, (0..k / 128).collect::<Vec<_>>(), "k={k}");
    }
}

#[test]
fn the_launcher_passes_the_kernel_its_six_arguments_on_one_launch() {
    let gpu = MockGpuBackend::new();
    let h = gpu
        .kernel(SWIGLU_QUANT_KERNEL.0, SWIGLU_QUANT_KERNEL.1)
        .unwrap();
    let bufs: Vec<_> = (0..4).map(|_| gpu.alloc(4096).unwrap()).collect();
    ops::swiglu_quant_fp8_ada(&gpu, h, bufs[0], bufs[1], bufs[2], bufs[3], 17, 9216, 0).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    let l = &launches[0];
    assert_eq!(
        (l.grid, l.block, l.shared_mem),
        ([17, 9, 1], [128, 1, 1], 0)
    );
    assert_eq!(l.args.len(), 6);
}

#[test]
fn the_fused_kernel_spells_the_silu_exactly_as_moe_silu_mul_does() {
    // Bit identity rests on the same source expression compiled with the same
    // flags. If either side is edited, this fails before a card run would.
    let fused =
        std::fs::read_to_string(repo("kernels/ada/common/swiglu_quant_fp8_ada.cu")).unwrap();
    let pair = std::fs::read_to_string(repo("kernels/gb10/common/moe_silu_mul.cu")).unwrap();
    for line in [
        "float sigmoid_g = 1.0f / (1.0f + __expf(-g));",
        "float result = g * sigmoid_g * u;",
    ] {
        assert!(
            fused.contains(line),
            "swiglu_quant_fp8_ada.cu lost `{line}`"
        );
        assert!(
            pair.contains(line),
            "moe_silu_mul.cu no longer has `{line}`"
        );
    }
    assert!(fused.contains("__bfloat162float(__float2bfloat16(result))"));
    assert!(fused.contains("__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)"));
}

#[test]
fn no_ada_target_shadows_moe_silu_mul() {
    // The fused kernel lives in common/ and matches common's `moe_silu_mul`. A
    // model directory with its own moe_silu_mul.cu (gb10's swiglu_limit models
    // carry one) would change the unfused product and not the fused one.
    let common = repo("kernels/ada/common/moe_silu_mul.cu")
        .canonicalize()
        .unwrap();
    let mut checked = 0;
    for model in std::fs::read_dir(repo("kernels/ada")).unwrap() {
        let model = model.unwrap().path();
        if !model.is_dir() || model.ends_with("common") {
            continue;
        }
        for quant in std::fs::read_dir(&model).unwrap() {
            let quant = quant.unwrap().path();
            let shadow = quant.join("moe_silu_mul.cu");
            if quant.is_dir() {
                checked += 1;
                assert!(
                    !shadow.exists() || shadow.canonicalize().unwrap() == common,
                    "{} shadows moe_silu_mul.cu; drop swiglu_quant_fp8_ada for it",
                    shadow.display()
                );
            }
        }
    }
    assert!(checked > 0, "scanned no ada model directory");
}
