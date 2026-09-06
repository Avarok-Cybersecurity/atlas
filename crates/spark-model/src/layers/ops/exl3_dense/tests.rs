// SPDX-License-Identifier: AGPL-3.0-only

//! Mock-backend launch-plan tests for `ops::exl3_dense` (split from the
//! parent for the 500-LoC cap): the recorded kernel resolutions ARE the
//! dispatch plan per tier; numerics are the GPU parity example's job.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

fn weight(gpu: &MockGpuBackend, k: usize, n: usize) -> Exl3DenseWeight {
    weight_k(gpu, k, n, 4)
}

fn weight_k(gpu: &MockGpuBackend, k: usize, n: usize, k_bits: u32) -> Exl3DenseWeight {
    Exl3DenseWeight {
        trellis: gpu
            .alloc((k / 16) * (n / 16) * 16 * k_bits as usize * 2)
            .unwrap(),
        suh: gpu.alloc(k * 2).unwrap(),
        svh: gpu.alloc(n * 2).unwrap(),
        in_dim: k,
        out_dim: n,
        k_bits,
        cb: 2,
    }
}

fn kernel_names(gpu: &MockGpuBackend) -> Vec<String> {
    gpu.kernel_lookups_snapshot()
        .into_iter()
        .map(|(_, f)| f)
        .collect()
}

fn contiguous(dst: DevicePtr) -> Exl3DenseOut {
    Exl3DenseOut::contiguous(dst)
}

#[test]
fn launch_plan_gemv_tier_and_row_batched_gemm_tier() {
    // The mock records kernel resolutions, which IS the launch plan:
    // ingress / matmul / egress per tier, and the row batching above the
    // stage capacity (numerics are the GPU parity example's job).
    let gpu = MockGpuBackend::new();
    let launch = std::sync::Arc::new(Exl3LaunchState::new(&gpu).unwrap());
    // 256-row stage (the env override would break the batching
    // assertion — the test never sets it).
    let stage = Exl3DenseStage::new(&gpu, launch, 256, 6144, 12288).unwrap();
    let w = weight(&gpu, 2560, 10240);
    let a = gpu.alloc(700 * 2560 * 2).unwrap();
    let dst = gpu.alloc(700 * 10240 * 2).unwrap();

    let before = gpu.kernel_lookups_snapshot().len();
    exl3_dense_linear(&gpu, &w, a, contiguous(dst), 1, &stage, 0).unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    assert_eq!(
        names,
        vec![
            "exl3_bf16_to_f16",
            "exl3_gemv_k4_cb2_m0_cfg1_f32",
            "exl3_f32_to_bf16"
        ],
        "m=1 must take the f32-C GEMV tier"
    );

    let before = gpu.kernel_lookups_snapshot().len();
    exl3_dense_linear(&gpu, &w, a, contiguous(dst), 700, &stage, 0).unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    // 3 batches (256 + 256 + 188) x (ingress, gemm, in-place convert).
    assert_eq!(names.len(), 9, "{names:?}");
    for b in 0..3 {
        assert_eq!(names[b * 3], "exl3_bf16_to_f16");
        assert!(names[b * 3 + 1].starts_with("exl3_gemm_k4_cb2_sh"));
        assert!(names[b * 3 + 1].ends_with("_f16"));
        assert_eq!(names[b * 3 + 2], "exl3_f16_to_bf16");
    }

    // Strided destination: staged f16 C + the 2-D egress; a shared-A
    // pair ingresses ONCE per row batch.
    let z = weight(&gpu, 2560, 6144);
    let arena = gpu.alloc(700 * 16384 * 2).unwrap();
    let before = gpu.kernel_lookups_snapshot().len();
    exl3_dense_linear_shared_a(
        &gpu,
        &[
            (w, Exl3DenseOut::strided(arena, 16384)),
            (z, Exl3DenseOut::strided(arena.offset(10240 * 2), 16384)),
        ],
        a,
        300,
        &stage,
        0,
    )
    .unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    assert_eq!(
        names.iter().filter(|n| *n == "exl3_bf16_to_f16").count(),
        2,
        "one ingress per row batch: {names:?}"
    );
    assert_eq!(
        names.iter().filter(|n| *n == "exl3_f16_to_bf16_2d").count(),
        4,
        "one strided egress per (batch, weight): {names:?}"
    );
    assert!(!names.iter().any(|n| n == "exl3_f16_to_bf16"));
}

#[test]
fn launch_plan_k6_skips_the_gemv_tier_at_small_m() {
    // K in {5,6,8} has no GEMV instances: the m<=8 tier goes straight to the
    // f32-C GEMM's BF16-ingress twin (`_f32_abf16` — the bf16->f16 convert is
    // fused into its input-Hadamard prologue, so NO separate ingress launch
    // and never a k6 gemv name), then egress from the same f32 slab. m>8 is
    // the ordinary GEMM tier.
    let gpu = MockGpuBackend::new();
    let launch = std::sync::Arc::new(Exl3LaunchState::new(&gpu).unwrap());
    let stage = Exl3DenseStage::new(&gpu, launch, 256, 6144, 12288).unwrap();
    let a = gpu.alloc(64 * 2560 * 2).unwrap();
    let dst = gpu.alloc(64 * 10240 * 2).unwrap();
    for k_bits in [5u32, 6, 8] {
        let w = weight_k(&gpu, 2560, 10240, k_bits);
        for m in [1usize, 8] {
            let before = gpu.kernel_lookups_snapshot().len();
            exl3_dense_linear(&gpu, &w, a, contiguous(dst), m, &stage, 0).unwrap();
            let names = kernel_names(&gpu)[before..].to_vec();
            assert_eq!(names.len(), 2, "K={k_bits} m={m}: {names:?}");
            assert!(
                names[0].starts_with(&format!("exl3_gemm_k{k_bits}_cb2_sh"))
                    && names[0].ends_with("_f32_abf16"),
                "K={k_bits} m={m}: {names:?}"
            );
            assert_eq!(names[1], "exl3_f32_to_bf16");
            assert!(
                !names
                    .iter()
                    .any(|n| n.contains("gemv") || n == "exl3_bf16_to_f16")
            );
        }
        let before = gpu.kernel_lookups_snapshot().len();
        exl3_dense_linear(&gpu, &w, a, contiguous(dst), 64, &stage, 0).unwrap();
        let names = kernel_names(&gpu)[before..].to_vec();
        assert_eq!(names.len(), 3, "K={k_bits} m=64: {names:?}");
        assert!(names[1].starts_with(&format!("exl3_gemm_k{k_bits}_cb2_sh")));
        assert!(names[1].ends_with("_f16"));
    }
    // K=3 HAS gemv instances: the tier is attempted (the heuristic may
    // still decline for some shapes; [2560->10240] accepts the wide cfg).
    let w3 = weight_k(&gpu, 2560, 10240, 3);
    let before = gpu.kernel_lookups_snapshot().len();
    exl3_dense_linear(&gpu, &w3, a, contiguous(dst), 1, &stage, 0).unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    assert!(
        names[1].starts_with("exl3_gemv_k3_cb2_m0_cfg") || names[1].starts_with("exl3_gemm_k3_"),
        "{names:?}"
    );
}

#[test]
fn contract_checks_refuse_bad_geometry() {
    let gpu = MockGpuBackend::new();
    let launch = std::sync::Arc::new(Exl3LaunchState::new(&gpu).unwrap());
    let stage = Exl3DenseStage::new(&gpu, launch, 64, 2560, 4096).unwrap();
    let a = gpu.alloc(8 * 6144 * 2).unwrap();
    let dst = gpu.alloc(8 * 12288 * 2).unwrap();
    // in_dim above the stage's max_in.
    let wide_in = weight(&gpu, 6144, 2560);
    assert!(exl3_dense_linear(&gpu, &wide_in, a, contiguous(dst), 1, &stage, 0).is_err());
    // out_dim above the stage's max_out.
    let wide_out = weight(&gpu, 2560, 12288);
    assert!(exl3_dense_linear(&gpu, &wide_out, a, contiguous(dst), 1, &stage, 0).is_err());
    // Row stride narrower than out_dim.
    let w = weight(&gpu, 2560, 2560);
    let narrow = Exl3DenseOut::strided(dst, 2048);
    assert!(exl3_dense_linear(&gpu, &w, a, narrow, 1, &stage, 0).is_err());
    // Shared-A weights must agree on in_dim.
    let other = weight(&gpu, 1280, 2560);
    let pair = [(w, contiguous(dst)), (other, contiguous(dst))];
    assert!(exl3_dense_linear_shared_a(&gpu, &pair, a, 1, &stage, 0).is_err());
    // m == 0, then the happy path; a refused call must have left no
    // stale section claim behind.
    assert!(exl3_dense_linear(&gpu, &w, a, contiguous(dst), 0, &stage, 0).is_err());
    exl3_dense_linear(&gpu, &w, a, contiguous(dst), 8, &stage, 0).unwrap();
    exl3_dense_linear(&gpu, &w, a, contiguous(dst), 1, &stage, 0).unwrap();
}

#[test]
fn from_exl3_maps_codebook_and_rejects_envelope_breaches() {
    let base = Exl3Weight {
        trellis: DevicePtr(0x1000),
        suh: DevicePtr(0x2000),
        svh: DevicePtr(0x3000),
        in_dim: 2560,
        out_dim: 10240,
        k_bits: 4,
        cb: Exl3Codebook::Mul1,
    };
    assert_eq!(Exl3DenseWeight::from_exl3(&base).unwrap().cb, 2);
    let mcg = Exl3Weight {
        cb: Exl3Codebook::Mcg,
        ..base
    };
    assert_eq!(Exl3DenseWeight::try_from(&mcg).unwrap().cb, 1);
    let inst3 = Exl3Weight {
        cb: Exl3Codebook::Inst3,
        ..base
    };
    assert!(Exl3DenseWeight::from_exl3(&inst3).is_err());
    // Every K with gemm instances is admitted (K>4 skips the GEMV tier);
    // K=1/7 have no instances at all.
    for k in [2u32, 3, 4, 5, 6, 8] {
        assert!(Exl3DenseWeight::from_exl3(&Exl3Weight { k_bits: k, ..base }).is_ok());
    }
    for k in [1u32, 7] {
        assert!(Exl3DenseWeight::from_exl3(&Exl3Weight { k_bits: k, ..base }).is_err());
    }
    let odd = Exl3Weight {
        out_dim: 10200,
        ..base
    };
    assert!(Exl3DenseWeight::from_exl3(&odd).is_err());
}
