// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `exl3_dense.rs` (child module; split out for the ≤500 LoC
//! cap — the mock records kernel resolutions, which IS the launch plan).

use spark_runtime::gpu::mock::MockGpuBackend;

use super::*;

fn weight(gpu: &MockGpuBackend, k: usize, n: usize) -> Exl3DenseWeight {
    Exl3DenseWeight {
        trellis: gpu.alloc((k / 16) * (n / 16) * 16 * 4 * 2).unwrap(),
        suh: gpu.alloc(k * 2).unwrap(),
        svh: gpu.alloc(n * 2).unwrap(),
        in_dim: k,
        out_dim: n,
        k_bits: 4,
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
    assert!(Exl3DenseWeight::from_exl3(&Exl3Weight { k_bits: 6, ..base }).is_err());
    let odd = Exl3Weight {
        out_dim: 10200,
        ..base
    };
    assert!(Exl3DenseWeight::from_exl3(&odd).is_err());
}
