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
    // 256-row stage with the reconstruct tier pinned OFF (its default
    // threshold of 512 would take the 700-row call below; this test is the
    // trellis tier's batching plan — the env is never read here).
    let stage =
        Exl3DenseStage::new_with_reconstruct(&gpu, launch, 256, 6144, 12288, 12288, None).unwrap();
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
    // K in {5,6,8} has no GEMV instances: the m<=8 tier is ONE launch of the
    // f32-C GEMM's BF16-in/BF16-out twin (`_f32_abf16_obf16` — bf16->f16
    // convert fused into the input-Hadamard prologue, f32->bf16 egress fused
    // into the output-Hadamard epilogue, so no converter launch on either
    // side and never a k6 gemv name). m>8 is the ordinary GEMM tier.
    let gpu = MockGpuBackend::new();
    let launch = std::sync::Arc::new(Exl3LaunchState::new(&gpu).unwrap());
    let stage = Exl3DenseStage::new(&gpu, launch, 256, 6144, 12288).unwrap();
    let a = gpu.alloc(64 * 2560 * 2).unwrap();
    let dst = gpu.alloc(64 * 10240 * 2).unwrap();
    for k_bits in [5u32, 6, 8] {
        let w = weight_k(&gpu, 2560, 10240, k_bits);
        for m in [1usize, 8] {
            let before = gpu.kernel_lookups_snapshot().len();
            let ws = [(w, contiguous(dst))];
            dense_linear_shared_a(&gpu, &ws, a, m, &stage, 0, true).unwrap();
            let names = kernel_names(&gpu)[before..].to_vec();
            assert_eq!(names.len(), 1, "K={k_bits} m={m}: {names:?}");
            assert!(
                names[0].starts_with(&format!("exl3_gemm_k{k_bits}_cb2_sh"))
                    && names[0].ends_with("_f32_abf16_obf16"),
                "K={k_bits} m={m}: {names:?}"
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
fn launch_plan_fused_egress_kill_switch_restores_the_converter() {
    // `ATLAS_EXL3_NO_FUSED_EGRESS` (the A/B control arm): the `_f32_abf16`
    // twin followed by the standalone f32 egress — `exl3_f32_to_bf16` for a
    // contiguous destination, `_2d` for a pitched one. The fused arm's plan
    // for the same calls is a single `_abf16_obf16` launch each, with the
    // pitch carried as a kernel argument. Each shared-A weight is one launch
    // in both arms; the ingress launch never appears at K=6.
    let gpu = MockGpuBackend::new();
    let launch = std::sync::Arc::new(Exl3LaunchState::new(&gpu).unwrap());
    let stage = Exl3DenseStage::new(&gpu, launch, 256, 6144, 12288).unwrap();
    let a = gpu.alloc(8 * 2560 * 2).unwrap();
    let arena = gpu.alloc(8 * 16384 * 2).unwrap();
    let qkv = weight_k(&gpu, 2560, 10240, 6);
    let z = weight_k(&gpu, 2560, 6144, 6);
    let ws = [
        (qkv, Exl3DenseOut::strided(arena, 16384)),
        (z, Exl3DenseOut::strided(arena.offset(10240 * 2), 16384)),
    ];
    for m in [1usize, 3, 8] {
        let before = gpu.kernel_lookups_snapshot().len();
        dense_linear_shared_a(&gpu, &ws, a, m, &stage, 0, false).unwrap();
        let names = kernel_names(&gpu)[before..].to_vec();
        assert_eq!(names.len(), 4, "control m={m}: {names:?}");
        for (i, sh) in [(0usize, "sh3"), (2, "sh3")] {
            assert_eq!(
                names[i],
                format!("exl3_gemm_k6_cb2_{sh}_f32_abf16"),
                "{names:?}"
            );
            assert_eq!(names[i + 1], "exl3_f32_to_bf16_2d", "{names:?}");
        }

        let before = gpu.kernel_lookups_snapshot().len();
        dense_linear_shared_a(&gpu, &ws, a, m, &stage, 0, true).unwrap();
        let names = kernel_names(&gpu)[before..].to_vec();
        assert_eq!(
            names,
            vec![
                "exl3_gemm_k6_cb2_sh3_f32_abf16_obf16",
                "exl3_gemm_k6_cb2_sh3_f32_abf16_obf16"
            ],
            "fused m={m}"
        );
    }
    // Contiguous destination, control arm: the 1-D converter.
    let dst = gpu.alloc(8 * 10240 * 2).unwrap();
    let before = gpu.kernel_lookups_snapshot().len();
    dense_linear_shared_a(&gpu, &[(qkv, contiguous(dst))], a, 1, &stage, 0, false).unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    assert_eq!(
        names,
        vec!["exl3_gemm_k6_cb2_sh3_f32_abf16", "exl3_f32_to_bf16"]
    );
    // The GEMV tier (K=4) is untouched by the switch: same plan both ways.
    let w4 = weight(&gpu, 2560, 10240);
    let mut plans = Vec::new();
    for fused in [false, true] {
        let before = gpu.kernel_lookups_snapshot().len();
        dense_linear_shared_a(&gpu, &[(w4, contiguous(dst))], a, 1, &stage, 0, fused).unwrap();
        plans.push(kernel_names(&gpu)[before..].to_vec());
    }
    assert_eq!(plans[0], plans[1]);
    assert_eq!(plans[0][0], "exl3_bf16_to_f16");
    assert_eq!(plans[0][2], "exl3_f32_to_bf16");
    // The public entry reads the switch: no env in the test process, so it
    // takes the fused plan.
    let before = gpu.kernel_lookups_snapshot().len();
    exl3_dense_linear(&gpu, &qkv, a, contiguous(dst), 1, &stage, 0).unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    assert_eq!(
        names,
        vec!["exl3_gemm_k6_cb2_sh3_f32_abf16_obf16"],
        "default plan must be the fused one (is ATLAS_EXL3_NO_FUSED_EGRESS set?)"
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

// ── reconstruct-to-BF16 prefill tier (`exl3_dense/reconstruct.rs`) ────────

#[test]
fn reconstruct_rows_env_parsing_defaults_to_512_and_the_kill_switch_wins() {
    // Unset = the measured default threshold (2026-09-06 A/B + agentic gate).
    assert_eq!(
        parse_reconstruct_rows(None, false),
        Some(EXL3_DENSE_RECONSTRUCT_DEFAULT_ROWS)
    );
    assert_eq!(EXL3_DENSE_RECONSTRUCT_DEFAULT_ROWS, 512);
    // A row count arms at that count.
    assert_eq!(parse_reconstruct_rows(Some("512"), false), Some(512));
    assert_eq!(parse_reconstruct_rows(Some(" 1024 "), false), Some(1024));
    // Presence arms: `=0`, a value inside the decode tier, or garbage all
    // clamp to the minimum — they never mean "off".
    for v in ["0", "1", "8", "abc", ""] {
        assert_eq!(
            parse_reconstruct_rows(Some(v), false),
            Some(EXL3_DENSE_RECONSTRUCT_MIN_ROWS),
            "{v:?}"
        );
    }
    assert_eq!(EXL3_DENSE_RECONSTRUCT_MIN_ROWS, EXL3_GEMV_MAX_M + 1);
    // The kill switch wins over any threshold, including the default.
    assert_eq!(parse_reconstruct_rows(Some("512"), true), None);
    assert_eq!(parse_reconstruct_rows(None, true), None);
}

#[test]
fn reconstruct_tier_decision_never_reaches_the_decode_arm() {
    // Off: never.
    for m in [1usize, 8, 9, 512, 8192] {
        assert!(!reconstruct_tier_takes(m, None), "m={m}");
    }
    // Armed at 512: m >= 512 only.
    assert!(!reconstruct_tier_takes(511, Some(512)));
    assert!(reconstruct_tier_takes(512, Some(512)));
    assert!(reconstruct_tier_takes(8192, Some(512)));
    // Even a threshold inside the decode tier cannot pull m <= 8 in.
    for m in 1..=EXL3_GEMV_MAX_M {
        assert!(!reconstruct_tier_takes(m, Some(1)), "m={m}");
    }
    assert!(reconstruct_tier_takes(EXL3_GEMV_MAX_M + 1, Some(1)));
}

#[test]
fn reconstruct_scratch_is_sized_from_the_stage_maxima() {
    // qwen4_exp maxima: 6144 x 12288 x 2 B = 151 MB per slab (f16 [in, out]
    // + bf16 [out, in]), the number the design quotes.
    let (f16, bf16) = reconstruct_scratch_bytes(6144, 12288);
    assert_eq!(f16, 6144 * 12288 * 2);
    assert_eq!(bf16, f16);
    assert_eq!(f16, 150_994_944);
    let gpu = MockGpuBackend::new();
    let launch = std::sync::Arc::new(Exl3LaunchState::new(&gpu).unwrap());
    // Off (None): no scratch, no GEMM kernel probe.
    let off = Exl3DenseStage::new_with_reconstruct(&gpu, launch.clone(), 256, 2560, 10240, 0, None)
        .unwrap();
    assert!(off.recon.is_none());
    assert!(
        !kernel_names(&gpu)
            .iter()
            .any(|n| n == "dense_gemm_bf16_pipelined")
    );
    off.release(&gpu).unwrap();
    // Armed: both slabs at max_in x max_out, the GEMM probed at construction.
    let on =
        Exl3DenseStage::new_with_reconstruct(&gpu, launch, 256, 2560, 10240, 0, Some(64)).unwrap();
    let rs = on.recon.as_ref().unwrap();
    assert_eq!(rs.elems, 2560 * 10240);
    assert_eq!(rs.threshold, 64);
    assert!(rs.takes(64) && !rs.takes(63) && !rs.takes(8));
    assert!(
        kernel_names(&gpu)
            .iter()
            .any(|n| n == "dense_gemm_bf16_pipelined")
    );
    on.release(&gpu).unwrap();
    // A threshold inside the decode tier is refused outright.
    assert!(Exl3ReconScratch::new(&gpu, 2560, 10240, EXL3_GEMV_MAX_M).is_err());
}

#[test]
fn launch_plan_reconstruct_tier_above_threshold_only() {
    let gpu = MockGpuBackend::new();
    let launch = std::sync::Arc::new(Exl3LaunchState::new(&gpu).unwrap());
    // fp32-C capacity for the widest weight so the `with_fp32` leg below
    // passes the stage contract (the layers only pin out = hidden to fp32).
    let stage =
        Exl3DenseStage::new_with_reconstruct(&gpu, launch, 256, 2560, 10240, 10240, Some(64))
            .unwrap();
    let w = weight_k(&gpu, 2560, 10240, 6);
    let a = gpu.alloc(700 * 2560 * 2).unwrap();
    let dst = gpu.alloc(700 * 10240 * 2).unwrap();

    // m <= 8: the decode arm, byte-for-byte the plan it had before — one
    // fused-ingress/fused-egress GEMM launch (the eighth lever).
    let before = gpu.kernel_lookups_snapshot().len();
    exl3_dense_linear(&gpu, &w, a, contiguous(dst), 8, &stage, 0).unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    assert_eq!(names.len(), 1, "{names:?}");
    assert!(
        names[0].starts_with("exl3_gemm_k6_cb2_sh") && names[0].ends_with("_f32_abf16_obf16"),
        "{names:?}"
    );

    // 8 < m < threshold: the trellis GEMM tier, unchanged.
    let before = gpu.kernel_lookups_snapshot().len();
    exl3_dense_linear(&gpu, &w, a, contiguous(dst), 63, &stage, 0).unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    assert_eq!(names.len(), 3, "{names:?}");
    assert!(names[1].starts_with("exl3_gemm_k6_cb2_sh") && names[1].ends_with("_f16"));

    // m >= threshold, contiguous: reconstruct, transpose, ONE GEMM straight
    // into dst — no converter, no trellis GEMM, even above rows_cap.
    let d2d_before = gpu.d2d_2d_count();
    let before = gpu.kernel_lookups_snapshot().len();
    exl3_dense_linear(&gpu, &w, a, contiguous(dst), 700, &stage, 0).unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    assert_eq!(
        names,
        vec![
            "exl3_reconstruct_had_k6_cb2",
            "exl3_f16_to_bf16_t",
            "dense_gemm_bf16_pipelined",
        ]
    );
    assert_eq!(
        gpu.d2d_2d_count(),
        d2d_before,
        "contiguous dst needs no copy"
    );

    // fp32-C destinations take the same plan (fp32 accumulate + one BF16
    // round is what the GEMM does anyway).
    let before = gpu.kernel_lookups_snapshot().len();
    exl3_dense_linear(&gpu, &w, a, contiguous(dst).with_fp32(), 300, &stage, 0).unwrap();
    let names = kernel_names(&gpu)[before..].to_vec();
    assert_eq!(names.len(), 3, "{names:?}");
    assert_eq!(names[2], "dense_gemm_bf16_pipelined");
    assert!(!names.iter().any(|n| n.contains("f32_to_bf16")));

    // Strided shared-A pair at m=300 over a 256-row stage: per weight ONE
    // reconstruct + transpose, then a GEMM + 2-D copy per row batch (2).
    let z = weight_k(&gpu, 2560, 6144, 6);
    let arena = gpu.alloc(300 * 16384 * 2).unwrap();
    let d2d_before = gpu.d2d_2d_count();
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
        names,
        vec![
            "exl3_reconstruct_had_k6_cb2",
            "exl3_f16_to_bf16_t",
            "dense_gemm_bf16_pipelined",
            "dense_gemm_bf16_pipelined",
            "exl3_reconstruct_had_k6_cb2",
            "exl3_f16_to_bf16_t",
            "dense_gemm_bf16_pipelined",
            "dense_gemm_bf16_pipelined",
        ]
    );
    assert_eq!(
        gpu.d2d_2d_count() - d2d_before,
        4,
        "one 2-D copy per (batch, weight)"
    );
    assert!(!names.iter().any(|n| n == "exl3_bf16_to_f16"));
    stage.release(&gpu).unwrap();
}
