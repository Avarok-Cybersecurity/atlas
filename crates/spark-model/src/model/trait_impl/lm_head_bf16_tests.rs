// SPDX-License-Identifier: AGPL-3.0-only

//! Exercise the BF16 projection used by both ordinary and mixed decode.

use super::{batchm_max_from_value, bf16_batch_gemv_from_value, project_bf16_lm_head};
use crate::layers::ops;
use crate::weight_map::DenseWeight;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

fn run_case(m: u32, k: u32, present: bool, enabled: bool, expect_batch: bool) {
    run_band(
        m,
        k,
        present,
        enabled,
        ops::DENSE_GEMV_BATCHM_DECODE_MAX_M,
        expect_batch,
    )
}

fn run_band(m: u32, k: u32, present: bool, enabled: bool, batchm_max: u32, expect_batch: bool) {
    let gpu = MockGpuBackend::new();
    let n = 67_u32;
    let input = gpu.alloc((m * k * 2) as usize).unwrap();
    let weight = DenseWeight {
        weight: gpu.alloc((n * k * 2) as usize).unwrap(),
    };
    let output = gpu.alloc((m * n * 2) as usize).unwrap();
    let allocated = gpu.alloc_count();
    let batch = KernelHandle(if present { 0xBF16 } else { 0 });
    project_bf16_lm_head(
        &gpu,
        KernelHandle(0xCAFE),
        batch,
        input,
        &weight,
        output,
        [m, n, k],
        enabled,
        batchm_max,
        7,
    )
    .unwrap();
    assert_eq!(
        gpu.alloc_count(),
        allocated,
        "the checkpoint weight must not be copied or quantized"
    );
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    let launch = &launches[0];
    assert_eq!(
        launch.func,
        if expect_batch { batch.0 } else { 0xCAFE },
        "M={m}: BF16 head selected the wrong kernel"
    );
    assert_eq!(launch.stream, 7);
    assert_eq!(
        &launch.args[..3],
        &[
            MockArg::Buffer(input),
            MockArg::Buffer(weight.weight),
            MockArg::Buffer(output)
        ]
    );
    let mut sizes = vec![m, n, k];
    if expect_batch {
        sizes.push(n);
    }
    assert_eq!(
        launch.args[3..],
        sizes
            .iter()
            .map(|value| MockArg::Bytes(value.to_ne_bytes().to_vec()))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        launch.grid,
        if expect_batch {
            [n.div_ceil(4), 1, 1]
        } else {
            [n.div_ceil(16), m.div_ceil(16), 1]
        }
    );
    assert_eq!(
        launch.block,
        if expect_batch {
            [256, 1, 1]
        } else {
            [16, 16, 1]
        }
    );
}

#[test]
fn default_small_bf16_head_uses_existing_batch_gemv() {
    for m in [1, 2, 4, 8] {
        run_case(m, 128, true, bf16_batch_gemv_from_value(None), true);
    }
}

#[test]
fn opt_out_missing_kernel_and_wide_head_keep_scalar_fallback() {
    run_case(4, 128, true, bf16_batch_gemv_from_value(Some("0")), false);
    run_case(4, 128, false, true, false);
    // Existing uint4 loads need 16-byte alignment at every input/weight row.
    run_case(4, 130, true, true, false);
    for m in [9, 16] {
        run_case(m, 128, true, true, false);
    }
}

/// The `ATLAS_LM_HEAD_BATCHM_MAX` lever (#927). DEFAULT IS UNCHANGED: widths
/// 9..=16 keep the reassociating tile GEMM they have always used on GB10, so
/// the frozen band's bits are untouched unless an operator asks.
#[test]
fn lm_head_band_defaults_to_the_frozen_decode_edge() {
    assert_eq!(
        batchm_max_from_value(None),
        ops::DENSE_GEMV_BATCHM_DECODE_MAX_M
    );
    for m in [9, 12, 16] {
        run_case(m, 128, true, true, false);
    }
}

/// The H100 recipe's value. 9..=16 move onto the batched GEMV; 17 is still
/// outside the kernel's compile-time row bound and must not.
#[test]
fn lm_head_band_widened_to_sixteen_claims_nine_through_sixteen() {
    assert_eq!(batchm_max_from_value(Some("16")), 16);
    for m in [1, 8, 9, 12, 16] {
        run_band(m, 128, true, true, 16, true);
    }
    run_band(17, 128, true, true, 16, false);
}

/// A lever above the kernel's `MAX_M` is CLAMPED, not honoured: the kernel
/// refuses wider launches, and an Err every decode step is worse than
/// ignoring the excess. Junk and 0 keep the default — the value is a band,
/// not a switch, so there is no "off".
#[test]
fn lm_head_band_lever_is_clamped_and_defaults_on_junk() {
    assert_eq!(
        batchm_max_from_value(Some("64")),
        ops::DENSE_GEMV_BATCHM_MAX_M
    );
    assert_eq!(batchm_max_from_value(Some(" 12 ")), 12);
    for value in [Some("0"), Some(""), Some("sixteen"), Some("-4"), None] {
        assert_eq!(
            batchm_max_from_value(value),
            ops::DENSE_GEMV_BATCHM_DECODE_MAX_M,
            "value {value:?} must keep the frozen default"
        );
    }
}

#[test]
fn legacy_opt_out_value_is_preserved() {
    assert!(!bf16_batch_gemv_from_value(Some("0")));
    for value in [None, Some("1"), Some(""), Some("false"), Some(" 0 ")] {
        assert!(bf16_batch_gemv_from_value(value));
    }
}
