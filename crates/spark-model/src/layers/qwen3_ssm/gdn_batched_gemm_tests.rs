// SPDX-License-Identifier: AGPL-3.0-only

//! Mock-backend launch tests for the BF16-kept GDN verify projections
//! (`ops::gdn_batched_gemm`), driven through the real
//! `decode_verify_multi` -> `decode_batched_inner` body.
//!
//! The mock records every launch with its arguments, so a projection is
//! identified by the weight buffer it reads. These tests pin the dispatch
//! with the row at its declared default (off on every target): they are the
//! code-level check of the per-row re-read at 3 and 4 rows, and the guard
//! that the row's default changes nothing. The row-on arm calls cuBLASLt,
//! which the mock cannot stand in for, so its choice is pinned by the pure
//! dispatch tests in `ops/gdn_batched_gemm_tests.rs`.

use super::tests::{native_fp8_gdn_layer, run_batched_verify};
use super::*;
use crate::layers::ops::gdn_batched_gemm::{self as row, OutProjArm, QkvzArm};
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

/// A Qwen3.5-dense `Bf16Raw`-shaped GDN layer: `in_proj_qkvz` and
/// `out_proj_dense` are the only live projection weights.
fn bf16_gdn_layer(gpu: &MockGpuBackend, config: &ModelConfig) -> Qwen3SsmLayer {
    let mut layer = native_fp8_gdn_layer(gpu, config, false, false);
    let h = config.hidden_size;
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    layer.ssm.in_proj_qkvz = DenseWeight {
        weight: gpu.alloc(config.ssm_qkvz_size() * h * 2).unwrap(),
    };
    layer.out_proj_dense = Some(DenseWeight {
        weight: gpu.alloc(h * value_dim * 2).unwrap(),
    });
    layer
}

/// Launches that read `weight`.
fn launches_reading(gpu: &MockGpuBackend, weight: DevicePtr) -> usize {
    gpu.launches_snapshot()
        .iter()
        .filter(|l| l.args.contains(&MockArg::Buffer(weight)))
        .count()
}

/// One single-sequence verify of `rows` rows; returns the QKVZ and out_proj
/// launch counts.
fn verify_counts(rows: usize) -> (usize, usize) {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = bf16_gdn_layer(&gpu, &config);
    run_batched_verify(&gpu, &config, &layer, &[rows]).unwrap();
    let out = layer.out_proj_dense.as_ref().unwrap().weight;
    (
        launches_reading(&gpu, layer.ssm.in_proj_qkvz.weight),
        launches_reading(&gpu, out),
    )
}

fn expected_qkvz_launches(rows: usize) -> usize {
    match row::qkvz_arm(row::row(), rows) {
        QkvzArm::GemvPerRow => rows,
        QkvzArm::GemvBatch2 | QkvzArm::DenseGemm => 1,
        // cuBLASLt is not a mock launch.
        QkvzArm::Cublas => 0,
    }
}

fn expected_out_proj_launches(rows: usize) -> usize {
    match row::out_proj_arm(row::row(), rows) {
        OutProjArm::DenseGemm => 1,
        OutProjArm::Cublas => 0,
    }
}

/// The table in `ops/gdn_batched_gemm.rs`, at the shipped default.
#[test]
fn bf16_gdn_verify_launches_per_row_count() {
    if std::env::var_os(row::ENV).is_some() {
        return; // an operator override changes the arm under test
    }
    assert!(
        !row::row(),
        "every target declares gdn_batched_gemm = false"
    );
    for rows in [1usize, 2, 3, 4, 8] {
        let (qkvz, out) = verify_counts(rows);
        assert_eq!(qkvz, expected_qkvz_launches(rows), "qkvz at {rows} rows");
        assert_eq!(
            out,
            expected_out_proj_launches(rows),
            "out_proj at {rows} rows"
        );
    }
}

/// The per-row re-read the row exists for: four weight passes at four rows
/// with the row off, against one at two rows.
#[test]
fn bf16_gdn_verify_rereads_qkvz_per_row_at_four_rows() {
    if std::env::var_os(row::ENV).is_some() {
        return;
    }
    assert_eq!(verify_counts(4).0, 4);
    assert_eq!(verify_counts(2).0, 1);
}
