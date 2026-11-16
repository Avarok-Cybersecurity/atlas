// SPDX-License-Identifier: AGPL-3.0-only

//! Pure dispatch tests for the `gdn_batched_gemm` row. The mock-backend
//! launch tests live beside the layer (`qwen3_ssm/gdn_batched_gemm_tests.rs`).

use super::*;
use crate::layers::ops::target_defaults::Source;

#[test]
fn row_off_keeps_every_pre_row_arm() {
    assert_eq!(qkvz_arm(false, 1), QkvzArm::DenseGemm);
    assert_eq!(qkvz_arm(false, 2), QkvzArm::GemvBatch2);
    assert_eq!(qkvz_arm(false, 3), QkvzArm::GemvPerRow);
    assert_eq!(qkvz_arm(false, 4), QkvzArm::GemvPerRow);
    for rows in [5, 8, 16, 32] {
        assert_eq!(qkvz_arm(false, rows), QkvzArm::DenseGemm, "rows={rows}");
    }
    for rows in [1, 2, 4, 8, 32] {
        assert_eq!(out_proj_arm(false, rows), OutProjArm::DenseGemm);
    }
}

#[test]
fn row_on_moves_every_multi_pass_or_scalar_arm_to_one_gemm() {
    // One row is never a batched shape; two rows keep the one-pass GEMV that
    // is bit-identical to decode.
    assert_eq!(qkvz_arm(true, 1), QkvzArm::DenseGemm);
    assert_eq!(qkvz_arm(true, 2), QkvzArm::GemvBatch2);
    for rows in [3, 4, 8, 16, 32] {
        assert_eq!(qkvz_arm(true, rows), QkvzArm::Cublas, "rows={rows}");
    }
    assert_eq!(out_proj_arm(true, 1), OutProjArm::DenseGemm);
    for rows in [2, 4, 8, 32] {
        assert_eq!(out_proj_arm(true, rows), OutProjArm::Cublas, "rows={rows}");
    }
}

#[test]
fn resolution_follows_the_toggle_grammar() {
    assert!(!resolve(false, None).value);
    assert_eq!(resolve(false, None).source, Source::Target);
    assert!(resolve(false, Some("1")).value);
    assert_eq!(resolve(false, Some("1")).source, Source::Env);
    for off in ["0", "false", "off", "no"] {
        assert!(!resolve(true, Some(off)).value, "{off}");
    }
    assert!(resolve(true, None).value);
}
