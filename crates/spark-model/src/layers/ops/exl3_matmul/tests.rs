// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the EXL3 GEMM shape heuristic and the locks-buffer sizing
//! (`ops/exl3_matmul.rs`); split from the parent on the 500-LoC cap.

use super::*;

#[test]
fn blackwell_shape_heuristic_matches_upstream_branch() {
    // Shape 1: K in {2,4}, !multi, k <= 2048.
    assert_eq!(select_exl3_gemm_shape(2048, 640, 4, false, 1, 1), 1);
    assert_eq!(select_exl3_gemm_shape(2048, 640, 2, false, 1, 1), 1);
    // multi never takes shape 1.
    assert_ne!(select_exl3_gemm_shape(2048, 640, 4, true, 1, 1), 1);
    // K=3 small-n mod-256: shape 2 unless k > 8192.
    assert_eq!(select_exl3_gemm_shape(2560, 1024, 3, false, 1, 1), 2);
    assert_eq!(select_exl3_gemm_shape(9216, 1024, 3, false, 1, 1), 3);
    // Wide n, mod-512: shape 4.
    assert_eq!(select_exl3_gemm_shape(2560, 248320, 4, true, 1, 1), 4);
    // qwen4_exp lm_head-ish n % 256 != 0 -> universal shape 2 fallback.
    assert_eq!(select_exl3_gemm_shape(2560, 640, 5, false, 1, 1), 2);
    // bszm scaling: mod_256 uses UNSCALED n, comparisons use scaled.
    assert_eq!(select_exl3_gemm_shape(2560, 512, 6, true, 1, 64), 4);
}

#[test]
fn shape_compat_gates_divisibility() {
    assert!(exl3_gemm_shape_compat(2, 2560, 640));
    assert!(!exl3_gemm_shape_compat(3, 2560, 640)); // 640 % 256 != 0
    assert!(!exl3_gemm_shape_compat(2, 2504, 640)); // k % 32 != 0
    assert!(exl3_gemm_shape_compat(4, 2560, 1024));
}

#[test]
fn locks_sizing_matches_upstream_devctx() {
    assert_eq!(EXL3_LOCKS_BYTES, 4_202_760);
}
