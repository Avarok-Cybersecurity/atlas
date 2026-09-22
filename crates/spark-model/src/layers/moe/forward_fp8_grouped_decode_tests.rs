// SPDX-License-Identifier: AGPL-3.0-only
//
// CPU tests for the pure admission helpers of the grouped FP8 MoE decode path.

use super::{FP8_GROUPED_DECODE_MAX_ROWS, fp8_grouped_decode_shape_ok, grouped_decode_buffer_need};

/// Qwen3.6-35B-A3B-FP8: hidden 2048, moe_intermediate 512, 256 experts, top-8.
const H: u32 = 2048;
const INTER: u32 = 512;

#[test]
fn row_envelope_is_2_to_max() {
    assert!(!fp8_grouped_decode_shape_ok(0, H, INTER));
    assert!(
        !fp8_grouped_decode_shape_ok(1, H, INTER),
        "M=1 is the single-token path"
    );
    for m in [2usize, 3, 4, 8, 16, 24, 32, FP8_GROUPED_DECODE_MAX_ROWS] {
        assert!(fp8_grouped_decode_shape_ok(m, H, INTER), "m={m}");
    }
    assert!(!fp8_grouped_decode_shape_ok(
        FP8_GROUPED_DECODE_MAX_ROWS + 1,
        H,
        INTER
    ));
}

#[test]
fn hidden_must_be_a_multiple_of_16_for_the_uint4_pair_loads() {
    assert!(fp8_grouped_decode_shape_ok(4, 16, INTER));
    assert!(!fp8_grouped_decode_shape_ok(4, 2040, INTER));
    assert!(!fp8_grouped_decode_shape_ok(4, 8, INTER));
    assert!(!fp8_grouped_decode_shape_ok(4, 0, INTER));
}

#[test]
fn intermediate_must_be_a_multiple_of_8_and_fit_the_smem_pass() {
    assert!(fp8_grouped_decode_shape_ok(4, H, 768));
    assert!(fp8_grouped_decode_shape_ok(4, H, 1024));
    assert!(!fp8_grouped_decode_shape_ok(4, H, 516));
    assert!(!fp8_grouped_decode_shape_ok(4, H, 0));
    // 8 rows x inter x 4 B + 1 KB LUT must fit the 48 KB (49152 B) no-opt-in
    // limit: inter=1504 lands exactly on it (48128 + 1024), 1512 is over.
    assert!(fp8_grouped_decode_shape_ok(4, H, 1504));
    assert!(!fp8_grouped_decode_shape_ok(4, H, 1512));
}

#[test]
fn buffer_need_matches_the_launch_layout() {
    let n = grouped_decode_buffer_need(16, 2048, 512, 256, 8);
    let te = 16 * 8;
    assert_eq!(n.scratch, 2 * te * 4);
    // sort scratch 3*te*4 + (E+1)*4 + (min(te,E)+1)*4 = 3080 < 16x256 BF16 = 8192
    assert_eq!(
        n.gate_logits,
        (16 * 256 * 2).max(3 * te * 4 + 257 * 4 + 129 * 4)
    );
    assert_eq!(n.gate_logits, 8192);
    assert_eq!(n.expert_gate_out, te * 512 * 2);
    assert_eq!(n.expert_down_out, te * 2048 * 2);
    assert_eq!(n.shared_inter, 16 * 512 * 2);
    assert_eq!(n.row_hidden, 16 * 2048 * 2);
    // At M=2 the sort scratch dominates the logits extent (cap = 16 < E).
    let n2 = grouped_decode_buffer_need(2, 2048, 256, 256, 8);
    assert_eq!(n2.gate_logits, 3 * 16 * 4 + 257 * 4 + 17 * 4);
    // Wide batch: cap saturates at E; the [64, 256] BF16 logits (32768 B)
    // still dominate the sort scratch (3*512*4 + 257*4 + 257*4 = 8200 B).
    let n64 = grouped_decode_buffer_need(64, 2048, 512, 256, 8);
    assert_eq!(n64.gate_logits, 32768);
}
