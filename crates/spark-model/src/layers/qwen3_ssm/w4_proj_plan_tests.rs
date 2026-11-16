// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the pure half of the GDN W4 arm
//! (`w4_proj_plan.rs`): the row plan table, the coverage and validity
//! invariants of every plan, the roles' shapes, and the pair checks.

use avarok_core::config::ModelConfig;

use super::{W4At, W4Chunk, W4Role, W4Step, check_pair, check_pair_against, plan};
use crate::layers::qwen3_ssm::w4_proj::fake::HostW4;
use crate::port::w4::select;

fn c(start: usize, len: usize, step: W4Step) -> W4Chunk {
    W4Chunk { start, len, step }
}

#[test]
fn the_row_plan_table() {
    let (d, p) = (W4Step::Decode, W4Step::Prefill);
    assert!(plan(W4At::Rows, 0, 16).is_empty());
    assert!(plan(W4At::Prefill, 0, 16).is_empty());
    assert_eq!(plan(W4At::Rows, 1, 16), [c(0, 1, d)]);
    assert_eq!(plan(W4At::Rows, 16, 16), [c(0, 16, d)]);
    assert_eq!(
        plan(W4At::Rows, 40, 16),
        [c(0, 16, d), c(16, 16, d), c(32, 8, d)]
    );
    // Decode-side rows never reach prefill, however wide the batch.
    assert_eq!(plan(W4At::Rows, 300, 16).len(), 19);
    assert!(plan(W4At::Rows, 300, 16).iter().all(|ch| ch.step == d));
    // Prompts: one decode inside the window, else the shared even chunks.
    assert_eq!(plan(W4At::Prefill, 16, 16), [c(0, 16, d)]);
    assert_eq!(plan(W4At::Prefill, 40, 16), [c(0, 40, p)]);
    assert_eq!(plan(W4At::Prefill, 2048, 16), [c(0, 2048, p)]);
    assert_eq!(
        plan(W4At::Prefill, 2049, 16),
        [c(0, 1025, p), c(1025, 1024, p)]
    );
    assert_eq!(
        plan(W4At::Prefill, 5000, 16),
        [c(0, 1667, p), c(1667, 1667, p), c(3334, 1666, p)]
    );
    // A weight that reports a 0 window is refused before any plan; the plan
    // itself still covers every row (as 1-row decodes).
    assert_eq!(plan(W4At::Rows, 3, 0), [c(0, 1, d), c(1, 1, d), c(2, 1, d)]);
}

/// Every plan covers `rows` exactly once with contiguous chunks; decode
/// chunks fit the window; prefill chunks are wider than it and within the
/// shared chunk cap (the scratch bound).
#[test]
fn every_plan_is_a_valid_cover() {
    for at in [W4At::Rows, W4At::Prefill] {
        for max in [1usize, 8, 16, 64] {
            let cap = select::prefill_chunk_cap(max);
            for rows in (0..300).chain([2047, 2048, 2049, 4095, 4096, 4097, 9000]) {
                let mut next = 0;
                for ch in plan(at, rows, max) {
                    assert_eq!(ch.start, next, "{at:?} rows={rows} max={max}");
                    assert!(ch.len > 0, "{at:?} rows={rows} max={max}");
                    match ch.step {
                        W4Step::Decode => assert!(ch.len <= max, "{at:?} rows={rows} max={max}"),
                        W4Step::Prefill => {
                            assert_eq!(at, W4At::Prefill, "rows={rows} max={max}");
                            assert!(ch.len > max && ch.len <= cap, "rows={rows} max={max}");
                        }
                    }
                    next += ch.len;
                }
                assert_eq!(next, rows, "{at:?} rows={rows} max={max}");
            }
        }
    }
}

#[test]
fn roles_take_their_shapes_from_the_config() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let value = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    assert_eq!(
        W4Role::Qkvz.dims(&cfg),
        (cfg.ssm_qkvz_size(), cfg.hidden_size)
    );
    assert_eq!(W4Role::OutProj.dims(&cfg), (cfg.hidden_size, value));
    assert_ne!(W4Role::Qkvz.decline_key(), W4Role::OutProj.decline_key());
    assert!(W4Role::Qkvz.what().contains("in_proj_qkvz"));
    assert!(W4Role::OutProj.what().contains("out_proj"));
}

#[test]
fn the_pair_check_refuses_impossible_shapes() {
    assert!(check_pair(12288, 2048, 2048, 4096).is_ok());
    let e = check_pair(2048, 4096, 12288, 2048).unwrap_err().to_string();
    assert!(e.contains("wrong order"), "swapped pair: {e}");
    assert!(check_pair(0, 2048, 2048, 4096).is_err());
    assert!(check_pair(12288, 2044, 2044, 4096).is_err(), "unaligned K");
    assert!(
        check_pair(12288, 2048, 2048, 4092).is_err(),
        "unaligned out_proj K"
    );
}

#[test]
fn the_config_check_names_the_projection_and_both_shapes() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let ((qn, qk), (on, ok)) = (W4Role::Qkvz.dims(&cfg), W4Role::OutProj.dims(&cfg));
    let q = HostW4::new(qn, qk, 16, true, false);
    let o = HostW4::new(on, ok, 16, true, false);
    check_pair_against(&*q, &*o, &cfg).unwrap();

    let wide = HostW4::new(qn + 128, qk, 16, true, false);
    let e = check_pair_against(&*wide, &*o, &cfg)
        .unwrap_err()
        .to_string();
    assert!(
        e.contains("in_proj_qkvz") && e.contains(&format!("[{}, {qk}]", qn + 128)),
        "{e}"
    );

    let narrow_out = HostW4::new(on, ok - 128, 16, true, false);
    let e = check_pair_against(&*q, &*narrow_out, &cfg)
        .unwrap_err()
        .to_string();
    assert!(e.contains("out_proj"), "{e}");

    let no_window = HostW4::new(on, ok, 0, true, false);
    let e = check_pair_against(&*q, &*no_window, &cfg)
        .unwrap_err()
        .to_string();
    assert!(e.contains("max_decode_rows 0"), "{e}");
}
