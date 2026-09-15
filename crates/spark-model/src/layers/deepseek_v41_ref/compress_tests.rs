// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! `torch_cpu_topk_set` against index sets recorded from this box's torch (2.11.0+cu130, CPU),
//! including the tie cases the indexer produces: many -inf entries and equal finite scores.

use super::*;

#[test]
fn topk_tie_order_matches_torch_cpu() {
    let cases: Vec<(Vec<f32>, usize, Vec<usize>)> = vec![
        (
            vec![
                1.0f32,
                1.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            6,
            vec![0, 1, 7, 8, 9, 11],
        ),
        (
            vec![
                0.0f32,
                0.0f32,
                0.0f32,
                f32::NEG_INFINITY,
                0.25f32,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.25f32,
            ],
            6,
            vec![0, 2, 4, 6, 8, 11],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                0.0f32,
                f32::NEG_INFINITY,
                0.25f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                2.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            6,
            vec![1, 3, 8, 9, 10, 12],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0f32,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
            ],
            6,
            vec![3, 8, 9, 10, 11, 12],
        ),
        (
            vec![
                1.0f32,
                0.0f32,
                1.0f32,
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                2.0f32,
                0.5f32,
                f32::NEG_INFINITY,
                1.0f32,
                1.0f32,
            ],
            6,
            vec![0, 2, 7, 8, 10, 11],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0f32,
                0.0f32,
                f32::NEG_INFINITY,
                0.5f32,
            ],
            6,
            vec![6, 7, 8, 9, 10, 12],
        ),
        (
            vec![
                0.25f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                1.0f32,
            ],
            6,
            vec![0, 7, 8, 9, 12, 13],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                1.0f32,
                0.5f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                1.0f32,
                f32::NEG_INFINITY,
                2.0f32,
                0.25f32,
                0.0f32,
                0.0f32,
                1.0f32,
            ],
            6,
            vec![1, 2, 5, 7, 8, 11],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                0.25f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.5f32,
                f32::NEG_INFINITY,
                0.0f32,
                2.0f32,
                f32::NEG_INFINITY,
                0.0f32,
                0.25f32,
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ],
            6,
            vec![1, 4, 6, 7, 10, 11],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.5f32,
                f32::INFINITY,
            ],
            3,
            vec![3, 4, 5],
        ),
        (
            vec![
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0f32,
                0.0f32,
                f32::INFINITY,
                0.5f32,
            ],
            3,
            vec![3, 5, 6],
        ),
        (
            vec![
                1.0f32,
                f32::NEG_INFINITY,
                0.5f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::INFINITY,
            ],
            3,
            vec![0, 2, 5],
        ),
    ];
    for (i, (vals, k, want)) in cases.iter().enumerate() {
        let mut got = torch_cpu_topk_set(vals, *k);
        got.sort_unstable();
        assert_eq!(&got, want, "case {i}: {vals:?} k={k}");
    }
}

#[test]
fn e2m1_rounding_ties_to_even_code() {
    // 0.25 sits between 0.0 (code 0) and 0.5 (code 1): even code wins -> 0.0
    assert_eq!(to_e2m1_rne(0.25), 0.0);
    // 0.75 between 0.5 (code 1) and 1.0 (code 2): -> 1.0
    assert_eq!(to_e2m1_rne(0.75), 1.0);
    // 2.5 between 2.0 (code 4) and 3.0 (code 5): -> 2.0
    assert_eq!(to_e2m1_rne(2.5), 2.0);
    // 5.0 between 4.0 (code 6) and 6.0 (code 7): -> 4.0
    assert_eq!(to_e2m1_rne(5.0), 4.0);
    assert_eq!(to_e2m1_rne(-1.2), -1.0);
    assert_eq!(to_e2m1_rne(7.0), 6.0);
}
