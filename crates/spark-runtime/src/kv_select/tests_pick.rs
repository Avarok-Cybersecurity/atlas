// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the picker's properties on random cases (equal to the independent
//! brute-force picker, budget never exceeded, sink and recent kept when they fit, sorted
//! output, invariance to an exact uniform score shift, the tail always kept), its edge
//! cases, the recall mass and the block-to-page expansion.

use super::oracle::pick_bruteforce;
use super::tests::Rng;
use super::*;

fn random_case(rng: &mut Rng) -> (Vec<f32>, usize, PickCfg) {
    let n = 1 + rng.below(300);
    // Multiples of 1/8 in [-4, 60): exact under the shift below, with frequent ties.
    let scores: Vec<f32> = (0..n)
        .map(|_| match rng.below(23) {
            0 => f32::NAN,
            _ => (rng.below(512) as f32 - 32.0) / 8.0,
        })
        .collect();
    let mandatory = (0..rng.below(6)).map(|_| rng.below(n + 3) as u32).collect();
    let cfg = PickCfg {
        budget_blocks: rng.below(n + 5),
        sink_blocks: rng.below(6),
        recent_blocks: rng.below(8),
        mandatory,
    };
    (scores, n, cfg)
}

#[test]
fn pick_equals_the_brute_force_picker() {
    let mut rng = Rng::new(11);
    for i in 0..3000 {
        let (scores, n, cfg) = random_case(&mut rng);
        assert_eq!(
            pick(&scores, n, &cfg),
            pick_bruteforce(&scores, n, &cfg),
            "case {i}: {cfg:?}"
        );
    }
}

#[test]
fn pick_properties_hold_on_random_cases() {
    let mut rng = Rng::new(12);
    for i in 0..3000 {
        let (scores, n, cfg) = random_case(&mut rng);
        let got = pick(&scores, n, &cfg);
        assert!(got.len() <= cfg.budget_blocks, "case {i}: budget exceeded");
        assert!(
            got.windows(2).all(|w| w[0] < w[1]),
            "case {i}: not chronological"
        );
        assert!(
            got.iter().all(|&b| (b as usize) < n),
            "case {i}: out of range"
        );
        if cfg.budget_blocks >= n {
            assert_eq!(got.len(), n, "case {i}: identity");
        } else {
            assert_eq!(got.len(), cfg.budget_blocks, "case {i}: budget not filled");
        }
        if cfg.budget_blocks >= 1 {
            assert!(got.contains(&((n - 1) as u32)), "case {i}: tail dropped");
        }
        let sink = cfg.sink_blocks.min(n);
        let recent = cfg.recent_blocks.min(n);
        if 1 + recent + sink <= cfg.budget_blocks {
            for b in (0..sink).chain(n - recent..n) {
                assert!(
                    got.contains(&(b as u32)),
                    "case {i}: block {b} of sink/recent dropped"
                );
            }
        }
        let shifted: Vec<f32> = scores.iter().map(|s| s + 3.0).collect();
        assert_eq!(
            pick(&shifted, n, &cfg),
            got,
            "case {i}: not shift invariant"
        );
        assert_eq!(pick(&scores, n, &cfg), got, "case {i}: not deterministic");
    }
}

#[test]
fn ties_break_toward_the_newer_block_and_nan_ranks_last() {
    let scores = [1.0, 2.0, 2.0, f32::NAN, 2.0, 0.5, 0.0];
    let cfg = PickCfg {
        budget_blocks: 3,
        sink_blocks: 0,
        recent_blocks: 0,
        mandatory: Vec::new(),
    };
    // Tail (6), then the newest of the 2.0 ties (4, 2).
    assert_eq!(pick(&scores, 7, &cfg), vec![2, 4, 6]);
    let cfg = PickCfg {
        budget_blocks: 6,
        ..cfg
    };
    assert_eq!(
        pick(&scores, 7, &cfg),
        vec![0, 1, 2, 4, 5, 6],
        "NaN block 3 is the one left out"
    );
}

#[test]
fn edge_budgets() {
    let scores = [5.0f32, 1.0, 9.0, 2.0];
    let none = PickCfg::default();
    assert!(pick(&scores, 4, &none).is_empty(), "budget 0 keeps nothing");
    assert!(
        pick(
            &[],
            0,
            &PickCfg {
                budget_blocks: 3,
                ..none.clone()
            }
        )
        .is_empty()
    );
    let all = PickCfg {
        budget_blocks: 9,
        ..none.clone()
    };
    assert_eq!(pick(&scores, 4, &all), vec![0, 1, 2, 3]);
    let one = PickCfg {
        budget_blocks: 1,
        sink_blocks: 2,
        recent_blocks: 2,
        ..none.clone()
    };
    assert_eq!(
        pick(&scores, 4, &one),
        vec![3],
        "the tail outranks every other class"
    );
    let mand = PickCfg {
        budget_blocks: 2,
        mandatory: vec![1, 99],
        ..none
    };
    assert_eq!(
        pick(&scores, 4, &mand),
        vec![1, 3],
        "mandatory before score; 99 ignored"
    );
    // Missing scores rank as NaN.
    let short = PickCfg {
        budget_blocks: 2,
        ..PickCfg::default()
    };
    assert_eq!(pick(&[0.0], 4, &short), vec![0, 3]);
}

#[test]
fn token_limits_round_toward_safety() {
    let c = PickCfg::from_tokens(16_384, 100, 2_049, 128, vec![4]);
    assert_eq!(c.budget_blocks, 128);
    assert_eq!(c.sink_blocks, 1);
    assert_eq!(c.recent_blocks, 17);
    assert_eq!(c.mandatory, vec![4]);
    assert_eq!(
        PickCfg::from_tokens(127, 0, 0, 128, Vec::new()).budget_blocks,
        0
    );
    assert_eq!(PickCfg::span_blocks(250, 513, 128), vec![1, 2, 3, 4]);
    assert_eq!(PickCfg::span_blocks(256, 384, 128), vec![2]);
    assert!(PickCfg::span_blocks(9, 9, 128).is_empty());
}

#[test]
fn recall_mass_is_the_kept_share_of_the_score_mass() {
    let scores = [1.0f32, 3.0, -2.0, f32::NAN, 4.0];
    assert!((recall_mass(&scores, &[1, 4]) - 7.0 / 8.0).abs() < 1e-12);
    assert!(
        (recall_mass(&scores, &[1, 1, 4, 4]) - 7.0 / 8.0).abs() < 1e-12,
        "duplicates once"
    );
    assert_eq!(recall_mass(&scores, &[2, 3]), 0.0);
    assert_eq!(recall_mass(&scores, &[0, 1, 2, 3, 4, 77]), 1.0);
    assert_eq!(recall_mass(&[0.0, -1.0], &[]), 1.0, "no mass, nothing lost");
}

#[test]
fn picked_blocks_expand_to_their_pages() {
    assert_eq!(
        expand_to_pages(&[2, 0, 2], 8, 20),
        vec![0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19]
    );
    assert_eq!(
        expand_to_pages(&[5], 8, 20),
        Vec::<usize>::new(),
        "past the end"
    );
    assert_eq!(
        expand_to_pages(&[1], 0, 3),
        vec![1],
        "pages_per_block 0 acts as 1"
    );
}
