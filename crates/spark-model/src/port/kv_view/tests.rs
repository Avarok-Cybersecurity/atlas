// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the bounded view's pure parts (port lane L32):
//! the pick-to-shape conversion, view composition and the anchor rule (the
//! oracle of the device kernel), the identity gate, and the history and
//! propagation formulas (alpha 0 identity, mass conservation, determinism).

use std::sync::Arc;

use super::history::{Edge, HIT_FRACTION, History, hit_threshold, propagate};
use super::view::{
    DESC_ACTIVE, Desc, ViewShape, best_match, build_row, mandatory_blocks, pages_of, prefix_hash,
    shape_from_pick,
};

const BS: u32 = 16;
const BP: u32 = 8;

/// A real table of `pages` pages with physical ids `1000 + i`.
fn table(pages: u32) -> Vec<u32> {
    (0..pages).map(|i| 1000 + i).collect()
}

/// A small deterministic generator (no external crate).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

#[test]
fn the_tail_is_the_last_run_of_picked_blocks() {
    // 32 blocks of 8 pages; blocks 0, 5, 6 selected, 20..32 the tail.
    let mut picked = vec![0, 5, 6];
    picked.extend(20..32);
    let shape = shape_from_pick(&picked, 256, BP).unwrap();
    assert_eq!(shape.tail_page, 160);
    let expect: Vec<u32> = (0..8).chain(40..56).collect();
    assert_eq!(shape.sel_pages, expect);
    assert_eq!(shape.drop_tokens(BS), (160 - 24) * BS);
}

#[test]
fn a_pick_that_drops_nothing_is_the_identity() {
    let all: Vec<u32> = (0..32).collect();
    assert_eq!(shape_from_pick(&all, 256, BP), None);
    assert_eq!(shape_from_pick(&[], 0, BP), None);
    // Unsorted, duplicated and out-of-range picks are cleaned; the last
    // (partial) block is always kept.
    let messy = [31, 3, 3, 99, 0];
    let shape = shape_from_pick(&messy, 250, BP).unwrap();
    assert_eq!(shape.tail_page, 31 * BP);
    let expect: Vec<u32> = (0..8).chain(24..32).collect();
    assert_eq!(shape.sel_pages, expect);
}

#[test]
fn compose_keeps_selected_pages_then_everything_from_the_tail() {
    let shape = ViewShape {
        sel_pages: vec![0, 1, 9],
        tail_page: 12,
    };
    let row = table(20);
    let len = 19 * BS + 5; // 20 pages, the last partial
    let (pages, view_len) = shape.compose(&row, len, BS).unwrap();
    let mut expect = vec![1000, 1001, 1009];
    expect.extend(1012..1020);
    assert_eq!(pages, expect);
    assert_eq!(view_len, len - (12 - 3) * BS);
    assert_eq!(pages_of(view_len, BS) as usize, pages.len());
    // A row that does not reach the tail page has no view.
    assert_eq!(shape.compose(&row, 12 * BS, BS), None);
    // A malformed shape is refused.
    let bad = ViewShape {
        sel_pages: vec![3, 2],
        tail_page: 12,
    };
    assert_eq!(bad.compose(&row, len, BS), None);
}

#[test]
fn anchors_bind_a_view_to_its_own_sequence_only() {
    let row = table(40);
    let len = 40 * BS;
    let shape = ViewShape {
        sel_pages: (0..8).collect(),
        tail_page: 24,
    };
    let d = Desc::bind(shape.clone(), &row, len, BS).unwrap();
    assert_eq!(
        (d.a0, d.a_tail, d.a_last, d.last_page),
        (1000, 1024, 1039, 39)
    );
    let w = d.words();
    assert_eq!(w[0], DESC_ACTIVE);
    assert_eq!(&w[1..7], &[1000, 1024, 1039, 39, 24, 8]);
    assert!(d.matches(&row, len, BS));
    // The same sequence one decode page later still matches.
    let mut grown = row.clone();
    grown.push(5000);
    assert!(d.matches(&grown, len + 1, BS));
    // Another sequence sharing only the prefix pages does not.
    let mut other = row.clone();
    other[39] = 7777;
    assert!(!d.matches(&other, len, BS));
    // A shorter row (before the selection point) does not.
    assert!(!d.matches(&row, 39 * BS, BS));
    // Binding needs the row to reach the tail page.
    assert_eq!(Desc::bind(shape, &row, 24 * BS, BS), None);
}

#[test]
fn the_most_specific_view_wins() {
    let row = table(64);
    let early = Desc::bind(
        ViewShape {
            sel_pages: vec![0],
            tail_page: 16,
        },
        &row,
        32 * BS,
        BS,
    )
    .unwrap();
    let late = Desc::bind(
        ViewShape {
            sel_pages: vec![0, 1],
            tail_page: 40,
        },
        &row,
        60 * BS,
        BS,
    )
    .unwrap();
    let both = [Some(&early), None, Some(&late)];
    assert_eq!(best_match(&both, &row, 64 * BS, BS), Some(&late));
    // Before the late selection point only the early view applies.
    assert_eq!(best_match(&both, &row, 50 * BS, BS), Some(&early));
}

#[test]
fn identity_gate_rows_without_a_view_keep_their_real_table() {
    let row = table(30);
    for len in [1, BS, 29 * BS + 3] {
        let (pages, l) = build_row(&row, len, BS, &[]);
        assert_eq!(l, len);
        assert_eq!(pages, row[..pages_of(len, BS) as usize]);
    }
    let (pages, l) = build_row(&row, 0, BS, &[]);
    assert!(pages.is_empty());
    assert_eq!(l, 0);
}

#[test]
fn build_row_matches_compose_on_random_views() {
    let mut g = Lcg(0x32_2026_1112);
    for _ in 0..200 {
        let pages = 8 + g.next() % 200;
        let len = (pages - 1) * BS + 1 + g.next() % BS;
        let row = table(pages);
        let tail_page = 1 + g.next() % (pages - 1);
        let mut sel: Vec<u32> = (0..tail_page)
            .filter(|_| g.next().is_multiple_of(3))
            .collect();
        sel.dedup();
        let shape = ViewShape {
            sel_pages: sel,
            tail_page,
        };
        let d = Desc::bind(shape.clone(), &row, len, BS).unwrap();
        let (got, view_len) = build_row(&row, len, BS, &[Some(&d)]);
        let (want, want_len) = shape.compose(&row, len, BS).unwrap();
        assert_eq!((got.clone(), view_len), (want, want_len));
        assert_eq!(pages_of(view_len, BS) as usize, got.len());
        assert_eq!(view_len, len - shape.drop_tokens(BS));
        // Chronological: physical ids (1000 + logical page) strictly increase.
        assert!(got.windows(2).all(|w| w[0] < w[1]));
    }
}

#[test]
fn mandatory_blocks_cover_the_query_span() {
    assert_eq!(mandatory_blocks(0..0, 128), Vec::<u32>::new());
    assert_eq!(mandatory_blocks(100..129, 128), vec![0, 1]);
    assert_eq!(mandatory_blocks(256..384, 128), vec![2]);
}

#[test]
fn prefix_hash_tells_prefixes_apart() {
    let a: Vec<u32> = (0..1000).collect();
    let mut b = a.clone();
    b[500] = 7;
    assert_eq!(prefix_hash(&a), prefix_hash(&a.clone()));
    assert_ne!(prefix_hash(&a), prefix_hash(&b));
    assert_ne!(prefix_hash(&a[..10]), prefix_hash(&a[..11]));
}

#[test]
fn propagation_with_alpha_zero_is_the_identity() {
    let s = vec![0.5, -0.25, 3.0, 0.0];
    assert_eq!(propagate(&s, &[], 0.0), s);
    let edges = [Edge {
        src: 2,
        dst: 0,
        w: 1.0,
    }];
    assert_eq!(propagate(&s, &edges, 0.0), s);
}

#[test]
fn propagation_conserves_mass_and_never_spreads_negative_scores() {
    let s = vec![1.0, -2.0, 4.0, 0.5, 0.0, 2.0];
    let edges = [
        Edge {
            src: 3,
            dst: 0,
            w: 2.0,
        },
        Edge {
            src: 3,
            dst: 1,
            w: 1.0,
        },
        // Self and future edges are ignored.
        Edge {
            src: 2,
            dst: 2,
            w: 1.0,
        },
        Edge {
            src: 1,
            dst: 4,
            w: 1.0,
        },
    ];
    let out = propagate(&s, &edges, 0.2);
    let total = |v: &[f32]| v.iter().map(|&x| f64::from(x)).sum::<f64>();
    assert!((total(&out) - total(&s)).abs() < 1e-5);
    // Block 1's negative score stays put except for what its neighbors send.
    let sent_to_1 = 0.2 / 4.0 * (1.0 + 4.0) + 0.2 / 2.0 * 0.5 / 3.0;
    assert!((f64::from(out[1]) - (-2.0 + sent_to_1)).abs() < 1e-5);
    // The edge share of block 3 goes 2:1 to blocks 0 and 1.
    let expect_0 = 1.0 - 0.2 / 4.0 * 1.0 + 0.2 / 2.0 * 0.5 * 2.0 / 3.0;
    assert!((f64::from(out[0]) - expect_0).abs() < 1e-5, "{out:?}");
    // Deterministic.
    assert_eq!(propagate(&s, &edges, 0.2), out);
}

#[test]
fn propagation_moves_a_quarter_alpha_to_each_neighbor() {
    let s = vec![0.0, 8.0, 0.0];
    let out = propagate(&s, &[], 0.2);
    assert!((out[0] - 0.4).abs() < 1e-6);
    assert!((out[2] - 0.4).abs() < 1e-6);
    assert!((out[1] - 7.2).abs() < 1e-6);
}

#[test]
fn history_with_lambda_zero_is_the_round_itself() {
    let h = History::default();
    let s = [0.1f32, 0.7, 0.2];
    assert_eq!(&*h.preview(&s, 0.0), &s);
}

#[test]
fn history_decays_and_gates_positive_residuals_by_hits() {
    let s: Vec<f32> = (0..10).map(|i| i as f32).collect();
    // The bottom 10% (one block) is not a hit.
    assert_eq!(hit_threshold(&s), 1.0);
    assert_eq!(hit_threshold(&s[..4]), 0.0, "four blocks: all are hits");
    assert!((HIT_FRACTION - 0.9).abs() < 1e-12);
    let h = History {
        committed: Arc::from(vec![2.0f32; 4]),
        rounds: 1,
    };
    let p = h.preview(&s, 0.5);
    let mean = 4.5f64;
    for (b, &v) in p.iter().enumerate() {
        let prev = if b < 4 { 2.0 } else { 0.0 };
        let r = b as f64 - mean;
        let hit = if b >= 1 { 1.0 } else { 0.0 };
        let want = 0.5 * prev + r.min(0.0) + hit * r.max(0.0);
        assert!(
            (f64::from(v) - want).abs() < 1e-5,
            "block {b}: {v} vs {want}"
        );
    }
    let c = h.commit(p.clone());
    assert_eq!(c.rounds, 2);
    assert_eq!(c.committed, p);
    // A related earlier round lifts a block the current round ranks low.
    let prior = History {
        committed: Arc::from(vec![0.0, 0.0, 6.0, 0.0]),
        rounds: 1,
    };
    let now = [1.0f32, 1.2, 0.9, 1.1];
    let ranked = prior.preview(&now, 0.5);
    let best = (0..4)
        .max_by(|&a, &b| ranked[a].total_cmp(&ranked[b]))
        .unwrap();
    assert_eq!(best, 2);
}
