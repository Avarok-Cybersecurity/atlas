// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the FP64 reference itself: the strided reshape is the antidiagonal sum,
//! block masses are a distribution, the selection keeps a planted block, and attention through a block
//! mask equals plain causal attention with the dropped keys removed (all blocks kept = dense).

use super::BLOCK;
use super::reference::{self, Problem};
use super::select::forced;

struct Data {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
}

fn data(
    q_len: usize,
    kv_len: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    seed: u64,
    scale: f32,
) -> Data {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
    };
    Data {
        q: (0..q_len * nq * hd).map(|_| next()).collect(),
        k: (0..kv_len * nkv * hd).map(|_| next()).collect(),
        v: (0..kv_len * nkv * hd).map(|_| next()).collect(),
    }
}

fn problem(
    d: &Data,
    q_len: usize,
    kv_len: usize,
    q_offset: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
) -> Problem<'_> {
    Problem {
        q: &d.q,
        k: &d.k,
        v: &d.v,
        q_len,
        kv_len,
        q_offset,
        nq,
        nkv,
        hd,
        inv_sqrt_d: 1.0 / (hd as f64).sqrt(),
    }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

#[test]
fn the_strided_reshape_is_the_antidiagonal_sum() {
    let (q_len, kv_len, hd) = (384, 384, 8);
    let d = data(q_len, kv_len, 2, 1, hd, 11, 1.0);
    let p = problem(&d, q_len, kv_len, 0, 2, 1, hd);
    let bl = BLOCK as usize;
    for stride in [4usize, 8, 16] {
        let r_rows = bl / stride;
        for (i, j) in [(2usize, 0usize), (2, 1), (1, 0)] {
            for h in 0..2 {
                let mut strided = 0.0;
                for a in 0..r_rows {
                    for b in 0..r_rows {
                        strided += reference::strided_logit(&p, h, i, a, j, b, stride).unwrap();
                    }
                }
                strided *= stride as f64 / p.inv_sqrt_d;
                // Every (query, key) pair of the two blocks whose in-group offsets sum to S - 1.
                let mut direct = 0.0;
                for x in 0..bl {
                    for y in 0..bl {
                        if x % stride + y % stride == stride - 1 {
                            direct += dot(
                                p.q_row(i * bl + x, h).unwrap(),
                                p.k_row(j * bl + y, 0).unwrap(),
                            );
                        }
                    }
                }
                assert!(
                    (strided - direct).abs() <= 1e-9 * direct.abs().max(1.0),
                    "S {stride} ({i},{j}) h {h}"
                );
            }
        }
    }
}

#[test]
fn the_diagonal_masks_later_strided_keys_and_past_kv_len_masks_the_group() {
    let (q_len, kv_len, hd) = (256, 250, 8);
    let d = data(q_len, 256, 1, 1, hd, 3, 1.0);
    let p = problem(&d, q_len, kv_len, 0, 1, 1, hd);
    assert!(reference::strided_logit(&p, 0, 1, 3, 1, 3, 8).is_some());
    assert!(
        reference::strided_logit(&p, 0, 1, 3, 1, 4, 8).is_none(),
        "b > a on the diagonal"
    );
    assert!(
        reference::strided_logit(&p, 0, 1, 15, 0, 15, 8).is_some(),
        "off the diagonal nothing is masked"
    );
    // kv_len 250: block 1's group b = 15 starts at 248 (< 250, kept), but at stride 4 group b = 31 starts at 252.
    assert!(reference::strided_logit(&p, 0, 1, 15, 1, 15, 8).is_some());
    assert!(reference::strided_logit(&p, 0, 1, 31, 1, 31, 4).is_none());
}

#[test]
fn block_masses_are_a_distribution_and_an_uncounted_block_keeps_everything() {
    let (q_len, kv_len, q_offset, hd) = (300, 812, 512, 16);
    let d = data(q_len, kv_len, 2, 1, hd, 5, 1.0);
    let p = problem(&d, q_len, kv_len, q_offset, 2, 1, hd);
    let (lo, hi) = p.query_blocks();
    assert_eq!((lo, hi), (4, 6));
    for stride in [4usize, 8, 16] {
        for h in 0..2 {
            for i in lo..=hi {
                let m = reference::block_mass(&p, h, i, stride).expect("counted rows");
                assert_eq!(m.len(), p.candidates(i));
                let s: f64 = m.iter().sum();
                assert!((s - 1.0).abs() < 1e-12, "S {stride} h {h} i {i}: {s}");
                assert!(m.iter().all(|x| *x >= 0.0));
            }
        }
    }
    // A three-row chunk inside block 1 has no complete strided row at stride 8.
    let d = data(3, 133, 1, 1, hd, 9, 1.0);
    let p = problem(&d, 3, 133, 130, 1, 1, hd);
    assert!(reference::block_mass(&p, 0, 1, 8).is_none());
    let masks = reference::masks(&p, 8, 0.5);
    assert_eq!(masks, vec![vec![true, true]]);
}

#[test]
fn masks_keep_the_forced_blocks_everything_at_one_and_grow_with_the_threshold() {
    let (q_len, kv_len, q_offset, hd) = (256, 1280, 1024, 16);
    let d = data(q_len, kv_len, 2, 2, hd, 21, 1.0);
    let p = problem(&d, q_len, kv_len, q_offset, 2, 2, hd);
    let all = reference::masks(&p, 8, 1.0);
    assert!(all.iter().all(|m| m.iter().all(|x| *x)));
    let mut prev: Option<Vec<Vec<bool>>> = None;
    for t in [0.3, 0.6, 0.9] {
        let m = reference::masks(&p, 8, t);
        for sel in &m {
            let jd = sel.len() - 1;
            assert!((0..sel.len()).filter(|&j| forced(j, jd)).all(|j| sel[j]));
        }
        if let Some(prev) = &prev {
            for (a, b) in prev.iter().zip(&m) {
                assert!(a.iter().zip(b).all(|(x, y)| !*x || *y), "threshold {t}");
            }
        }
        prev = Some(m);
    }
}

#[test]
fn a_planted_block_is_selected_and_most_others_are_not() {
    let (q_len, kv_len, q_offset, hd) = (256, 1536, 1280, 16);
    let mut d = data(q_len, kv_len, 1, 1, hd, 33, 0.05);
    // Query block 11 (rows 1408..1535 = chunk rows 128..255) and key block 4 share one direction.
    for row in 128..256 {
        d.q[row * hd..(row + 1) * hd]
            .iter_mut()
            .for_each(|x| *x = 2.0);
    }
    for pos in 4 * 128..5 * 128 {
        d.k[pos * hd..(pos + 1) * hd]
            .iter_mut()
            .for_each(|x| *x = 2.0);
    }
    let p = problem(&d, q_len, kv_len, q_offset, 1, 1, hd);
    let m = reference::block_mass(&p, 0, 11, 8).unwrap();
    assert!(m[4] > 0.9, "planted block mass {}", m[4]);
    let masks = reference::masks(&p, 8, 0.9);
    let sel = &masks[1];
    assert_eq!(sel.len(), 12);
    assert!(sel[4]);
    let kept: Vec<usize> = (0..12).filter(|&j| sel[j]).collect();
    assert_eq!(kept, vec![0, 4, 10, 11], "sink, planted, band");
}

#[test]
fn keep_all_is_dense_and_a_block_mask_drops_exactly_its_keys() {
    let (q_len, kv_len, q_offset, nq, nkv, hd) = (200, 600, 400, 4, 2, 16);
    let d = data(q_len, kv_len, nq, nkv, hd, 77, 1.0);
    let p = problem(&d, q_len, kv_len, q_offset, nq, nkv, hd);
    let bl = BLOCK as usize;
    // Independent dense: softmax over every causal key, no block logic.
    let mut naive = vec![0.0; q_len * nq * hd];
    for row in 0..q_len {
        let pos = q_offset + row;
        for h in 0..nq {
            let g = h / (nq / nkv);
            let q = p.q_row(pos, h).unwrap();
            let w: Vec<f64> = (0..=pos)
                .map(|kp| (dot(q, p.k_row(kp, g).unwrap()) * p.inv_sqrt_d).exp())
                .collect();
            let l: f64 = w.iter().sum();
            for (kp, wk) in w.iter().enumerate() {
                for x in 0..hd {
                    naive[(row * nq + h) * hd + x] += wk / l * d.v[(kp * nkv + g) * hd + x] as f64;
                }
            }
        }
    }
    let dense = reference::dense(&p);
    let max = dense
        .iter()
        .zip(&naive)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(max < 1e-12, "dense vs naive {max}");
    let all = reference::masks(&p, 8, 1.0);
    assert_eq!(
        reference::sparse(&p, &all),
        dense,
        "keep-all masks are the dense path, exactly"
    );

    // Drop key block 1 for head 2 only: head 2 must equal attention without keys 128..255.
    let (lo, hi) = p.query_blocks();
    let n_qb = hi - lo + 1;
    let mut masks = all.clone();
    for qb in 0..n_qb {
        masks[2 * n_qb + qb][1] = false;
    }
    let sparse = reference::sparse(&p, &masks);
    for row in 0..q_len {
        let pos = q_offset + row;
        for h in 0..nq {
            let at = (row * nq + h) * hd;
            if h != 2 {
                assert_eq!(sparse[at..at + hd], dense[at..at + hd]);
                continue;
            }
            let g = h / (nq / nkv);
            let q = p.q_row(pos, h).unwrap();
            let keys: Vec<usize> = (0..=pos).filter(|kp| kp / bl != 1).collect();
            let w: Vec<f64> = keys
                .iter()
                .map(|&kp| (dot(q, p.k_row(kp, g).unwrap()) * p.inv_sqrt_d).exp())
                .collect();
            let l: f64 = w.iter().sum();
            for x in 0..hd {
                let want: f64 = keys
                    .iter()
                    .zip(&w)
                    .map(|(&kp, wk)| wk / l * d.v[(kp * nkv + g) * hd + x] as f64)
                    .sum();
                assert!((sparse[at + x] - want).abs() < 1e-12, "row {row} dim {x}");
            }
        }
    }
}
