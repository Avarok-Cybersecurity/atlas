// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the Quest lane's (L30) CPU oracles: directed f16 rounding,
//! the Quest upper bound, incremental against full summaries (including reuse
//! and rejected-draft rewrites), the radix select against a sort, the identity
//! gate (a budget covering every page gives the dense table, bit for bit), and
//! the sparse compaction's chronology and lengths. The selector's mock-backend
//! tests are in `mock_tests.rs`.

use super::reference::{self as r, KPool, RefBudget, RefGeom, Slot};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn bf16_bits(v: f32) -> u16 {
    // Round to nearest even, as a model's K write does.
    let u = v.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}

const G: RefGeom = RefGeom {
    kv_heads: 2,
    head_dim: 64,
    block_size: 16,
    page_blocks: 4,
};

/// A BF16 (eb 2) or FP8 (eb 1) pool of `blocks` random blocks and a shuffled
/// table of `nblocks` blocks for one sequence.
fn fixture(seed: u64, blocks: usize, nblocks: usize, eb: usize) -> (Vec<u8>, Vec<i32>) {
    let mut rng = Rng(seed | 1);
    let stride = G.block_size * G.kv_heads * G.head_dim * eb;
    let mut data = vec![0u8; blocks * stride];
    if eb == 2 {
        for ch in data.chunks_exact_mut(2) {
            ch.copy_from_slice(&bf16_bits(rng.unit() * 3.0).to_le_bytes());
        }
    } else {
        for b in data.iter_mut() {
            // Finite E4M3 bytes only (0x7F and 0xFF are NaN).
            let v = (rng.next() & 0xFF) as u8;
            *b = if v & 0x7F == 0x7F { v & 0xFE } else { v };
        }
    }
    let mut ids: Vec<i32> = (0..blocks as i32).collect();
    for i in (1..ids.len()).rev() {
        ids.swap(i, rng.below(i + 1));
    }
    ids.truncate(nblocks);
    (data, ids)
}

fn pool(data: &[u8], eb: usize) -> KPool<'_> {
    KPool {
        data,
        elem_bytes: eb,
        block_stride: G.block_size * G.kv_heads * G.head_dim * eb,
        geom: G,
    }
}

#[test]
fn directed_rounding_brackets_every_value() {
    let mut rng = Rng(7);
    for _ in 0..20_000 {
        let x = rng.unit() * 2f32.powi(rng.below(40) as i32 - 24);
        let (lo, hi) = (r::f16_val(r::f16_rd(x)), r::f16_val(r::f16_ru(x)));
        assert!(lo <= x && x <= hi, "{x}: {lo} {hi}");
        let h = half::f16::from_f32(x).to_f32();
        if h == x {
            assert_eq!((lo, hi), (x, x), "an exact f16 rounds to itself");
        }
    }
    assert_eq!(r::f16_val(r::f16_rd(1e6)), 65504.0);
    assert_eq!(r::f16_val(r::f16_ru(1e6)), f32::INFINITY);
    assert_eq!(r::f16_val(r::f16_ru(-1e6)), -65504.0);
    assert!(r::f16_val(r::f16_rd(-1e-12)) < 0.0);
    assert_eq!(r::f16_val(r::f16_rd(1e-12)), 0.0);
}

#[test]
fn e4m3_decoding_matches_the_format() {
    assert_eq!(r::e4m3_val(0x38), 1.0);
    assert_eq!(r::e4m3_val(0x7E), 448.0);
    assert_eq!(r::e4m3_val(0xFE), -448.0);
    assert_eq!(r::e4m3_val(0x01), 2f32.powi(-9));
    assert!(r::e4m3_val(0x7F).is_nan());
}

#[test]
fn the_quest_score_bounds_every_key_of_its_page() {
    for eb in [2, 1] {
        let (data, table) = fixture(11 + eb as u64, 40, 32, eb);
        let p = pool(&data, eb);
        let mut rng = Rng(99);
        for page in 0..8 {
            for h in 0..G.kv_heads {
                let s = r::page_slot(&p, &table, page, h);
                let q: Vec<f32> = (0..G.head_dim).map(|_| rng.unit() * 2.0).collect();
                let b = r::bound(&q, &s.kmin, &s.kmax);
                for t in page * 64..(page + 1) * 64 {
                    let dot: f64 = (0..G.head_dim)
                        .map(|c| f64::from(q[c]) * f64::from(p.k_at(&table, t, h, c)))
                        .sum();
                    assert!(
                        b >= dot - 1e-9,
                        "eb {eb} page {page} head {h} token {t}: {b} < {dot}"
                    );
                }
            }
        }
    }
}

/// Fold a sequence of written ranges like the update kernel would.
fn fold_writes(p: &KPool, table: &[i32], slots: &mut [Vec<Slot>], writes: &[(usize, usize)]) {
    let pt = G.page_tokens();
    for &(first, count) in writes {
        let last = first + count - 1;
        for (page, row) in slots
            .iter_mut()
            .enumerate()
            .take(last / pt + 1)
            .skip(first / pt)
        {
            for (h, slot) in row.iter_mut().enumerate() {
                r::update_slot(slot, p, table, page, h, first, count);
            }
        }
    }
}

#[test]
fn incremental_summaries_equal_the_full_build() {
    for eb in [2, 1] {
        let (data, table) = fixture(21 + eb as u64, 48, 40, eb);
        let p = pool(&data, eb);
        let pages = 640 / 64;
        let mut slots = vec![vec![Slot::empty(G.head_dim); G.kv_heads]; pages];
        // A 100-token prefill chunk, a 157-token chunk, single decode tokens, then verify rows of 4.
        let mut writes = vec![(0, 100), (100, 157)];
        writes.extend((257..400).map(|t| (t, 1)));
        writes.extend((400..640).step_by(4).map(|t| (t, 4)));
        fold_writes(&p, &table, &mut slots, &writes);
        for (page, row) in slots.iter().enumerate() {
            for (h, slot) in row.iter().enumerate() {
                let full = r::page_slot(&p, &table, page, h);
                assert_eq!(*slot, full, "eb {eb} page {page} head {h}");
                assert!(r::trusted(slot, &p, &table, page, h));
            }
        }
    }
}

#[test]
fn a_reused_block_resets_its_stale_summary() {
    let (data_a, table) = fixture(31, 16, 8, 2);
    let (data_b, _) = fixture(32, 16, 8, 2);
    let (pa, pb) = (pool(&data_a, 2), pool(&data_b, 2));
    // Page 0 fully summarized for content A.
    let mut slot = r::page_slot(&pa, &table, 0, 0);
    // The same blocks now hold content B, written from the page start: reset.
    r::update_slot(&mut slot, &pb, &table, 0, 0, 0, 64);
    assert_eq!(slot, r::page_slot(&pb, &table, 0, 0));
    // Content B appended mid-page into a slot holding A: the fingerprint differs, so reset too.
    let mut stale = r::page_slot(&pa, &table, 0, 0);
    r::update_slot(&mut stale, &pb, &table, 0, 0, 20, 10);
    assert_ne!(
        stale.mask,
        r::full_mask(64),
        "a partial fold after a reset is not trusted"
    );
    assert!(!r::trusted(&stale, &pb, &table, 0, 0));
}

#[test]
fn rejected_drafts_only_loosen_the_bound() {
    let (mut data, table) = fixture(41, 16, 8, 2);
    let mut slot = Slot::empty(G.head_dim);
    {
        let p = pool(&data, 2);
        r::update_slot(&mut slot, &p, &table, 0, 0, 0, 40);
        r::update_slot(&mut slot, &p, &table, 0, 0, 40, 4); // drafts 40..43
    }
    // Drafts 41..43 rejected and rewritten with other keys, then the page completes.
    let bs = G.block_size;
    let stride = bs * G.kv_heads * G.head_dim * 2;
    let mut rng = Rng(42);
    for t in 41..44 {
        let base = table[t / bs] as usize * stride + (t % bs) * G.kv_heads * G.head_dim * 2;
        for ch in data[base..base + G.head_dim * 2].chunks_exact_mut(2) {
            ch.copy_from_slice(&bf16_bits(rng.unit() * 5.0).to_le_bytes());
        }
    }
    let p = pool(&data, 2);
    r::update_slot(&mut slot, &p, &table, 0, 0, 41, 23);
    assert!(r::trusted(&slot, &p, &table, 0, 0));
    let exact = r::page_slot(&p, &table, 0, 0);
    for (got, want) in slot.kmin.iter().zip(&exact.kmin) {
        assert!(r::f16_val(*got) <= r::f16_val(*want));
    }
    for (got, want) in slot.kmax.iter().zip(&exact.kmax) {
        assert!(r::f16_val(*got) >= r::f16_val(*want));
    }
}

#[test]
fn the_radix_select_finds_the_kth_largest_key() {
    let mut rng = Rng(5);
    for trial in 0..200 {
        let n = 1 + rng.below(3000);
        // Few distinct scores: ties are the hard case.
        let scores: Vec<f32> = (0..n).map(|_| (rng.below(7) as f32 - 3.0) * 0.5).collect();
        let keys: Vec<u64> = scores
            .iter()
            .enumerate()
            .map(|(p, &s)| r::key(s, p))
            .collect();
        let k = 1 + rng.below(n);
        let mut sorted = keys.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(
            r::radix_threshold(&keys, k),
            sorted[k - 1],
            "trial {trial} n {n} k {k}"
        );
    }
    // NaN ranks above +inf's neighbors as +inf; -inf ranks last.
    assert!(r::key(f32::NAN, 0) > r::key(1e30, 1));
    assert!(r::key(f32::NEG_INFINITY, 9) < r::key(-1e30, 0));
}

fn budget(topk: usize, min_ctx: usize) -> RefBudget {
    RefBudget {
        sink_pages: 1,
        recent_tokens: 256,
        topk_pages: topk,
        min_ctx,
    }
}

#[test]
fn the_identity_gate_returns_the_dense_table_bit_for_bit() {
    let (_, table) = fixture(51, 300, 260, 2);
    let mut stride_table = table.clone();
    stride_table.resize(272, -1);
    let lens = [4100, 4101, 4102, 4103];
    let rows: Vec<&[i32]> = (0..4).map(|_| stride_table.as_slice()).collect();
    let scores = vec![1.0f32; 68];
    // The budget covers every candidate page.
    for b in [budget(1000, 0), budget(10, 1 << 20)] {
        let c = r::compact(&rows, &lens, &scores, G, b, 272, 68, 8192);
        assert!(!c.sparse);
        assert_eq!(c.lens, lens);
        for (row, &e) in c.tables.iter().zip(&lens) {
            assert_eq!(row.as_slice(), &stride_table[..(e as usize).div_ceil(16)]);
        }
    }
}

#[test]
fn a_sparse_view_keeps_sinks_topk_and_the_recent_window_in_order() {
    let mut rng = Rng(61);
    let (_, table) = fixture(61, 300, 260, 2);
    let lens = [4100i32, 4101, 4102, 4103]; // verify rows: one selection shared
    let rows: Vec<&[i32]> = (0..4).map(|_| table.as_slice()).collect();
    let scores: Vec<f32> = (0..65).map(|_| rng.unit()).collect();
    let b = budget(8, 1024);
    let c = r::compact(&rows, &lens, &scores, G, b, 260, 65, 8192);
    assert!(c.sparse);
    let rp = r::recent_page(4100, 256, 64);
    assert_eq!(rp, (4100 - 256) / 64);
    let chosen = r::topk_sorted(&scores, 1, rp, 8);
    // The chosen pages are exactly the eight best candidates.
    let mut best: Vec<usize> = (1..rp).collect();
    best.sort_by(|&x, &y| scores[y].total_cmp(&scores[x]));
    let mut best8 = best[..8].to_vec();
    best8.sort_unstable();
    assert_eq!(chosen, best8);
    for (i, (view, &e)) in c.tables.iter().zip(&lens).enumerate() {
        let len = c.lens[i] as usize;
        assert_eq!(len, (1 + 8) * 64 + (e as usize - rp * 64));
        let pos =
            r::compacted_positions(&table, view, len, 16).expect("every block is the row's own");
        assert!(pos.windows(2).all(|w| w[0] < w[1]), "chronological");
        assert_eq!(
            *pos.last().unwrap(),
            e as usize - 1,
            "the row's own newest token is last"
        );
        let kept_pages: Vec<usize> = pos.iter().map(|p| p / 64).collect();
        assert!(kept_pages.contains(&0), "the sink page");
        for p in &chosen {
            assert!(kept_pages.contains(p));
        }
        assert!(
            (rp * 64..e as usize).all(|t| pos.contains(&t)),
            "the recent window is whole"
        );
    }
    assert_eq!(c.pages, 1 + 8 + (4103usize.div_ceil(16).div_ceil(4) - rp));
}

#[test]
fn a_planted_needle_page_is_selected() {
    // Keys of page 20 align with the query; every other key is noise.
    let (mut data, table) = fixture(71, 300, 260, 2);
    let mut rng = Rng(72);
    let q: Vec<f32> = (0..2 * 2 * G.head_dim).map(|_| rng.unit()).collect(); // nq = 4, kvh = 2
    let bs = G.block_size;
    let stride = bs * G.kv_heads * G.head_dim * 2;
    for t in 20 * 64..21 * 64 {
        let off = table[t / bs] as usize * stride + (t % bs) * G.kv_heads * G.head_dim * 2;
        for (c, qc) in q.iter().take(G.head_dim).enumerate() {
            let v = bf16_bits(qc.signum() * 4.0).to_le_bytes();
            data[off + 2 * c..off + 2 * c + 2].copy_from_slice(&v);
        }
    }
    let p = pool(&data, 2);
    let rp = r::recent_page(4000, 256, 64);
    let mut scores = vec![0f32; 65];
    for (page, s) in scores.iter_mut().enumerate().take(rp).skip(1) {
        let slots: Vec<Slot> = (0..G.kv_heads)
            .map(|h| r::page_slot(&p, &table, page, h))
            .collect();
        *s = r::page_score(&[q.as_slice()], 4, &slots, G.head_dim, false) as f32;
    }
    assert!(r::topk_sorted(&scores, 1, rp, 2).contains(&20));
}
