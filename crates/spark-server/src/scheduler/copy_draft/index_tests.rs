// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracle for [`super::index::CopyIndex`].
//!
//! `longest` must equal a brute-force scan of the same contract (the longest
//! window whose suffix occurs earlier, its LATEST earlier occurrence, the
//! match extended backward) for EVERY prefix of random, periodic, planted-copy
//! and collision-forced streams, tie order included; `continuation` must equal
//! its definition; a rewritten stream must index exactly as a fresh one; the
//! bounded index must match the scan restricted to its floor.

use super::index::{CopyIndex, MATCH_LIMIT, Match, SyncOutcome, WINDOWS};

/// xorshift64*: deterministic and dependency-free.
pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub(super) fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

fn brute_extend(toks: &[u32], end: usize, w: usize) -> usize {
    let len = toks.len();
    let mut n = w;
    while n < MATCH_LIMIT && n <= end && toks[end - n] == toks[len - 1 - n] {
        n += 1;
    }
    n
}

/// The contract of `CopyIndex::longest`, by plain scan.
pub(super) fn brute_longest(toks: &[u32], floor: usize) -> Option<Match> {
    let len = toks.len();
    if len < 2 {
        return None;
    }
    for (wi, &w) in WINDOWS.iter().enumerate() {
        if len < w + 1 {
            continue;
        }
        let suffix = &toks[len - w..];
        let lo = floor.max(w - 1);
        // Ends strictly before the anchor, latest first.
        for e in (lo..=len - 2).rev() {
            if toks[e + 1 - w..=e] == *suffix {
                return Some(Match {
                    window: wi,
                    end: e,
                    n: brute_extend(toks, e, w),
                });
            }
        }
    }
    None
}

fn brute_continuation(toks: &[u32], next: usize, want: usize) -> Vec<u32> {
    let len = toks.len();
    let mut out: Vec<u32> = Vec::new();
    if next >= len {
        return out;
    }
    for j in 0..want {
        if next + j < len {
            out.push(toks[next + j]);
        } else {
            let v = out[j - (len - next)];
            out.push(v);
        }
    }
    out
}

fn check_every_prefix(stream: &[u32], mut idx: CopyIndex, label: &str) {
    for k in 0..=stream.len() {
        idx.sync(&stream[..k], None);
        assert_eq!(idx.tokens(), &stream[..k], "{label}: held tokens at {k}");
        let want = brute_longest(&stream[..k], idx.floor());
        assert_eq!(idx.longest(), want, "{label}: longest at prefix {k}");
    }
}

fn random_stream(rng: &mut Rng, len: usize, symbols: u64) -> Vec<u32> {
    (0..len).map(|_| rng.below(symbols) as u32).collect()
}

#[test]
fn longest_equals_the_scan_on_random_streams() {
    let mut rng = Rng::new(7);
    for round in 0..120 {
        let symbols = 1 + rng.below(4);
        let len = 1 + rng.below(300) as usize;
        let s = random_stream(&mut rng, len, symbols);
        check_every_prefix(&s, CopyIndex::new(), &format!("random round {round}"));
    }
}

#[test]
fn longest_equals_the_scan_on_wide_alphabets_with_planted_copies() {
    let mut rng = Rng::new(11);
    for round in 0..40 {
        let mut s = random_stream(&mut rng, 200, 5000);
        // Plant three copies of earlier spans, some longer than 16.
        for _ in 0..3 {
            let from = rng.below(150) as usize;
            let n = 4 + rng.below(40) as usize;
            let span: Vec<u32> = s[from..(from + n).min(s.len())].to_vec();
            s.extend_from_slice(&span);
            s.extend(random_stream(&mut rng, 10, 5000));
        }
        check_every_prefix(&s, CopyIndex::new(), &format!("planted round {round}"));
    }
}

#[test]
fn longest_equals_the_scan_on_periodic_streams() {
    for period in 1..=7u32 {
        let s: Vec<u32> = (0..180).map(|i| i % period).collect();
        check_every_prefix(&s, CopyIndex::new(), &format!("period {period}"));
    }
}

#[test]
fn collisions_never_hide_or_invent_a_source() {
    // A two-bit key mask: four buckets per window table, so nearly every
    // insertion collides with a different content.
    let mut rng = Rng::new(23);
    for round in 0..60 {
        let symbols = 2 + rng.below(30);
        let s = random_stream(&mut rng, 250, symbols);
        check_every_prefix(
            &s,
            CopyIndex::with_params(1 << 20, 0b11),
            &format!("collision round {round}"),
        );
    }
}

#[test]
fn the_latest_earlier_occurrence_wins_ties() {
    // The suffix [7, 8] (window 2) occurs earlier at ends 1, 5 and 10: the
    // latest earlier one (end 10) is the source, never the earliest.
    let s = [7, 8, 1, 2, 7, 8, 3, 4, 5, 7, 8, 6, 7, 8];
    let mut idx = CopyIndex::new();
    idx.sync(&s, None);
    let m = idx.longest().expect("a source");
    assert_eq!(WINDOWS[m.window], 2);
    assert_eq!(m.end, 10);
    assert_eq!(Some(m), brute_longest(&s, 0));
}

#[test]
fn copies_the_continuation_of_the_latest_earlier_match() {
    // TensorFold's own fixture: 0..19 twice. The 16-token suffix occurred
    // once before, ending at 19; the match extends to all 20 tokens.
    let s: Vec<u32> = (0..40).map(|i| i % 20).collect();
    let mut idx = CopyIndex::new();
    idx.sync(&s, None);
    let m = idx.longest().expect("a source");
    assert_eq!((WINDOWS[m.window], m.end, m.n), (16, 19, 20));
    let mut out = Vec::new();
    idx.chain(&m, 15, &mut out);
    assert_eq!(out, (0..15).collect::<Vec<u32>>());
    idx.chain(&m, 0, &mut out);
    assert!(out.is_empty());
}

#[test]
fn continuation_repeats_with_the_period_past_the_anchor() {
    let mut rng = Rng::new(5);
    for _ in 0..200 {
        let len = 2 + rng.below(40) as usize;
        let s = random_stream(&mut rng, len, 6);
        let mut idx = CopyIndex::new();
        idx.sync(&s, None);
        let next = rng.below(len as u64) as usize;
        let want = rng.below(20) as usize;
        let mut out = Vec::new();
        idx.continuation(next, want, &mut out);
        assert_eq!(out, brute_continuation(&s, next, want));
    }
    // `a b a b`: the source two back continues `a b a b ...` forever.
    let s = [1, 2, 1, 2];
    let mut idx = CopyIndex::new();
    idx.sync(&s, None);
    let mut out = Vec::new();
    idx.continuation(2, 6, &mut out);
    assert_eq!(out, [1, 2, 1, 2, 1, 2]);
}

#[test]
fn a_rewritten_tail_indexes_like_a_fresh_stream() {
    let mut rng = Rng::new(31);
    for _ in 0..40 {
        let a = random_stream(&mut rng, 120, 3);
        let mut b = a[..60 + rng.below(50) as usize].to_vec();
        b.extend(random_stream(&mut rng, 40, 3));
        let mut idx = CopyIndex::new();
        idx.sync(&a, None);
        assert_eq!(idx.sync(&b, None), SyncOutcome::Rebuilt);
        let mut fresh = CopyIndex::new();
        fresh.sync(&b, None);
        assert_eq!(idx.longest(), fresh.longest());
        assert_eq!(idx.longest(), brute_longest(&b, 0));
        // A plain truncation is a rewrite too.
        assert_eq!(idx.sync(&b[..30], None), SyncOutcome::Rebuilt);
        assert_eq!(idx.longest(), brute_longest(&b[..30], 0));
    }
}

#[test]
fn head_plus_last_is_the_committed_stream() {
    // The scheduler passes `seq.tokens` and the sampled `last_token`.
    let s: Vec<u32> = (0..50).map(|i| (i * 7 % 11) as u32).collect();
    let mut a = CopyIndex::new();
    let mut b = CopyIndex::new();
    for k in 1..s.len() {
        let out_a = a.sync(&s[..k], Some(s[k]));
        let out_b = b.sync(&s[..=k], None);
        assert_eq!(out_a, out_b);
        assert_eq!(a.tokens(), b.tokens());
        assert_eq!(a.longest(), b.longest());
    }
}

#[test]
fn the_bounded_index_drops_the_oldest_ends_only() {
    let mut rng = Rng::new(41);
    let s = random_stream(&mut rng, 2000, 4);
    let cap = 64;
    let mut idx = CopyIndex::with_params(cap, u64::MAX);
    for k in 0..=s.len() {
        idx.sync(&s[..k], None);
        assert_eq!(
            idx.longest(),
            brute_longest(&s[..k], idx.floor()),
            "prefix {k}"
        );
        assert!(
            idx.floor() + cap >= k.saturating_sub(1),
            "floor too high at {k}"
        );
        // Live window ends are at most cap plus the uncompacted quarter.
        assert!(
            idx.entries() <= WINDOWS.len() * (cap + cap / 4 + 1),
            "entries {} at prefix {k}",
            idx.entries()
        );
    }
    // A fresh long history starts at the horizon instead of indexing all.
    let mut fresh = CopyIndex::with_params(cap, u64::MAX);
    fresh.sync(&s, None);
    assert_eq!(fresh.longest(), brute_longest(&s, fresh.floor()));
}
