// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracle for [`super::legacy::LegacyLookup`]: it must
//! return the token today's `NgramProposer::propose` scan returns (the scan
//! itself is the oracle, unchanged in `ngram.rs`), EARLIEST match included,
//! for every prefix of random, periodic and repetitive streams, and across the
//! history edits the n-gram mode makes (a rejected draft pops one token, a new
//! request replaces the history).

use super::index_tests::Rng;
use super::legacy::LegacyLookup;
use crate::ngram::NgramProposer;

const MIN: usize = 2;
const MAX: usize = 16;

fn scan(h: &[u32]) -> Option<u32> {
    NgramProposer::new(4).propose(h)
}

#[test]
fn the_old_tests_through_the_index() {
    let mut l = LegacyLookup::new();
    assert_eq!(l.propose(&[1, 2, 3, 4, 5, 1, 2, 3], MIN, MAX), Some(4));
    assert_eq!(l.propose(&[1, 2, 3, 4, 5, 6, 7, 8], MIN, MAX), None);
    assert_eq!(l.propose(&[1, 2], MIN, MAX), None);
    assert_eq!(
        l.propose(&[10, 20, 30, 10, 20, 30, 10, 20], MIN, MAX),
        Some(30)
    );
}

#[test]
fn equals_the_scan_on_every_prefix() {
    let mut rng = Rng::new(3);
    for round in 0..150 {
        let symbols = 1 + rng.below(6);
        let len = rng.below(260) as usize;
        let s: Vec<u32> = (0..len).map(|_| rng.below(symbols) as u32).collect();
        let mut l = LegacyLookup::new();
        for k in 0..=s.len() {
            assert_eq!(
                l.propose(&s[..k], MIN, MAX),
                scan(&s[..k]),
                "round {round} prefix {k}"
            );
        }
    }
}

#[test]
fn the_earliest_occurrence_wins_like_the_scan() {
    // [5, 6] occurs at 0 (followed by 1) and at 3 (followed by 2): the scan
    // takes the EARLIEST, so the proposal is 1, not 2.
    let h = [5, 6, 1, 5, 6, 2, 9, 5, 6];
    let mut l = LegacyLookup::new();
    assert_eq!(scan(&h), Some(1));
    assert_eq!(l.propose(&h, MIN, MAX), Some(1));
}

#[test]
fn equals_the_scan_on_periodic_and_long_repeats() {
    for period in 1..=9u32 {
        let s: Vec<u32> = (0..120).map(|i| i % period).collect();
        let mut l = LegacyLookup::new();
        for k in 0..=s.len() {
            assert_eq!(
                l.propose(&s[..k], MIN, MAX),
                scan(&s[..k]),
                "period {period} prefix {k}"
            );
        }
    }
}

#[test]
fn equals_the_scan_across_pops_and_new_requests() {
    let mut rng = Rng::new(17);
    let mut l = LegacyLookup::new();
    for round in 0..60 {
        // A new request: a fresh history, sometimes sharing a prefix with
        // the previous one (the same prompt sent twice).
        let mut h: Vec<u32> = (0..20 + rng.below(40))
            .map(|_| rng.below(4) as u32)
            .collect();
        for step in 0..120 {
            assert_eq!(
                l.propose(&h, MIN, MAX),
                scan(&h),
                "round {round} step {step}"
            );
            match rng.below(4) {
                // Accepted draft: two tokens appended.
                0 => {
                    h.push(rng.below(4) as u32);
                    h.push(rng.below(4) as u32);
                }
                // Rejected draft: the verify pushed one, then popped it.
                1 => {
                    h.push(rng.below(4) as u32);
                    assert_eq!(l.propose(&h, MIN, MAX), scan(&h));
                    h.pop();
                }
                _ => h.push(rng.below(4) as u32),
            }
        }
        if rng.below(2) == 0 {
            let keep = rng.below(h.len() as u64) as usize;
            h.truncate(keep);
            assert_eq!(
                l.propose(&h, MIN, MAX),
                scan(&h),
                "round {round} truncated to {keep}"
            );
        }
    }
    assert!(l.held() <= 1000);
}

#[test]
fn propose_next_keeps_the_scan_with_the_drafter_off() {
    // The process-wide copy_draft row is off in a test binary, so the
    // n-gram mode's entry point must be exactly the scan.
    assert!(!super::active());
    let mut p = NgramProposer::new(4);
    let mut rng = Rng::new(29);
    let s: Vec<u32> = (0..200).map(|_| rng.below(3) as u32).collect();
    for k in 0..=s.len() {
        assert_eq!(p.propose_next(&s[..k]), scan(&s[..k]));
    }
}
