// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the graph-family cache state machine (`graph_family.rs`), run
//! against the fake driver in `graph_family_fake.rs`. Two proof obligations:
//!
//! * **Bit identity on the host side.** After every step, each cached key's
//!   executable bakes THAT key's parameters (an updated executable carries the
//!   new capture, never the victim's), no template or executable leaks, and
//!   nothing is destroyed twice.
//! * **LRU order unchanged.** Driven side by side with the row-off reference
//!   (`decode_graph_key::lru_insert_graph`, the hit-path tick bump of
//!   `decode_a2.rs` / `verify_e.rs`), the cache holds the same keys with the
//!   same ticks after every step, hits and misses included.

use std::collections::{HashMap, HashSet};

use spark_runtime::gpu::GraphHandle;

use super::super::decode_graph_key::lru_insert_graph;
use super::{Finish, LruGraphCache, lru_victim, prune_signatures};
use crate::port::graph_stats::GraphCounts;

#[path = "graph_family_fake.rs"]
mod fake;
use fake::{Fake, Family, UpdateMode, check_invariants, fill, ticks, touch};

#[test]
fn below_the_cap_a_miss_instantiates_and_never_updates() {
    let f = Fake::new();
    let mut fam = Family::new(3);
    fam.lookup(&[0, 1]);
    let (g, how) = fam.miss(&f, &[0, 1], 2).unwrap();
    assert_eq!(how, Finish::Instantiated);
    assert_ne!(g.0, 0);
    let calls = f.take_calls();
    assert_eq!(calls, ["close", "topology", "instantiate", "drop_template"]);
    fam.lookup(&[2, 3]);
    fam.miss(&f, &[2, 3], 2).unwrap();
    assert!(
        !f.take_calls().contains(&"update"),
        "cached keys keep theirs"
    );
    let c = fam.counts();
    let seen = (c.captures, c.instantiations, c.updates, c.evictions);
    assert_eq!(seen, (2, 2, 0, 0));
    check_invariants(&f, &fam);
}

#[test]
fn a_full_cache_updates_the_same_topology_lru_victim_in_place() {
    let f = Fake::new();
    let mut fam = Family::new(2);
    fill(&f, &mut fam, &[&[0, 1], &[2, 3]], 2);
    let victim = fam.cache.0[&vec![0, 1]].0;
    // Touch [2, 3] so [0, 1] is the LRU.
    assert!(fam.lookup(&[2, 3]).is_some());
    assert!(fam.lookup(&[4, 5]).is_none());
    let (g, how) = fam.miss(&f, &[4, 5], 2).unwrap();
    assert_eq!(how, Finish::Updated);
    assert_eq!(
        g.0, victim.0,
        "the victim's executable moved to the new key"
    );
    let calls = f.take_calls();
    assert_eq!(calls, ["close", "topology", "update", "drop_template"]);
    assert_eq!(fam.keys(), HashSet::from([vec![2, 3], vec![4, 5]]));
    let c = fam.counts();
    let seen = (c.captures, c.instantiations, c.updates, c.update_failures);
    assert_eq!(seen, (3, 2, 1, 0));
    assert_eq!(c.evictions, 1);
    check_invariants(&f, &fam);
}

#[test]
fn a_different_topology_victim_is_destroyed_and_the_capture_instantiated() {
    let f = Fake::new();
    let mut fam = Family::new(2);
    // The width-3 entry goes in first, so it is the LRU victim.
    fill(&f, &mut fam, &[&[2, 3, 4]], 3);
    fill(&f, &mut fam, &[&[0, 1]], 2);
    fam.lookup(&[5, 6]);
    let (_, how) = fam.miss(&f, &[5, 6], 2).unwrap();
    assert_eq!(how, Finish::Instantiated);
    let calls = f.take_calls();
    assert!(!calls.contains(&"update"), "{calls:?}");
    assert!(calls.contains(&"drop_exec"), "{calls:?}");
    assert_eq!(fam.keys(), HashSet::from([vec![0, 1], vec![5, 6]]));
    assert_eq!(fam.counts().evictions, 1);
    check_invariants(&f, &fam);
}

#[test]
fn a_refused_or_failed_update_falls_back_to_instantiate_and_evict() {
    for mode in [UpdateMode::Refuse, UpdateMode::Fail] {
        let f = Fake::new();
        let mut fam = Family::new(2);
        fill(&f, &mut fam, &[&[0, 1], &[2, 3]], 2);
        f.update.set(mode);
        fam.lookup(&[4, 5]);
        let (_, how) = fam.miss(&f, &[4, 5], 2).unwrap();
        assert_eq!(how, Finish::Instantiated, "{mode:?}");
        let calls = f.take_calls();
        let want = [
            "close",
            "topology",
            "update",
            "instantiate",
            "drop_template",
            "drop_exec",
        ];
        assert_eq!(calls, want, "{mode:?}");
        let keys = HashSet::from([vec![2, 3], vec![4, 5]]);
        assert_eq!(fam.keys(), keys, "{mode:?}");
        let c = fam.counts();
        let seen = (c.update_failures, c.updates, c.evictions);
        assert_eq!(seen, (1, 0, 1), "{mode:?}");
        check_invariants(&f, &fam);
    }
}

#[test]
fn an_unknown_victim_signature_is_never_updated() {
    let f = Fake::new();
    let mut fam = Family::new(1);
    fill(&f, &mut fam, &[&[0, 1]], 2);
    fam.sigs.clear();
    fam.lookup(&[2, 3]);
    let (_, how) = fam.miss(&f, &[2, 3], 2).unwrap();
    assert_eq!(how, Finish::Instantiated);
    assert!(!f.take_calls().contains(&"update"));
    check_invariants(&f, &fam);
}

#[test]
fn a_failed_signature_instantiates_and_records_no_class() {
    let f = Fake::new();
    let mut fam = Family::new(1);
    fill(&f, &mut fam, &[&[0, 1]], 2);
    f.topology_fails.set(true);
    // A stale entry for the handle value the next executable will get (the
    // template takes the next id, the executable the one after).
    let next_exec = f.next_id.get() + 1;
    fam.sigs.insert(next_exec, 99);
    fam.lookup(&[2, 3]);
    let (g, how) = fam.miss(&f, &[2, 3], 2).unwrap();
    assert_eq!(how, Finish::Instantiated);
    assert_eq!(g.0, next_exec);
    assert!(!fam.sigs.contains_key(&g.0), "no class from a dead handle");
    assert!(!f.take_calls().contains(&"update"));
    check_invariants(&f, &fam);
}

#[test]
fn a_failed_instantiate_destroys_the_template_and_leaves_the_cache_alone() {
    let f = Fake::new();
    let mut fam = Family::new(1);
    fill(&f, &mut fam, &[&[0, 1]], 2);
    f.update.set(UpdateMode::Refuse);
    f.instantiate_fails.set(true);
    fam.lookup(&[2, 3]);
    assert!(fam.miss(&f, &[2, 3], 2).is_err());
    assert_eq!(fam.keys(), HashSet::from([vec![0, 1]]));
    check_invariants(&f, &fam);
}

#[test]
fn a_null_executable_from_a_real_capture_is_an_error() {
    let f = Fake::new();
    let mut fam = Family::new(2);
    f.instantiate_null.set(true);
    fam.lookup(&[0, 1]);
    assert!(fam.miss(&f, &[0, 1], 2).is_err());
    assert!(fam.cache.0.is_empty());
    assert!(f.templates.borrow().is_empty());
}

#[test]
fn a_null_capture_installs_nothing_and_counts_nothing() {
    let f = Fake::new();
    let mut fam = Family::new(2);
    f.null_capture.set(true);
    fam.lookup(&[0, 1]);
    let (g, how) = fam.miss(&f, &[0, 1], 2).unwrap();
    assert_eq!((g.0, how), (0, Finish::Empty));
    assert!(fam.cache.0.is_empty());
    assert_eq!(fam.counts(), GraphCounts::default());
}

#[test]
fn the_victim_is_the_entry_lru_insert_would_evict() {
    let mut cache: LruGraphCache = (HashMap::new(), 0);
    for (i, k) in [[0u32], [1], [2]].iter().enumerate() {
        lru_insert_graph(&mut cache, 3, k.to_vec(), GraphHandle(i as u64 + 1));
    }
    touch(&mut cache, &[0]);
    assert_eq!(lru_victim(&cache, 3, &[9]), Some(vec![1]));
    assert_eq!(lru_victim(&cache, 4, &[9]), None, "below the cap");
    assert_eq!(
        lru_victim(&cache, 3, &[2]),
        None,
        "a cached key evicts nothing"
    );
    let dropped = lru_insert_graph(&mut cache, 3, vec![9], GraphHandle(9));
    assert_eq!(dropped.iter().map(|g| g.0).collect::<Vec<_>>(), [2]);
}

#[test]
fn the_signature_table_is_pruned_to_live_executables() {
    let f = Fake::new();
    let mut fam = Family::new(1);
    for dead in 1..=10u64 {
        fam.sigs.insert(dead, 2);
    }
    fam.lookup(&[0, 1]);
    let (g, _) = fam.miss(&f, &[0, 1], 2).unwrap();
    assert_eq!(fam.sigs.keys().copied().collect::<Vec<_>>(), [g.0]);
    // At or below twice the cap nothing is pruned.
    let mut small: HashMap<u64, u64> = HashMap::from([(1, 1), (2, 2)]);
    prune_signatures(&mut small, &fam.cache, 1);
    assert_eq!(small.len(), 2);
}

/// One pseudo-random generator per seed (a 64-bit LCG; high bits out).
fn lcg(seed: u64) -> impl FnMut() -> u32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    }
}

/// The oracle: thousands of steps over a churning key pool with two topology
/// classes, the row-off reference and the family cache side by side, with
/// refused and failed updates mixed in. Same hits, same misses, same keys and
/// ticks after every step; every key replays its own capture; and every miss
/// is exactly one instantiation or one update.
#[test]
fn randomized_family_matches_the_row_off_cache_step_for_step() {
    for (seed, cap) in [(1u64, 5usize), (2, 1), (3, 8), (4, 3)] {
        let mut next = lcg(seed);
        let f = Fake::new();
        let mut fam = Family::new(cap);
        let mut reference: LruGraphCache = (HashMap::new(), 0);
        let mut misses = 0u64;
        for step in 0..3000u64 {
            // 12 keys; widths 2 and 3 are the two topology classes.
            let k = next() % 12;
            let key: Vec<u32> = (k..k + 2 + u32::from(k >= 8)).collect();
            let sig = key.len() as u64;
            f.update.set(match next() % 10 {
                0 => UpdateMode::Refuse,
                1 => UpdateMode::Fail,
                _ => UpdateMode::Apply,
            });
            let ref_hit = touch(&mut reference, &key).is_some();
            let fam_hit = fam.lookup(&key);
            assert_eq!(ref_hit, fam_hit.is_some(), "seed {seed} step {step}");
            if let Some(g) = fam_hit {
                let execs = f.execs.borrow();
                assert_eq!(execs[&g.0].content, key, "a hit replays its own key");
            } else {
                misses += 1;
                let h = GraphHandle(misses);
                lru_insert_graph(&mut reference, cap, key.clone(), h);
                fam.miss(&f, &key, sig).unwrap();
            }
            assert_eq!(ticks(&reference), fam.ticks(), "seed {seed} step {step}");
            assert_eq!(reference.1, fam.cache.1);
            check_invariants(&f, &fam);
        }
        let c = fam.counts();
        assert_eq!(c.captures, misses);
        assert_eq!(c.instantiations + c.updates, misses, "seed {seed}");
        assert!(c.updates > 0, "seed {seed}: churn met same-class victims");
        assert!(
            c.instantiations < misses,
            "seed {seed}: updates replaced some"
        );
        f.take_calls();
    }
}
