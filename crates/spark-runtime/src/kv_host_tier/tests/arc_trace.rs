// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: ARC on scripted traces against the naive reference (`arc_ref.rs`): every
//! victim list, list membership and the adaptive target agree after every operation, on mixed
//! random traces and with pinned (rejected) keys; plus the textbook properties (scan
//! resistance, ghost hits adapt the target, bounded ghost lists).

use super::arc_ref::RefArc;
use crate::kv_host_tier::policy::Policy;
use crate::kv_host_tier::{BlockKey, TierPolicy};

fn key(i: u64) -> BlockKey {
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&i.to_le_bytes());
    BlockKey(b)
}

/// xorshift64*: a fixed, dependency-free stream for scripted traces.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn check_same(p: &Policy, r: &RefArc, universe: u64, step: usize) {
    let a = p.as_arc().expect("arc policy");
    assert_eq!(a.target(), r.p, "target p at step {step}");
    assert_eq!(
        a.sizes(),
        (r.t1.len(), r.t2.len(), r.b1.len(), r.b2.len()),
        "list sizes at step {step}"
    );
    for i in 0..universe {
        assert_eq!(
            a.where_is(&key(i)),
            r.where_is(&key(i)),
            "key {i} at step {step}"
        );
    }
}

fn run_trace(seed: u64, capacity: usize, universe: u64, steps: usize) {
    let mut p = Policy::new(TierPolicy::Arc, capacity, 1024);
    let mut r = RefArc::new(capacity);
    let mut rng = Rng(seed);
    let mut resident: Vec<u64> = Vec::new();
    for step in 0..steps {
        // Keys pinned this step (a lease or staging): rejected by the predicate.
        let pin_a = rng.below(universe);
        let pin_b = rng.below(universe);
        let ok = move |k: &BlockKey| *k != key(pin_a) && *k != key(pin_b);
        match rng.below(10) {
            // Admission of a key (new, or remembered in a ghost list).
            0..=3 => {
                let k = rng.below(universe);
                if !resident.contains(&k) {
                    p.insert_batch(&[key(k)]);
                    r.admit(key(k));
                    resident.push(k);
                }
            }
            // A batch admission, applied tail-first by the policy.
            4 => {
                let base = rng.below(universe);
                let batch: Vec<u64> = (0..3)
                    .map(|j| (base + j) % universe)
                    .filter(|k| !resident.contains(k))
                    .collect();
                let keys: Vec<BlockKey> = batch.iter().map(|&k| key(k)).collect();
                p.insert_batch(&keys);
                for k in keys.iter().rev() {
                    r.admit(*k);
                }
                resident.extend(batch);
            }
            // A hit on a resident key.
            5..=7 if !resident.is_empty() => {
                let k = resident[rng.below(resident.len() as u64) as usize];
                p.touch_batch(&[key(k)]);
                r.hit(key(k));
            }
            // A removal without an eviction decision (failed copy).
            8 if !resident.is_empty() => {
                let i = rng.below(resident.len() as u64) as usize;
                let k = resident.swap_remove(i);
                p.remove(&key(k));
                r.forget(&key(k));
            }
            // Watermark eviction of a small batch.
            _ => {
                let want = 1 + rng.below(3) as usize;
                let got = p.victims(want, &ok);
                let exp = r.victims(want, &ok);
                assert_eq!(got, exp, "victims at step {step}");
                for v in &got {
                    let id = u64::from_le_bytes(v.0[..8].try_into().unwrap());
                    resident.retain(|&x| x != id);
                    assert!(ok(v), "a rejected key was chosen at step {step}");
                }
            }
        }
        // Keep residents within capacity, as the byte budget does.
        while resident.len() > capacity {
            let got = p.victims(1, &ok);
            let exp = r.victims(1, &ok);
            assert_eq!(got, exp, "capacity eviction at step {step}");
            match got.first() {
                Some(v) => {
                    let id = u64::from_le_bytes(v.0[..8].try_into().unwrap());
                    resident.retain(|&x| x != id);
                }
                None => break,
            }
        }
        check_same(&p, &r, universe, step);
    }
}

#[test]
fn arc_matches_the_reference_on_random_traces() {
    for seed in [1u64, 7, 42, 0xDEAD_BEEF, 123_456_789] {
        run_trace(seed, 8, 24, 2_000);
    }
    run_trace(99, 1, 4, 500);
    run_trace(5, 32, 40, 3_000);
}

#[test]
fn arc_resists_a_scan() {
    let c = 8;
    let mut p = Policy::new(TierPolicy::Arc, c, 1024);
    let hot: Vec<BlockKey> = (0..4).map(key).collect();
    p.insert_batch(&hot);
    for _ in 0..3 {
        p.touch_batch(&hot);
    }
    // A one-pass scan of 40 cold keys, evicting to stay within capacity.
    for i in 100..140 {
        p.insert_batch(&[key(i)]);
        if p.len() > c {
            p.victims(p.len() - c, &|_: &BlockKey| true);
        }
    }
    let a = p.as_arc().unwrap();
    for k in &hot {
        assert_eq!(a.where_is(k), "t2", "a hot key was flushed by the scan");
    }
}

#[test]
fn a_ghost_hit_adapts_the_target() {
    let mut p = Policy::new(TierPolicy::Arc, 4, 1024);
    for i in 0..4 {
        p.insert_batch(&[key(i)]);
    }
    let gone = p.victims(2, &|_: &BlockKey| true);
    assert_eq!(gone, vec![key(0), key(1)]);
    assert_eq!(p.as_arc().unwrap().where_is(&key(0)), "b1");
    assert_eq!(p.as_arc().unwrap().target(), 0);
    // Re-admitting a key remembered in B1: T1 was too small, p grows, the key enters T2.
    p.insert_batch(&[key(0)]);
    let a = p.as_arc().unwrap();
    assert_eq!(a.target(), 1);
    assert_eq!(a.where_is(&key(0)), "t2");
}

#[test]
fn ghost_lists_stay_bounded() {
    let c = 4;
    let mut p = Policy::new(TierPolicy::Arc, c, 1024);
    for i in 0..1_000 {
        p.insert_batch(&[key(i)]);
        if i.is_multiple_of(3) {
            p.touch_batch(&[key(i)]);
        }
        if p.len() > c {
            p.victims(p.len() - c, &|_: &BlockKey| true);
        }
        let (t1, t2, b1, b2) = p.as_arc().unwrap().sizes();
        assert!(t1 + b1 <= c, "|T1| + |B1| <= c");
        assert!(t1 + t2 + b1 + b2 <= 2 * c, "total <= 2c");
    }
}
