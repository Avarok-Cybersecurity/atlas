// SPDX-License-Identifier: AGPL-3.0-only

//! §4 unification (ATLAS_SSM_TIER_UNIFIED) tests, including the default-OFF
//! regression guard (moved from the pre-split ssm_tier.rs).

use super::super::{MockSnapshotTransport, RdmaSnapshotStore};
use super::*;

// Fixed blob size + a bounded-store helper, duplicated from arena_store_tests:
// the default-OFF guard exercises the bounded store's drop-on-full policy.
const BLOB: usize = 4;
fn rdma_store(slots: usize) -> RdmaSnapshotStore {
    let t = Box::new(MockSnapshotTransport::new(slots * BLOB));
    RdmaSnapshotStore::new(t, BLOB, slots)
}

#[test]
fn unified_flag_truthy_parse_matches_hss_style() {
    for on in ["1", "true", "on", "yes", " 1 ", "yes "] {
        assert!(unified_flag_truthy(Some(on)), "{on:?} must engage the flag");
    }
    for off in ["", "0", "false", "off", "no", "TRUE", "2"] {
        assert!(!unified_flag_truthy(Some(off)), "{off:?} must stay off");
    }
    assert!(!unified_flag_truthy(None), "unset = default OFF");
}

fn unified_store(slots: usize) -> UnifiedSnapshotStore {
    UnifiedSnapshotStore::new(
        Box::new(atlas_tier::VecSlotArena::new(BLOB, slots)),
        Box::new(atlas_tier::MemSwapStore::new(BLOB)),
        BLOB,
    )
    .unwrap()
}

/// THE §4 fix: where the bounded stores FIFO-evict or drop-on-full, the
/// unified store never rejects — overflow LRU-spills to the swap tier and
/// every key faults back byte-identical.
#[test]
fn unified_store_never_rejects_and_faults_back() {
    let s = unified_store(2);
    for k in 0..32u64 {
        assert!(
            s.put(k, &[k as u8; BLOB]).unwrap(),
            "put {k} must never be refused"
        );
    }
    assert_eq!(s.len(), 32, "all keys tracked — nothing dropped");
    let mut o = [0u8; BLOB];
    for k in 0..32u64 {
        assert!(s.get(k, &mut o).unwrap(), "key {k} present");
        assert_eq!(o, [k as u8; BLOB], "key {k} byte-identical");
    }
    assert_eq!(s.stats.put_rejects.load(Ordering::Relaxed), 0);
}

/// LRU (not FIFO): touching the oldest-inserted key protects it — the
/// spill victim is the least-recently-USED key. (A capped MemBlobStore
/// would evict key 1 here; RdmaSnapshotStore would refuse key 3 outright.)
#[test]
fn unified_store_victim_is_lru_not_fifo_and_not_a_reject() {
    let s = unified_store(2);
    assert!(s.put(1, &[1; BLOB]).unwrap());
    assert!(s.put(2, &[2; BLOB]).unwrap());
    let mut o = [0u8; BLOB];
    assert!(s.get(1, &mut o).unwrap()); // touch 1 → 2 is now coldest
    assert!(s.put(3, &[3; BLOB]).unwrap(), "no drop-on-full");
    assert_eq!(s.bytes_resident(), 2 * BLOB, "two hot slots resident");
    // The hot-again key SURVIVED IN THE HOT TIER: getting key 1 is a
    // resident hit (no disk fault), where FIFO would have evicted it as
    // oldest-inserted; key 2 was the LRU spill victim and faults back.
    let faults0 = s.inner.lock().stats().faults_from_disk;
    assert!(s.get(1, &mut o).unwrap(), "hot-again key survives");
    assert_eq!(o, [1u8; BLOB]);
    assert_eq!(
        s.inner.lock().stats().faults_from_disk,
        faults0,
        "key 1 was still RESIDENT — the LRU victim was key 2, not the FIFO-oldest"
    );
    assert!(
        s.get(2, &mut o).unwrap(),
        "spilled key faults back, never dropped"
    );
    assert_eq!(o, [2u8; BLOB]);
    assert_eq!(
        s.inner.lock().stats().faults_from_disk,
        faults0 + 1,
        "key 2 came back via a disk fault"
    );
    assert!(s.get(3, &mut o).unwrap());
    assert_eq!(o, [3u8; BLOB]);
}

/// Read-pins are honored through the unified store: a read-pinned key can
/// never be chosen as the LRU spill victim while pinned (the peer's
/// mid-RDMA-READ guarantee survives the in-process adoption), and returns
/// to normal LRU rotation after the last unpin.
#[test]
fn unified_store_honors_read_pins() {
    let s = unified_store(2);
    assert!(s.put(1, &[1; BLOB]).unwrap());
    assert!(s.put(2, &[2; BLOB]).unwrap());
    s.inner.lock().pin_read(1);
    // Churn well past arena capacity: every spill victim must be a key
    // OTHER than the pinned one.
    for k in 10..20u64 {
        assert!(
            s.put(k, &[k as u8; BLOB]).unwrap(),
            "puts never rejected while pinned"
        );
    }
    let mut o = [0u8; BLOB];
    {
        let mut r = s.inner.lock();
        assert_eq!(r.read_pin_count(1), 1);
        let faults0 = r.stats().faults_from_disk;
        assert!(r.get_blob(1, &mut o).unwrap(), "pinned key present");
        assert_eq!(
            r.stats().faults_from_disk,
            faults0,
            "pinned key stayed RESIDENT through the churn — never spilled"
        );
        r.unpin_read(1);
        assert_eq!(r.read_pin_count(1), 0);
    }
    assert_eq!(o, [1u8; BLOB], "pinned key bytes intact");
    // After the last unpin the key is evictable again: more churn spills
    // it, and it faults back byte-identical (never dropped).
    for k in 20..30u64 {
        assert!(s.put(k, &[k as u8; BLOB]).unwrap());
    }
    let faults1 = s.inner.lock().stats().faults_from_disk;
    assert!(
        s.get(1, &mut o).unwrap(),
        "unpinned key spilled but never dropped"
    );
    assert_eq!(o, [1u8; BLOB]);
    assert_eq!(
        s.inner.lock().stats().faults_from_disk,
        faults1 + 1,
        "unpinned key was evicted normally and faulted back from swap"
    );
}

#[test]
fn unified_store_wrong_size_refused_gracefully() {
    let s = unified_store(2);
    assert!(
        !s.put(1, &[0; BLOB + 1]).unwrap(),
        "off-size put refused, not corrupt"
    );
    assert!(s.put(1, &[7; BLOB]).unwrap());
    let mut big = [0u8; BLOB + 4];
    assert!(
        !s.get(1, &mut big).unwrap(),
        "never scatter a wrong-sized blob"
    );
    assert_eq!(big, [0u8; BLOB + 4], "out untouched on refusal");
}

#[test]
fn unified_store_remove_is_clean_miss() {
    let s = unified_store(2);
    assert!(s.put(1, &[1; BLOB]).unwrap());
    s.remove(1);
    let mut o = [0u8; BLOB];
    assert!(!s.get(1, &mut o).unwrap());
    assert_eq!(s.len(), 0);
}

/// Unified over the SAME transport geometry the bounded RDMA store uses:
/// where `RdmaSnapshotStore` returns Ok(false) at slot 5, the unified wrap
/// keeps accepting (LRU spill to the swap tier) — the live §4 bug arm.
#[test]
fn unified_over_transport_never_drops_where_bounded_store_did() {
    const SLOTS: usize = 4;
    let hot = Box::new(TransportSlotArena {
        transport: Box::new(MockSnapshotTransport::new(SLOTS * BLOB)),
        slot_bytes: BLOB,
        num_slots: SLOTS,
    });
    let s = UnifiedSnapshotStore::new(hot, Box::new(atlas_tier::MemSwapStore::new(BLOB)), BLOB)
        .unwrap();
    let mut o = [0u8; BLOB];
    for k in 0..16u64 {
        assert!(
            s.put(k, &[k as u8; BLOB]).unwrap(),
            "arena-full put {k} accepted"
        );
    }
    for k in 0..16u64 {
        assert!(s.get(k, &mut o).unwrap(), "key {k} recoverable");
        assert_eq!(o, [k as u8; BLOB]);
    }
}

/// DEFAULT-OFF byte/behavior identity: with the flag unset the selectors
/// construct exactly today's stores with today's policies — the bounded
/// arena still drop-on-fulls here, the FIFO MemBlobStore still evicts
/// oldest-inserted (`cap_evicts_fifo` in store_tests), and `build_tier_store`
/// still yields the unbounded host-RAM store
/// (`build_tier_store_defaults_to_host_ram_unbounded` in selectors_tests).
#[test]
fn unified_flag_default_off_preserves_todays_policies() {
    // Pure-logic half: holds regardless of the ambient environment.
    assert!(!unified_flag_truthy(None), "absent env ⇒ flag OFF");
    // Env-dependent half. This is the §4 default-OFF regression guard, so it
    // must NEVER pass vacuously: fail loudly rather than skip when the flag is
    // exported into the test environment.
    assert!(
        std::env::var_os("ATLAS_SSM_TIER_UNIFIED").is_none(),
        "ATLAS_SSM_TIER_UNIFIED is set in the test environment — unset it. This test is \
         the default-OFF regression guard for the TIERED-CACHE-CONSOLIDATION §4 fix; \
         skipping it would green-light a change to the default path. To exercise the \
         flag-ON arms, run the flag-ON tests selectively instead of exporting the var \
         across the whole suite."
    );
    assert!(!ssm_tier_unified(), "flag must default OFF");
    let s = rdma_store(1);
    assert!(s.put(1, &[1; BLOB]).unwrap());
    assert!(
        !s.put(2, &[2; BLOB]).unwrap(),
        "flag OFF: drop-on-full unchanged"
    );
}
