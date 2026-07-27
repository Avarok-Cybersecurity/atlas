// SPDX-License-Identifier: AGPL-3.0-only
//
// `carve_disk_slots` precedence pins — moved to their own file (SDD test
// convention) when `cache_peer.rs` split into `cache_peer/`; body unchanged.

use super::carve_disk_slots;

/// cross-KIND pin against the REAL registry (no RDMA needed — anon
/// mmap + O_DIRECT swap file only): `(kind, blob_bytes)` keying gives SSM
/// (kind 0) and KV (kind 1) arenas their OWN residency map and their OWN
/// swap file even with IDENTICAL blob_bytes — so a numerically equal wire
/// key can never be looked up across kinds. This is the structural guarantee
/// the KV client's `KV_DOMAIN` namespace fold is defense-in-depth on top of;
/// a refactor that collapses the registry keying fails here. Same-kind
/// re-connects share the ONE arena (capacity pooling).
#[test]
fn cross_kind_arenas_are_disjoint_in_the_real_registry() {
    // Real-filesystem dir (tmpfs/overlay EINVALs on O_DIRECT — skip like
    // atlas-tier's direct_swap tests so containerized CI doesn't break).
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/atlas-kv-paging-registry-test");
    std::fs::create_dir_all(&dir).unwrap();
    let rdma = super::RdmaConfig {
        swap_dir: Some(dir.clone()),
        ..Default::default()
    };
    let ledger = std::sync::Arc::new(crate::blade_cap::CommitLedger::new(0));
    // blob_bytes unique to this test (the registry is process-global) and a
    // 4 KiB multiple (O_DIRECT record contract).
    let blob = 8192usize;
    let ssm = match super::get_or_init_shared_paging(&rdma, 0, 4 * blob, blob, &ledger) {
        Ok(sh) => sh,
        Err(e) => {
            eprintln!("skipping registry test (filesystem refused O_DIRECT): {e:#}");
            return;
        }
    };
    let kv = super::get_or_init_shared_paging(&rdma, 1, 4 * blob, blob, &ledger).unwrap();
    assert!(
        !std::sync::Arc::ptr_eq(&ssm, &kv),
        "same blob_bytes, different kind ⇒ different arenas"
    );
    // Same (kind, blob) ⇒ the ONE shared arena.
    let kv2 = super::get_or_init_shared_paging(&rdma, 1, 4 * blob, blob, &ledger).unwrap();
    assert!(std::sync::Arc::ptr_eq(&kv, &kv2));
    // Per-kind swap files exist, named atlas-snap-{kind}-{blob}.swap.
    assert!(dir.join(format!("atlas-snap-0-{blob}.swap")).exists());
    assert!(dir.join(format!("atlas-snap-1-{blob}.swap")).exists());
    // A key resident in the KV arena is INVISIBLE to the SSM arena.
    let key = 0x4B56_4B56_4B56_4B56u64;
    kv.residency
        .lock()
        .unwrap()
        .put_blob(key, &vec![0x4B; blob])
        .unwrap();
    assert_eq!(ssm.residency.lock().unwrap().locate(key).unwrap(), None);
    assert!(kv.residency.lock().unwrap().locate(key).unwrap().is_some());
}

#[test]
fn carve_disk_slots_precedence() {
    let bb = 4u64; // tiny blob
    // Per-kind override: fixed budget, shared remainder UNTOUCHED (no starve).
    let (slots, rem) = carve_disk_slots(Some(40), 100, 100, bb);
    assert_eq!(slots, 10);
    assert_eq!(
        rem, 100,
        "per-kind override must not consume the shared remainder"
    );
    // Per-kind 0 = unbounded for that kind, remainder untouched.
    assert_eq!(carve_disk_slots(Some(0), 100, 100, bb), (0, 100));
    // No override, shared cap set: claim the remainder (and it drops).
    let (slots, rem) = carve_disk_slots(None, 100, 100, bb);
    assert_eq!(slots, 25);
    assert_eq!(rem, 0, "shared carve consumes the remainder");
    // No override, shared cap 0 = unbounded.
    assert_eq!(carve_disk_slots(None, 0, 0, bb), (0, 0));
    // Starved shared remainder still floors at 1 record (never 0=unbounded).
    assert_eq!(carve_disk_slots(None, 100, 0, bb), (1, 0));
}
