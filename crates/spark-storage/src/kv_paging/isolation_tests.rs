// SPDX-License-Identifier: AGPL-3.0-only

//! KV paging isolation pins — hardware-free, mirroring spark-model's
//! `paging_isolation_tests.rs`. The "peer" here is the EXACT residency type
//! the real peer's `SharedPaging` owns (`atlas_tier::Residency`, via the
//! `snapshot_swap` re-export), keyed purely by the u64 wire key — so these
//! tests prove the client-side namespace fold is the only thing standing
//! between two models (or two same-model clients) and silent KV cross-serve,
//! and that it stands.

use std::collections::HashMap;
use std::num::NonZeroU64;

use super::ns::{derive_kv_ns, wire_key};
use crate::group::{GroupKey, GroupLayout, KvKind};
use crate::snapshot_swap::{MemSwapStore, Residency, VecSlotArena};

/// Tiny fake block for the mock peer (the real block_bytes is layout-derived;
/// isolation semantics don't depend on the size).
const BB: usize = 64;

fn layout() -> GroupLayout {
    // 4 layers × 16 blocks × 2 kv_heads, bs 16, hd 128, BF16.
    GroupLayout::new(4, 16, 2, 16, 128, 2, 4096)
}

type MockPeer = Residency<VecSlotArena, MemSwapStore>;

/// A shared peer arena: `slots` hot slots over an UNBOUNDED in-memory swap —
/// the miss-proof (`--swap-cap-gb-kv 0`) configuration the KV kind requires.
fn peer(slots: usize) -> MockPeer {
    Residency::new(VecSlotArena::new(BB, slots), MemSwapStore::new(BB)).unwrap()
}

/// One simulated KV paging client: its namespace, folding keys exactly as
/// `KvPagingBackend::block_key` does (base dense group id → wire_key).
struct Client {
    ns: NonZeroU64,
}

impl Client {
    fn new(fp: u64, salt: u64) -> Self {
        Self {
            ns: derive_kv_ns(fp, &layout(), 2, 16, 128, salt),
        }
    }
    fn key(&self, layer: u32, block: u32) -> u64 {
        let base = layout()
            .group_id(GroupKey::new(layer, block, 0, KvKind::K))
            .0;
        wire_key(self.ns, base)
    }
    fn put(&self, peer: &mut MockPeer, layer: u32, block: u32, tag: u8) {
        peer.put_blob(self.key(layer, block), &[tag; BB]).unwrap();
    }
    fn get(&self, peer: &mut MockPeer, layer: u32, block: u32) -> Option<Vec<u8>> {
        let mut out = vec![0u8; BB];
        peer.get_blob(self.key(layer, block), &mut out)
            .unwrap()
            .then_some(out)
    }
}

const FP_A: u64 = 0x5629_922c_51a1_6a10; // spark-model's pinned hybrid fp
const FP_B: u64 = 0x971e_b3b4_bd13_22f1; // spark-model's pinned dense fp
const SALT: u64 = 0x0BAD_5EED;

// ── T1: two MODELS, same client-local (layer, block), one shared peer ────

#[test]
fn two_models_do_not_cross_serve() {
    let mut peer = peer(8);
    let a = Client::new(FP_A, SALT);
    let b = Client::new(FP_B, SALT);
    a.put(&mut peer, 1, 3, 0xAA);
    assert_eq!(
        b.get(&mut peer, 1, 3),
        None,
        "model B must MISS model A's KV block for the same (layer, block)"
    );
    // …while model A still hits its own entry (isolation, not amnesia).
    assert_eq!(a.get(&mut peer, 1, 3), Some(vec![0xAA; BB]));
}

// ── T2 (the KV-specific hazard): SAME model, two CLIENTS. GroupKey.block is
// a client-local pool index, so without the salt this would be a cache HIT
// with the OTHER client's attention state — guaranteed corruption, silent.
// The same mechanism models a client RESTART (fresh salt ⇒ its own stale
// pre-restart blocks become unreachable instead of being served back). ──────

#[test]
fn same_model_two_salts_do_not_cross_serve() {
    let mut peer = peer(8);
    let c1 = Client::new(FP_A, 0x1111);
    let c2 = Client::new(FP_A, 0x2222);
    assert_ne!(c1.ns, c2.ns, "the salt must separate same-model clients");
    c1.put(&mut peer, 0, 0, 0xC1);
    assert_eq!(
        c2.get(&mut peer, 0, 0),
        None,
        "same model + same block id + different client ⇒ MISS, never the other \
         client's bytes"
    );
    c2.put(&mut peer, 0, 0, 0xC2);
    // Both coexist on one peer (capacity pooling), each seeing only its own.
    assert_eq!(c1.get(&mut peer, 0, 0), Some(vec![0xC1; BB]));
    assert_eq!(c2.get(&mut peer, 0, 0), Some(vec![0xC2; BB]));
}

// ── T3: same model + same salt is DETERMINISTIC across derivations (a
// reconnect that pins ATLAS_KV_PAGING_SALT resumes its own keyspace). ───────

#[test]
fn same_model_same_salt_round_trips_across_instances() {
    let mut peer = peer(8);
    let c1 = Client::new(FP_A, SALT);
    let c2 = Client::new(FP_A, SALT); // independent derivation, same inputs
    assert_eq!(c1.ns, c2.ns);
    c1.put(&mut peer, 2, 5, 0x77);
    assert_eq!(c2.get(&mut peer, 2, 5), Some(vec![0x77; BB]));
}

// ── T4: cross-KIND. The peer's registry keys arenas by (kind, blob_bytes):
// each kind gets its OWN residency map + swap file, so an SSM key can never
// be looked up in the KV arena even if numerically equal AND blob_bytes
// coincide. Pin that structural guarantee (this must never collapse) —
// the KV_DOMAIN fold in the ns is defense-in-depth on top. ──────────────────

#[test]
fn cross_kind_registry_never_mixes() {
    // The registry shape: (kind, blob_bytes) → its own residency.
    let mut registry: HashMap<(u8, usize), MockPeer> = HashMap::new();
    registry.insert((0, BB), peer(4)); // SSM arena
    registry.insert((1, BB), peer(4)); // KV arena — same blob_bytes!
    let key = 0xDEAD_BEEF_DEAD_BEEFu64; // numerically equal in both keyspaces
    registry
        .get_mut(&(1, BB))
        .unwrap()
        .put_blob(key, &[0x4B; BB]) // "K"
        .unwrap();
    let mut out = vec![0u8; BB];
    assert!(
        !registry
            .get_mut(&(0, BB))
            .unwrap()
            .get_blob(key, &mut out)
            .unwrap(),
        "an SSM lookup must never see a KV entry, even with equal key AND blob_bytes"
    );
    assert!(
        registry
            .get_mut(&(1, BB))
            .unwrap()
            .get_blob(key, &mut out)
            .unwrap()
    );
    assert_eq!(out, vec![0x4B; BB]);
}

// ── T5: KV-shaped spill/fault byte-identity on a miss-proof peer. Far more
// blocks than hot slots ⇒ the residency MUST spill coldest to the (unbounded)
// swap and fault them back byte-identical — never a drop. This is the peer
// behavior the KV kind depends on under --swap-cap-gb-kv 0. ─────────────────

#[test]
fn kv_blocks_survive_spill_and_fault_byte_identical() {
    let mut peer = peer(4); // 4 hot slots
    let c = Client::new(FP_A, SALT);
    let pat = |layer: u32, block: u32| ((layer * 31 + block) & 0xFF) as u8;
    // 2 layers × 16 blocks = 32 blocks through 4 slots ⇒ 28 forced spills.
    for layer in 0..2u32 {
        for block in 0..16u32 {
            c.put(&mut peer, layer, block, pat(layer, block));
        }
    }
    for layer in 0..2u32 {
        for block in 0..16u32 {
            assert_eq!(
                c.get(&mut peer, layer, block),
                Some(vec![pat(layer, block); BB]),
                "block (L{layer},B{block}) must fault back byte-identical, never drop"
            );
        }
    }
    assert!(
        peer.stats().spills_to_disk > 0,
        "the test must force spills"
    );
    assert!(peer.stats().faults_from_disk > 0);
    assert_eq!(
        peer.stats().disk_evictions,
        0,
        "miss-proof: nothing dropped"
    );
}

// ── T6: overwrite-in-place (disk-id reuse). HighSpeedSwap recycles freed
// disk_block_ids, so the SAME wire key is re-PUT with new bytes; the peer's
// alloc→commit must overwrite, and a subsequent GET must see the NEW bytes. ─

#[test]
fn disk_id_reuse_overwrites_in_place() {
    let mut peer = peer(4);
    let c = Client::new(FP_A, SALT);
    c.put(&mut peer, 0, 7, 0x01);
    c.put(&mut peer, 0, 7, 0x02); // freed + reallocated id, new sequence data
    assert_eq!(c.get(&mut peer, 0, 7), Some(vec![0x02; BB]));
}
