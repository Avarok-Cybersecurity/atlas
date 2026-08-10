// SPDX-License-Identifier: AGPL-3.0-only

//! THE cross-model shared-peer regression tests — hardware-free.
//!
//! The paging peer owns ONE residency map keyed purely by the u64 wire key
//! the client sends, so the fingerprint-derived namespace folded by
//! [`PagingSnapshotStore::wire`] is the ONLY thing preventing two models from
//! silently serving each other's recurrent state for the same (model-
//! independent) `prefix_hash`. Before the fingerprint work this failure was
//! SILENT — a cross-model GET was a cache *hit* with the wrong bytes, the
//! worst failure mode in this system. These tests drive two stores with
//! different namespaces over ONE shared mock peer and pin both directions:
//! distinct namespaces isolate; equal namespaces (the old ns=0-passthrough /
//! shared-DECODE_DOMAIN defaults) collide.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use parking_lot::Mutex;

use super::super::fingerprint::{ModelFingerprint, mix64};
use super::super::{MockSnapshotTransport, PagingTransport, SnapshotBlobStore, SnapshotTransport};
use super::PagingSnapshotStore;

const BLOB: usize = 64;
/// A logical tier key. `prefix_hash` is computed over TOKENS only, so two
/// models given the same prompt derive this SAME key — model identity must
/// come from the namespace, nowhere else.
const K: u64 = 0x5EED_F00D_CAFE_D00D;

// ── Mock peer: byte-faithful to the shared paging peer's service ──
// ONE residency map (wire-key → slot) + ONE flat arena shared by every
// client store — exactly like production clients sharing one RAM blade.
// Models residency + the arena only (no LRU eviction / NVMe spill /
// read-pins: collision semantics don't depend on those).

// `pub(super)` so a later cross-client decode-salt regression module can drive
// the SAME byte-faithful mock peer instead of transcribing a second copy.
pub(super) struct MockPagingPeer {
    blob_bytes: usize,
    inner: Mutex<MockPeerInner>,
    arena: MockSnapshotTransport,
}

struct MockPeerInner {
    /// wire-key → slot: the peer-side `Residency.map` (`HashMap<u64, Loc>`).
    map: HashMap<u64, usize>,
    free: Vec<usize>,
}

impl MockPagingPeer {
    pub(super) fn new(blob_bytes: usize, slots: usize) -> Self {
        Self {
            blob_bytes,
            inner: Mutex::new(MockPeerInner {
                map: HashMap::new(),
                free: (0..slots).rev().collect(),
            }),
            arena: MockSnapshotTransport::new(blob_bytes * slots),
        }
    }
}

impl PagingTransport for Arc<MockPagingPeer> {
    fn paging_put(&self, key: u64, bytes: &[u8]) -> Result<()> {
        let slot = {
            let mut g = self.inner.lock();
            match g.map.get(&key) {
                Some(&s) => s,
                None => {
                    let s = g.free.pop().expect("mock peer full — size the test arena");
                    g.map.insert(key, s);
                    s
                }
            }
        };
        self.arena
            .write_blob((slot * self.blob_bytes) as u64, bytes)
    }
    fn paging_get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        let slot = match self.inner.lock().map.get(&key) {
            Some(&s) => s,
            None => return Ok(false),
        };
        self.arena.read_blob((slot * self.blob_bytes) as u64, out)?;
        Ok(true)
    }
    fn paging_remove(&self, key: u64) -> Result<()> {
        let mut g = self.inner.lock();
        if let Some(s) = g.map.remove(&key) {
            g.free.push(s);
        }
        Ok(())
    }
}

// ── Fixtures: two DIFFERENT models, fingerprints derived the real way ────

fn hybrid() -> ModelConfig {
    ModelConfig::qwen3_next_80b_nvfp4()
}

fn dense() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3".to_string();
    c.num_hidden_layers = 28;
    c.layer_types = vec![LayerType::FullAttention; 28];
    c.num_experts = 0;
    c.linear_num_key_heads = 0;
    c.linear_key_head_dim = 0;
    c.linear_num_value_heads = 0;
    c.linear_value_head_dim = 0;
    c
}

fn ns_of(cfg: &ModelConfig) -> NonZeroU64 {
    ModelFingerprint::derive_with_id(cfg, BLOB, "")
        .unwrap()
        .nonzero()
}

fn store(peer: &Arc<MockPagingPeer>, ns: NonZeroU64) -> PagingSnapshotStore {
    PagingSnapshotStore::new(Box::new(peer.clone()), BLOB, ns)
}

// ── T1 (the headline): distinct fingerprints do not cross-serve ──────────

#[test]
fn distinct_fingerprints_do_not_cross_serve() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let a = store(&peer, ns_of(&hybrid()));
    let b = store(&peer, ns_of(&dense()));
    a.put(K, &[0xAA; BLOB]).unwrap();
    let mut out = [0u8; BLOB];
    assert!(
        !b.get(K, &mut out).unwrap(),
        "model B must MISS model A's state for the same logical key"
    );
    assert_eq!(out, [0u8; BLOB], "a miss must leave `out` untouched");
    // …while model A still hits its own entry (isolation, not amnesia).
    assert!(a.get(K, &mut out).unwrap());
    assert_eq!(out, [0xAA; BLOB]);
}

// ── T2: pin the OLD bug so it cannot come back ────────────────────────────
// OLD behavior: with ATLAS_TARGET_MODEL unset both models derived ns=0 and
// wire() passed the key through unchanged — i.e. both folded ONE equal
// effective namespace (the decode tier likewise shared the bare
// DECODE_DOMAIN constant). ns=0 is now unrepresentable (NonZeroU64), so the
// pin is written as "EQUAL namespaces collide" — semantically the same bug,
// refactor-proof against the passthrough deletion.

#[test]
fn equal_namespaces_cross_serve_the_old_default_bug() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    // e.g. the old shared decode default: DECODE_DOMAIN for EVERY model.
    let shared = NonZeroU64::new(atlas_kernels::DECODE_DOMAIN).unwrap();
    let a = store(&peer, shared);
    let b = store(&peer, shared);
    a.put(K, &[0xAA; BLOB]).unwrap();
    let mut out = [0u8; BLOB];
    assert!(
        b.get(K, &mut out).unwrap(),
        "equal namespaces DO collide — this is the pinned bug"
    );
    assert_eq!(
        out, [0xAA; BLOB],
        "model B silently served model A's recurrent state as a cache HIT"
    );
}

// ── T3: identical fingerprint round-trips (the whole point of sharing) ────

#[test]
fn same_fingerprint_round_trips_across_clients() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let ns = ns_of(&hybrid());
    let c1 = store(&peer, ns);
    let c2 = store(&peer, ns);
    let blob: Vec<u8> = (0..BLOB as u8).collect();
    c1.put(K, &blob).unwrap();
    let mut out = vec![0u8; BLOB];
    assert!(
        c2.get(K, &mut out).unwrap(),
        "same model, second client: shared warm-cache HIT"
    );
    assert_eq!(out, blob, "bit-identical restore");
}

// ── Namespace scoping extends to REMOVE and to the decode domain ─────────

#[test]
fn remove_is_namespace_scoped() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let a = store(&peer, ns_of(&hybrid()));
    let b = store(&peer, ns_of(&dense()));
    a.put(K, &[0xAA; BLOB]).unwrap();
    b.remove(K); // must not evict A's entry
    let mut out = [0u8; BLOB];
    assert!(a.get(K, &mut out).unwrap(), "B's remove must not evict A");
    a.remove(K);
    assert!(!a.get(K, &mut out).unwrap(), "A's remove evicts A");
}

#[test]
fn decode_and_marconi_tiers_do_not_cross_serve_on_one_peer() {
    // Decode + Marconi share ONE peer residency whenever blob_bytes match;
    // the decode namespace mix64(fp, DECODE_DOMAIN) must keep one model's
    // decode spills off its own Marconi keys.
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let fp = ModelFingerprint::derive_with_id(&hybrid(), BLOB, "").unwrap();
    let marconi = store(&peer, fp.nonzero());
    let decode = store(
        &peer,
        NonZeroU64::new(mix64(fp.get(), atlas_kernels::DECODE_DOMAIN)).unwrap(),
    );
    marconi.put(K, &[0xAA; BLOB]).unwrap();
    let mut out = [0u8; BLOB];
    assert!(
        !decode.get(K, &mut out).unwrap(),
        "decode namespace must not serve Marconi state"
    );
}

// ── wire() determinism: what T3 (and every warm hit ever) depends on ──────

#[test]
fn wire_fold_is_deterministic_per_store_instance() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let ns = ns_of(&hybrid());
    let s1 = store(&peer, ns);
    let s2 = store(&peer, ns);
    for key in [0u64, 1, K, u64::MAX] {
        assert_eq!(s1.wire(key), s2.wire(key), "same (ns, key) → same wire key");
    }
    // And distinct namespaces fold the same key differently (T1's mechanism).
    let other = store(&peer, ns_of(&dense()));
    assert_ne!(s1.wire(K), other.wire(K));
}

/// GOLDEN PIN on the PERSISTED key. `wire_fold_is_deterministic_per_store_instance`
/// above proves self-consistency, so it would still pass if the entire fold
/// rotated — while every key already written to a peer's NVMe swap file became
/// unreachable. This pins the actual `u64` that goes on the wire and onto disk.
///
/// It also pins the SSOT identity `wire(key) == mix64(key, ns)`: `wire()` used to
/// be a third hand-transcription of the splitmix64 constants and now delegates to
/// `atlas_tier::hash::mix64`. If those ever diverge, this fails.
///
/// DO NOT update the literal to make this pass — a change here is a deliberate
/// fleet-wide cache flush and must be versioned.
#[test]
fn wire_key_is_mix64_of_key_and_ns() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    // The pinned hybrid-MoE fingerprint (fingerprint_tests::golden_fingerprint_*).
    let ns = NonZeroU64::new(0x5629_922c_51a1_6a10).unwrap();
    let s = store(&peer, ns);
    let key = 0x0123_4567_89ab_cdef_u64;

    assert_eq!(
        s.wire(key),
        0x51f6_0258_9d95_1e89,
        "persisted wire key rotated"
    );
    assert_eq!(
        s.wire(key),
        mix64(key, ns.get()),
        "wire() must BE mix64(key, ns)"
    );
}
