// SPDX-License-Identifier: AGPL-3.0-only

//! KV namespace derivation pins: frozen golden literals (the ns is a DURABLE
//! on-peer key contract), per-field sensitivity (the fingerprint-exclusion
//! trap: elem_bytes / block geometry are NOT in the SSM fp and must flip the
//! ns here), cross-crate hash-primitive pins (spark-model's vendored copies
//! share these exact literals in `fingerprint_tests.rs` — drift fails one
//! side), and the strict env resolvers (PCND).

use std::num::NonZeroU64;

use super::*;
use crate::group::GroupLayout;

/// Holo-like layout: 80 layers × 4096 blocks × 8 kv_heads, bs 16, hd 128,
/// BF16 → group_stride 4096, block_bytes 65536.
fn layout() -> GroupLayout {
    GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096)
}

const FP: u64 = 0x5629_922c_51a1_6a10; // spark-model's pinned hybrid-config fp
const SALT: u64 = 0xD00D_F00D_0000_0001;

fn ns_with(f: impl FnOnce(&mut (u64, GroupLayout, u32, u32, u32, u64))) -> NonZeroU64 {
    let mut a = (FP, layout(), 2u32, 16u32, 128u32, SALT);
    f(&mut a);
    derive_kv_ns(a.0, &a.1, a.2, a.3, a.4, a.5)
}

// ── frozen contracts ─────────────────────────────────────────────────────

#[test]
// The protocol-constant pins below are deliberate constant asserts: they are
// load-bearing frozen contracts (durable on-peer keys), kept as visible
// runtime test failures.
#[allow(clippy::assertions_on_constants)]
fn version_and_domain_frozen() {
    assert_eq!(KV_NS_VERSION, 1);
    assert_eq!(KV_DOMAIN, 0x4B56_5041_4745_0001, "\"KV\" + \"PAGE\" + 1");
    assert_ne!(KV_DOMAIN, 0, "the domain must be a usable ns fallback");
}

/// The vendored FNV-1a/64 matches the published reference vectors — the SAME
/// vectors spark-model's `fingerprint_tests.rs` pins its copy to, so the two
/// vendored primitives cannot drift apart without failing one crate.
#[test]
fn fnv1a_64_matches_reference_vectors() {
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
}

/// Cross-crate splitmix pin: these EXACT literals are also asserted against
/// spark-model's `mix64` (fingerprint_tests.rs `mix64_frozen_literals`). The
/// fold is what turns (group_id, ns) into the durable wire key on BOTH the
/// SSM and KV paths, so a drifted constant rotates or collides on-disk keys.
#[test]
fn mix64_frozen_literals() {
    assert_eq!(mix64(0, 0), 0x0); // splitmix64's fixed point at zero
    assert_eq!(mix64(1, 2), 0xbeeb_8da1_658e_ec67);
    assert_eq!(
        mix64(0x5EED_F00D_CAFE_D00D, 0xD3C0_DE12_A5B6_C7D8),
        0xe567_5d86_f750_1640
    );
}

/// The full derivation is frozen: same inputs → this literal, forever (bump
/// KV_NS_VERSION to rotate deliberately).
#[test]
fn derive_kv_ns_golden() {
    assert_eq!(ns_with(|_| {}).get(), 0x74b4_02b0_f421_5375);
}

#[test]
fn wire_key_golden_and_mechanism() {
    let ns = ns_with(|_| {});
    // wire_key IS the splitmix fold (same as PagingSnapshotStore::wire).
    assert_eq!(wire_key(ns, 42), mix64(42, ns.get()));
    assert_eq!(wire_key(ns, 42), 0x6d2d_470e_f594_7d4b);
    // Deterministic; distinct namespaces fold the same id differently.
    assert_eq!(wire_key(ns, 42), wire_key(ns, 42));
    let other = ns_with(|a| a.5 ^= 1);
    assert_ne!(wire_key(ns, 42), wire_key(other, 42));
}

// ── per-field sensitivity: the fingerprint-exclusion trap ────────────────
// elem_bytes / block_size / head_dim / num_blocks etc. are ABSENT from the
// SSM ModelFingerprint (by documented design); each must flip the KV ns or
// two byte-incompatible serves would share keys.

#[test]
fn every_field_flips_the_namespace() {
    let base = ns_with(|_| {});
    let variants: [(&str, NonZeroU64); 9] = [
        ("model_fp", ns_with(|a| a.0 ^= 1)),
        ("elem_bytes", ns_with(|a| a.2 = 1)), // fp8 KV vs bf16 KV
        ("block_size", ns_with(|a| a.3 = 32)),
        ("head_dim", ns_with(|a| a.4 = 64)),
        (
            "num_layers",
            ns_with(|a| a.1 = GroupLayout::new(48, 4096, 8, 16, 128, 2, 4096)),
        ),
        (
            "num_blocks", // a different --gpu-memory budget renumbers group_ids
            ns_with(|a| a.1 = GroupLayout::new(80, 2048, 8, 16, 128, 2, 4096)),
        ),
        (
            "num_kv_heads",
            ns_with(|a| a.1 = GroupLayout::new(80, 4096, 4, 16, 128, 2, 4096)),
        ),
        (
            "fs_block_size", // also moves group_stride padding
            ns_with(|a| a.1 = GroupLayout::new(80, 4096, 8, 16, 128, 2, 512)),
        ),
        ("client_salt", ns_with(|a| a.5 = SALT ^ 0xFFFF)),
    ];
    for (name, v) in variants {
        assert_ne!(v, base, "changing {name} must flip the KV namespace");
    }
}

// ── strict env resolvers (env-free cores; PCND) ──────────────────────────

#[test]
fn flag_selection_is_strict() {
    // Flag OFF (unset / "0") ⇒ the raw dumb one-sided path
    // (connect_kv_peer_backend calls RdmaKvBackend::connect — same data
    // plane; its handshake is the v2 header with blob == 0).
    assert!(!kv_paging_selected(None).unwrap());
    assert!(!kv_paging_selected(Some("0")).unwrap());
    assert!(kv_paging_selected(Some("1")).unwrap());
    // A typo must never silently pick a path.
    assert!(kv_paging_selected(Some("yes")).is_err());
    assert!(kv_paging_selected(Some("")).is_err());
}

/// Flag OFF wire half (the bare `total_bytes` handshake is retired):
/// `RdmaKvBackend` now sends the v2 header with `blob_bytes == 0` (RAW
/// one-sided mode). Pin that a REALISTIC Holo-like total round-trips through
/// encode/parse and stays under the peer's explicit arena sanity bound.
#[test]
fn flag_off_raw_v2_header_round_trips() {
    use crate::snapshot_swap::{PagingKind, encode_paging_v2_header, parse_paging_header};
    let l = layout();
    let num_groups = (l.num_layers as u64) * 2 * (l.num_blocks as u64) * (l.num_kv_heads as u64);
    let total = num_groups * l.group_stride; // what RdmaKvBackend::connect sizes
    assert!(
        total <= (1 << 42),
        "raw totals stay under the peer's arena sanity bound"
    );
    let w = encode_paging_v2_header(PagingKind::KV, total, 0);
    let first = u64::from_le_bytes(w[0..8].try_into().unwrap());
    let mut c = std::io::Cursor::new(w[8..].to_vec());
    assert_eq!(
        parse_paging_header(first, &mut c).unwrap(),
        (PagingKind::KV, total, 0),
        "flag-OFF clients ride the RAW one-sided mode (blob_bytes == 0)"
    );
}

#[test]
fn cascade_paging_conflict_is_detected() {
    // Only the (peer set AND flag on) combination conflicts.
    assert!(cascade_conflicts_with_paging(true, Some("1")).unwrap());
    assert!(!cascade_conflicts_with_paging(true, None).unwrap());
    assert!(!cascade_conflicts_with_paging(true, Some("0")).unwrap());
    assert!(!cascade_conflicts_with_paging(false, Some("1")).unwrap());
    // Strictness propagates (a typo'd flag is still a startup error).
    assert!(cascade_conflicts_with_paging(true, Some("junk")).is_err());
}

#[test]
fn ns_override_is_strict() {
    let derived = NonZeroU64::new(0xFEED).unwrap();
    assert_eq!(resolve_kv_ns_from(None, derived).unwrap(), derived);
    assert_eq!(
        resolve_kv_ns_from(Some("0xD3C0"), derived).unwrap().get(),
        0xD3C0
    );
    assert_eq!(resolve_kv_ns_from(Some("77"), derived).unwrap().get(), 77);
    assert!(resolve_kv_ns_from(Some("junk"), derived).is_err());
    assert!(
        resolve_kv_ns_from(Some("0"), derived).is_err(),
        "ns=0 stays unrepresentable"
    );
}

#[test]
fn salt_override_is_strict() {
    assert_eq!(resolve_salt_from(None).unwrap(), None); // caller randomizes
    assert_eq!(resolve_salt_from(Some("0x10")).unwrap(), Some(0x10));
    assert_eq!(resolve_salt_from(Some("7")).unwrap(), Some(7));
    assert!(resolve_salt_from(Some("nope")).is_err());
}

#[test]
fn arena_resolution_is_strict_and_block_aligned() {
    let bb = 65536u64; // Holo-like block_bytes
    // REQUIRED: absence is an error naming the variable (PCND).
    let err = resolve_arena_bytes_from(None, bb).unwrap_err().to_string();
    assert!(err.contains("ATLAS_KV_PAGING_ARENA_GB"), "{err}");
    // 1 GiB is already a block multiple.
    assert_eq!(resolve_arena_bytes_from(Some("1"), bb).unwrap(), 1 << 30);
    // Fractional GiB floors to a block multiple.
    let a = resolve_arena_bytes_from(Some("0.001"), bb).unwrap();
    assert_eq!(a % bb, 0);
    assert!(a > 0 && a <= (0.001 * (1u64 << 30) as f64) as u64);
    // Junk / zero / negative / sub-block all fail fast.
    assert!(resolve_arena_bytes_from(Some("lots"), bb).is_err());
    assert!(resolve_arena_bytes_from(Some("0"), bb).is_err());
    assert!(resolve_arena_bytes_from(Some("-1"), bb).is_err());
    assert!(resolve_arena_bytes_from(Some("0.00000001"), bb).is_err());
}

/// The MISS hard-error names the peer flag that makes KV miss-proof — the
/// message is the operator's runbook line, so pin its load-bearing parts.
#[test]
fn miss_error_names_the_miss_proof_config() {
    let e = super::super::kv_miss_error(3, 77).to_string();
    assert!(e.contains("layer 3"), "{e}");
    assert!(e.contains("--swap-cap-gb-kv 0"), "{e}");
    assert!(e.contains("unrecoverable"), "{e}");
}
