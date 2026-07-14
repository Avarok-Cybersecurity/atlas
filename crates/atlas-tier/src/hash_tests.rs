// SPDX-License-Identifier: AGPL-3.0-only

//! These pins are the whole safety story for durable cache keys: they catch a
//! "helpful" reimplementation that changes the value, which would silently
//! rotate (or collide) every key persisted in a peer's NVMe swap file.

use super::{fnv1a_64, mix64};

/// Pin the primitive to the PUBLISHED FNV-1a/64 reference vectors — someone
/// else's constants, so a swapped or "optimized" implementation cannot pass.
#[test]
fn fnv1a_64_matches_published_reference_vectors() {
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
}

#[test]
fn fnv1a_64_is_order_sensitive_and_length_sensitive() {
    assert_ne!(fnv1a_64(b"ab"), fnv1a_64(b"ba"));
    assert_ne!(fnv1a_64(b"a"), fnv1a_64(b"aa"));
}

/// mix64 is the fold behind EVERY persisted wire key. Pin it to literals: this
/// is the value that lands on disk, so it is arguably a more important pin than
/// the fingerprint itself.
#[test]
fn mix64_golden_pins() {
    assert_eq!(mix64(1, 0), 0x5692_161d_100b_05e5);
    assert_eq!(mix64(0, 1), 0xe220_a839_7b1d_cdaf);
    assert_eq!(mix64(0xdead_beef, 0xcafe_f00d), 0x5f3e_915c_5c36_4c56);
}

/// splitmix64 has a ZERO FIXED POINT: `mix64(0, 0) == 0`. Pinned because it is
/// load-bearing, not a curiosity — it is precisely why every caller that turns a
/// mix into a namespace needs zero-avoidance (`NonZeroU64`), and why the decode
/// namespace must not fall back to the fingerprint when the mix lands on zero.
#[test]
fn mix64_has_a_zero_fixed_point() {
    assert_eq!(mix64(0, 0), 0);
}

/// Namespace separation: the same logical key under two namespaces must not
/// collide. This is the property that stops two models cross-serving on one peer.
#[test]
fn mix64_separates_namespaces() {
    let key = 0x1234_5678_9abc_def0;
    assert_ne!(mix64(key, 1), mix64(key, 2));
    assert_ne!(mix64(key, 0), mix64(key, u64::MAX));
}

/// Deterministic + const-evaluable (both are load-bearing: the values are
/// durable, and callers use them in const contexts).
#[test]
fn mix64_and_fnv_are_const_and_deterministic() {
    const H: u64 = fnv1a_64(b"atlas");
    const M: u64 = mix64(H, 7);
    assert_eq!(H, fnv1a_64(b"atlas"));
    assert_eq!(M, mix64(H, 7));
}
