// SPDX-License-Identifier: AGPL-3.0-only

//! Stable hash primitives for **durable, cross-process cache keys**.
//!
//! These values are written into the paging peer's NVMe swap file, which
//! outlives the client that wrote them. So the same input must produce the same
//! `u64` forever — across toolchains, platforms, target architectures and
//! rebuilds. That is a *durable on-disk contract*, not a hashmap.
//!
//! `std::hash::DefaultHasher` is unusable here: std documents its algorithm as
//! unspecified and "not to be relied upon over releases", so a toolchain bump
//! silently rotates every persisted key (total cache miss) or collides with a
//! stale namespace (silent wrong-state corruption). `ahash` is worse still — it
//! varies by CPU feature (AES-NI). A crate dependency is also a hazard: a
//! version bump could rotate the fleet's keys. Hence: vendored, ~20 lines, here.
//!
//! They live in `atlas-tier` because it is pure (`anyhow` + `libc`, no CUDA, no
//! verbs) and a common ancestor of its consumers — the SSM tier fingerprint and
//! the KV paging namespace — so both derive keys from ONE definition of these
//! constants rather than separate copies free to drift.
//!
//! ## Threat model (read before reusing these)
//!
//! FNV-1a is **not collision-resistant against an adversary**: it is
//! multiplicative and trivially invertible, so collisions are *constructible*,
//! not merely improbable. That is fine for the current use — a trusted fleet
//! whose model configs are not attacker-chosen, where safety comes from the
//! *injective encoding* of the input plus determinism, and where the birthday
//! bound over a few dozen configs is nil.
//!
//! It would **not** be fine if a shared paging peer ever served untrusted or
//! multi-tenant configs: an attacker able to choose a config could craft one
//! whose namespace collides with another tenant's and read that tenant's
//! recurrent state off the shared peer — defeating the exact property the
//! namespace exists to provide. If that day comes, switch to a keyed hash
//! (keyed BLAKE3, or a vendored SipHash-2-4 with a fixed key) behind a
//! namespace-version bump, which is a deliberate fleet-wide cache flush.

/// FNV-1a/64 offset basis (published constant).
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a/64 prime (published constant).
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// FNV-1a/64 over a byte stream. Fully specified, endianness-free (it consumes
/// bytes, not words), and deterministic across every toolchain and platform.
pub const fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        h = (h ^ bytes[i] as u64).wrapping_mul(FNV_PRIME);
        i += 1;
    }
    h
}

/// splitmix64 finalizer over `a ^ b·GOLDEN` — domain-separated mixing.
///
/// This is the single definition of the fold used for **every persisted wire
/// key**: `PagingSnapshotStore::wire(key) == mix64(key, namespace)` and the
/// decode namespace is `mix64(fingerprint, DECODE_DOMAIN)`. Changing it rotates
/// every key on every peer.
pub const fn mix64(a: u64, b: u64) -> u64 {
    let mut h = a ^ b.wrapping_mul(GOLDEN);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

#[cfg(test)]
#[path = "hash_tests.rs"]
mod tests;
