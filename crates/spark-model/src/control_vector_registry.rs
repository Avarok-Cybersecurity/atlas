// SPDX-License-Identifier: AGPL-3.0-only

//! The loaded control vectors, and the per-request identity that selects one.
//!
//! # Why an identity and not a slot index
//!
//! A control vector changes every hidden state, so the KV a request produces
//! under vector A differs from the KV it produces under vector B or under none.
//! Serving one request's cached prefix to another is therefore a correctness
//! bug, not a performance detail.
//!
//! Atlas already solved exactly this for LoRA: the prefix cache is partitioned
//! by a `u64` adapter identity — folded into `hash_token_prefix` AND given a
//! physically disjoint radix root (`spark-runtime/src/radix_tree/inner.rs`),
//! because the hash alone would still collide on the children map. Control
//! vectors ride the same channel rather than growing a second one.
//!
//! The identity is derived from the vector's NAME, like `adapter_id_hash`, so
//! it survives registry reordering, and `0` is reserved for "no vector".
//!
//! # One composition function
//!
//! [`compose_variant_id`] is the ONLY place an adapter id and a cvec id are
//! combined. Two consumers read the result — the prefix-cache key and the
//! admission cohort — and if they ever computed it separately they would drift.
//! The failure mode of that drift is the worst kind: a cache hit returning
//! another variant's KV, which is silent and produces plausible text.

use crate::control_vector::ControlVector;

/// One loaded vector and its identity.
pub struct ControlVectorEntry {
    pub name: String,
    /// Name-derived; never 0 (0 means "no vector").
    pub id: u64,
    pub vector: ControlVector,
}

/// Every control vector this model loaded at boot.
#[derive(Default)]
pub struct ControlVectorRegistry {
    entries: Vec<ControlVectorEntry>,
}

/// Name-derived control-vector identity. FNV-1a, same construction as
/// `lora::key::adapter_id_hash`, with `0` reserved for "no vector" so a serve
/// with none hashes byte-identically to one built before this existed.
pub fn cvec_id_hash(name: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a basis
    for &b in name.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    if h == 0 { 1 } else { h }
}

/// Combine an adapter identity and a control-vector identity into the single
/// `u64` the prefix cache and the admission cohort both key on.
///
/// Two byte-identity pins are deliberate and must not be broken:
///
/// * `compose_variant_id(a, 0) == a` — a serve with no control vector keys
///   exactly as it did before, so every existing LoRA and base prefix entry
///   stays valid.
/// * `compose_variant_id(0, 0) == 0` — the base sentinel survives, which is
///   what keeps `hash_token_prefix`'s zero-fold a strict no-op.
#[inline]
pub fn compose_variant_id(adapter_id: u64, cvec_id: u64) -> u64 {
    if cvec_id == 0 {
        return adapter_id;
    }
    let mut h = adapter_id;
    for &b in cvec_id.to_le_bytes().iter() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    // Never alias the base sentinel: a request WITH a vector is not base.
    if h == 0 { 1 } else { h }
}

/// The control vector a BATCH selected.
///
/// The admission cohort filter restricts a batch to one variant identity
/// (`spark-server/src/scheduler/admission.rs`), so the first sequence is
/// authoritative. Debug builds assert it rather than trust it: a batch that
/// somehow mixed selections would steer some rows and not others, and the
/// output would be well-formed either way, so nothing downstream would catch
/// it. Empty batch = no steering.
#[inline]
pub fn batch_cvec_id(seqs: &[&mut crate::traits::SequenceState]) -> u64 {
    let Some(first) = seqs.first() else {
        return 0;
    };
    debug_assert!(
        seqs.iter().all(|s| s.cvec_id == first.cvec_id),
        "batch mixes control-vector selections: {:?} — the admission cohort \
         filter should have made this impossible",
        seqs.iter().map(|s| s.cvec_id).collect::<Vec<_>>()
    );
    first.cvec_id
}

impl ControlVectorRegistry {
    /// Add a loaded vector. Rejects a duplicate name, and a name whose hash
    /// collides with an existing one — astronomically unlikely, but a
    /// collision would silently merge two vectors' cache partitions.
    pub fn insert(&mut self, name: String, vector: ControlVector) -> anyhow::Result<u64> {
        anyhow::ensure!(
            !self.entries.iter().any(|e| e.name == name),
            "control vector '{name}' is already registered"
        );
        let id = cvec_id_hash(&name);
        if let Some(other) = self.entries.iter().find(|e| e.id == id) {
            anyhow::bail!(
                "control vector '{name}' hashes to the same id as '{}' — rename one",
                other.name
            );
        }
        self.entries.push(ControlVectorEntry { name, id, vector });
        Ok(id)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn by_name(&self, name: &str) -> Option<&ControlVectorEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn by_id(&self, id: u64) -> Option<&ControlVectorEntry> {
        if id == 0 {
            return None;
        }
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|e| e.name.as_str())
    }

    /// The vector a request selected, or `None` for "no vector".
    ///
    /// `None` is a FIRST-CLASS answer here, unlike the LoRA pool where a
    /// request that names nothing falls through to the installed active
    /// adapter and there is no selector meaning "base". Steering has to be
    /// switchable off per request or it cannot be A/B'd, which is most of the
    /// point of making it per-request at all.
    pub fn resolve(&self, id: u64) -> Option<&ControlVector> {
        self.by_id(id).map(|e| &e.vector)
    }
}

#[cfg(test)]
#[path = "control_vector_registry_tests.rs"]
mod control_vector_registry_tests;
