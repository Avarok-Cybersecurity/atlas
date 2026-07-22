// SPDX-License-Identifier: AGPL-3.0-only

//! KV paging namespace + wire-key derivation (`ATLAS_KV_PAGING`, part of
//! the tiered-cache consolidation).
//!
//! The paging peer (atlas-cache-peer) keys its KV arena purely by the u64 the
//! client sends, so the namespace folded into every key is the ONLY thing
//! preventing (a) two MODELS and (b) two same-model CLIENTS from silently
//! serving each other's KV blocks. Unlike the SSM tier's content-derived
//! `prefix_hash`, a KV `GroupKey.block` is a CLIENT-LOCAL disk-block-pool
//! index (`HighSpeedSwap::alloc_disk_block_id`), so identical keys from two
//! same-model clients hold UNRELATED sequence data — a model-only namespace
//! would cross-serve with certainty (every colliding block id), not 2^-64,
//! and a restarted client would hit its own stale pre-restart blocks. The
//! namespace therefore folds a per-client `client_salt` (fresh random per
//! connect; `ATLAS_KV_PAGING_SALT` pins it for tests/harness). Consequence,
//! stated honestly: the KV paging win is peer residency + NVMe depth +
//! capacity pooling, NOT cross-client warm hits (those need content-addressed
//! keys — a separate chunk, same seam as the SSM decode-ns residual).
//!
//! The SSM `ModelFingerprint` VALUE alone is insufficient here by documented
//! design: it excludes the KV dtype and every block-geometry field
//! (fingerprint.rs "a future KV paging tier must fold its own dtype /
//! block_size mix-in at its own call site"), and `GroupLayout::group_id`
//! numbering is layout-relative (`num_blocks` changes with the GPU-memory
//! budget). So the namespace re-folds the full layout identity alongside the
//! fingerprint.
//!
//! Everything here is a DURABLE on-peer contract (the peer's swap file
//! outlives client rebuilds): the tagged encoding is frozen behind
//! [`KV_NS_VERSION`], and the hash primitives are vendored byte-for-byte
//! (FNV-1a/64 + the splitmix64 finalizer, ~25 dependency-free lines) and
//! golden-pinned against spark-model's copies (`ns_tests.rs` here,
//! `fingerprint_tests.rs` there share frozen literals) so the two crates can
//! never drift. spark-storage stays `ModelConfig`-free: the fingerprint
//! arrives as a plain u64 (`ModelDims::model_fp`).

use std::num::NonZeroU64;

use anyhow::{Result, anyhow};

use crate::group::GroupLayout;

/// Bump = deliberate fleet-wide KV cache-key rotation (document it).
pub const KV_NS_VERSION: u64 = 1;

/// Domain separator folded into every KV namespace so a KV wire key is
/// domain-separated from SSM keys IN THE KEY MATERIAL, not merely by the
/// peer's `(kind, blob_bytes)` registry keying (which already gives each kind
/// its own residency map + swap file — this fold makes cross-kind aliasing
/// unrepresentable even if that registry keying were ever collapsed).
/// Frozen; mnemonic `"KV"` + `"PAGE"` + 1. The SSM decode tier's analog is
/// `atlas_kernels::DECODE_DOMAIN`.
pub const KV_DOMAIN: u64 = 0x4B56_5041_4745_0001;

/// Vendored FNV-1a/64 — byte-identical to spark-model's `fingerprint.rs`
/// copy; both are pinned to the published FNV reference vectors.
pub(crate) use atlas_tier::hash::{FNV_OFFSET, fnv1a_64};

// SSOT: this was a FOURTH transcription of the splitmix64 constants — its own doc
// comment asserted it was "byte-identical to spark-model's mix64", which is the
// violation stating itself. One definition, in atlas_tier::hash.
pub(crate) use atlas_tier::hash::mix64;

fn put_u64(buf: &mut Vec<u8>, tag: u8, v: u64) {
    buf.push(tag);
    buf.extend_from_slice(&v.to_le_bytes());
}

/// Derive the KV paging namespace. FROZEN tagged encoding (injective:
/// fixed-width `[tag][8-byte LE]` records) — any field or order change is a
/// deliberate fleet cache flush and must bump [`KV_NS_VERSION`]:
///
/// | tag  | field           | tag  | field          | tag  | field         |
/// |------|-----------------|------|----------------|------|---------------|
/// | 0x00 | KV_NS_VERSION   | 0x04 | block_size     | 0x08 | num_kv_heads  |
/// | 0x01 | model_fp        | 0x05 | head_dim       | 0x09 | fs_block_size |
/// | 0x02 | KV_DOMAIN       | 0x06 | num_layers     | 0x0a | group_stride  |
/// | 0x03 | elem_bytes      | 0x07 | num_blocks     | 0x0b | client_salt   |
///
/// `model_fp` carries the quant identity + model_type + `ATLAS_MODEL_ID`
/// salt (`ModelFingerprint::derive_kv` in spark-model); the geometry fields
/// make the layout identity explicit because `group_id` numbering is
/// layout-relative. Zero-avoidance falls back to `FNV_OFFSET` (p = 2^-64),
/// keeping the result total — ns = 0 stays unrepresentable end-to-end.
pub fn derive_kv_ns(
    model_fp: u64,
    layout: &GroupLayout,
    elem_bytes: u32,
    block_size: u32,
    head_dim: u32,
    client_salt: u64,
) -> NonZeroU64 {
    let mut buf = Vec::with_capacity(12 * 9);
    for (tag, v) in [
        (0x00u8, KV_NS_VERSION),
        (0x01, model_fp),
        (0x02, KV_DOMAIN),
        (0x03, elem_bytes as u64),
        (0x04, block_size as u64),
        (0x05, head_dim as u64),
        (0x06, layout.num_layers as u64),
        (0x07, layout.num_blocks as u64),
        (0x08, layout.num_kv_heads as u64),
        (0x09, layout.fs_block_size),
        (0x0a, layout.group_stride),
        (0x0b, client_salt),
    ] {
        put_u64(&mut buf, tag, v);
    }
    let h = fnv1a_64(&buf);
    NonZeroU64::new(h).unwrap_or(NonZeroU64::new(FNV_OFFSET).expect("FNV offset is non-zero"))
}

/// Wire key for one KV block: the namespace fold of the block's BASE dense
/// group id — `group_id(GroupKey::new(layer, block, 0, K))`, injective across
/// `(layer, block)` within a layout (layout identity rides in the ns, so
/// cross-layout ids cannot alias). Same splitmix fold as the SSM tier's
/// `PagingSnapshotStore::wire` — bijective per namespace, so no birthday risk
/// over the dense group-id keyspace.
pub fn wire_key(ns: NonZeroU64, base_group_id: u64) -> u64 {
    mix64(base_group_id, ns.get())
}

/// Strict u64 parser for the env overrides (decimal or `0x`-hex). Junk is a
/// startup ERROR, never a silent fallthrough (PCND).
pub fn parse_u64_strict(var: &str, raw: &str) -> Result<u64> {
    let s = raw.trim();
    let parsed = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => s.parse::<u64>(),
    };
    parsed.map_err(|e| anyhow!("{var}={raw:?} is not a valid u64 (decimal or 0x-hex): {e}"))
}

/// `ATLAS_KV_PAGING_NS` override (env-free core): strict parse, 0 rejected —
/// a shared peer must always be namespaced (the ns=0 passthrough is
/// unrepresentable, mirroring the landed SSM fix). `None` ⇒ the derived ns.
pub fn resolve_kv_ns_from(override_raw: Option<&str>, derived: NonZeroU64) -> Result<NonZeroU64> {
    match override_raw {
        None => Ok(derived),
        Some(raw) => {
            let v = parse_u64_strict("ATLAS_KV_PAGING_NS", raw)?;
            NonZeroU64::new(v).ok_or_else(|| {
                anyhow!(
                    "ATLAS_KV_PAGING_NS=0 is invalid: ns=0 is unrepresentable (it would \
                     cross-serve KV state on a shared peer); unset it to use the derived \
                     namespace (logged at INFO on connect)"
                )
            })
        }
    }
}

/// `ATLAS_KV_PAGING_SALT` override (env-free core): strict; `Ok(None)` ⇒ the
/// caller generates a fresh random per-connect salt (client isolation +
/// self-healing restart staleness — old-salt peer entries become unreachable
/// and LRU-age out), INFO-logging it for reproducibility.
pub fn resolve_salt_from(raw: Option<&str>) -> Result<Option<u64>> {
    raw.map(|r| parse_u64_strict("ATLAS_KV_PAGING_SALT", r))
        .transpose()
}

/// `ATLAS_KV_PAGING` selection (env-free core): unset or `0` ⇒ the raw dumb
/// one-sided `RdmaKvBackend` path (client-owned allocator; its
/// handshake is the v2 header with `blob_bytes == 0`); `1` ⇒ the peer-owned
/// paging backend. Anything else is a startup ERROR (PCND — a typo must never
/// silently pick a path).
pub fn kv_paging_selected(raw: Option<&str>) -> Result<bool> {
    match raw.map(str::trim) {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(anyhow!(
            "ATLAS_KV_PAGING={other:?} is invalid: 1 = peer-owned paging KV, 0/unset = the \
             raw one-sided KV blade"
        )),
    }
}

/// `ATLAS_KV_PAGING_ARENA_GB` (REQUIRED when the flag is on — no implicit
/// default, PCND): the peer warm-arena size in GiB (fractional accepted),
/// floored to a multiple of `block_bytes` and required to hold ≥ 1 block.
/// The raw path sized the peer to `num_groups × group_stride` (every group
/// a guaranteed slot); the paging arena is a deliberately smaller warm cache
/// over the peer's NVMe swap, so the operator must choose it explicitly.
pub fn resolve_arena_bytes_from(raw: Option<&str>, block_bytes: u64) -> Result<u64> {
    let raw = raw.ok_or_else(|| {
        anyhow!(
            "ATLAS_KV_PAGING=1 requires ATLAS_KV_PAGING_ARENA_GB (peer warm-arena size in \
             GiB, fractional ok) — explicit config or fail fast (PCND)"
        )
    })?;
    let gb: f64 = raw
        .trim()
        .parse()
        .map_err(|e| anyhow!("ATLAS_KV_PAGING_ARENA_GB={raw:?} is not a number: {e}"))?;
    if !gb.is_finite() || gb <= 0.0 {
        return Err(anyhow!(
            "ATLAS_KV_PAGING_ARENA_GB={raw:?} must be a finite value > 0"
        ));
    }
    let bb = block_bytes.max(1);
    let arena = ((gb * (1u64 << 30) as f64) as u64 / bb) * bb;
    if arena == 0 {
        return Err(anyhow!(
            "ATLAS_KV_PAGING_ARENA_GB={raw} is smaller than one KV block ({bb} B) — the \
             warm arena must hold at least one block"
        ));
    }
    Ok(arena)
}

/// Startup guard (env-free core): the cascade T1 (`ATLAS_KV_LOCAL_GB > 0`)
/// flushes evictions DOWN via per-head `write_from_host`, which the
/// block-record paging backend refuses — that combination must fail fast at
/// construction (PCND), never bail mid-decode on the first T1 eviction.
/// Returns `true` iff the incompatible combination is selected.
pub fn cascade_conflicts_with_paging(kv_peer_set: bool, flag_raw: Option<&str>) -> Result<bool> {
    Ok(kv_peer_set && kv_paging_selected(flag_raw)?)
}

#[cfg(test)]
#[path = "ns_tests.rs"]
mod tests;
