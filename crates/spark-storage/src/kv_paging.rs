// SPDX-License-Identifier: AGPL-3.0-only

//! KV as a first-class paging kind (part of the tiered-cache
//! consolidation, DEFAULT-OFF behind `ATLAS_KV_PAGING`).
//!
//! The flag-OFF KV overflow tier (`rdma_kv_backend`) takes the "dumb
//! one-sided path": the peer registers ONE RW MR and the CLIENT owns a static
//! allocator (`base + group_id × group_stride`) over a fixed arena — no peer
//! residency, no NVMe spill. It selects that mode with the v2
//! header's `blob_bytes == 0` RAW sentinel (the bare `[u64 total_bytes]`
//! handshake is retired).
//! This module is the alternative client: it sends the paging handshake
//! (`PAGING_MAGIC_V2`, kind = `PagingKind::KV`) so KV inherits what the SSM
//! snapshot tier already has — peer-owned residency, an NVMe-backed cold tier
//! (unbounded depth with `--swap-cap-gb-kv 0`), and peer capacity pooling.
//! The peer side is ALREADY wired: the `(kind, blob_bytes)` arena registry
//! accepts kind 1 and `--swap-cap-gb-kv` parses; the whole feature is this
//! client.
//!
//! Layout: the paging RECORD is one whole KV block (`GroupLayout::
//! block_bytes()` = `2·num_kv_heads·group_stride`) — exactly the fixed-size
//! record `atlas_tier::Residency` demands, one contiguous blob at one pointer
//! on both ends, and 1 control RTT per block instead of `2·num_kv_heads`.
//! Keys fold the per-model fingerprint + full layout identity + a per-client
//! salt (see [`ns`]); `PagingKind::KV` in the handshake plus the
//! [`ns::KV_DOMAIN`] fold domain-separate KV from SSM.
//!
//! FLAG OFF (`ATLAS_KV_PAGING` unset/0) the selection helper returns the raw
//! one-sided `RdmaKvBackend` — identical data plane (v2 RAW handshake since
//! the v2 handshake), so any regression bisects on this single flag.

pub mod ns;

#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
mod backend;
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
mod connect;

#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub use backend::{KvPagingBackend, KvPagingConnect};
#[cfg(all(feature = "cuda", atlas_rdma_verbs))]
pub use connect::connect_kv_peer_backend;

/// The hard-error a KV paging GET miss maps to. `StorageBackend::read` has no
/// miss channel and an evicted KV block is UNRECOVERABLE without recomputing
/// the prefix (unlike SSM snapshots, which recompute on miss by design) — so
/// ST_MISS is corruption-equivalent and must fail loudly, naming the peer
/// config that makes the KV kind miss-proof.
pub fn kv_miss_error(layer: u32, block: u32) -> anyhow::Error {
    anyhow::anyhow!(
        "kv-paging: block (layer {layer}, disk block {block}) is not on the peer — an \
         evicted KV block is unrecoverable (silent KV loss would corrupt long-context \
         output). Run the peer with --swap-cap-gb-kv 0 (unbounded KV disk) and size \
         --max-blade-gb / ATLAS_KV_PAGING_ARENA_GB for the working set"
    )
}

#[cfg(test)]
mod isolation_tests;
