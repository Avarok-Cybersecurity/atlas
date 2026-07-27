// SPDX-License-Identifier: AGPL-3.0-only

//! The SSM snapshot spill tier: the model-safety contract (the model-agnostic
//! durable-key [`ModelFingerprint`] and the [`ensure_ssm_tier_capability`]
//! gate) plus the byte-store substrate an evicted snapshot spills into.
//!
//! * [`SnapshotBlobStore`] — the seam: a keyed fixed-size blob store. Backends
//!   are hardware-free and in-process here ([`MemBlobStore`] host-RAM;
//!   [`ArenaSnapshotStore`]/[`PagingSnapshotStore`] over a [`SnapshotTransport`]/
//!   [`PagingTransport`], proven on [`MockSnapshotTransport`]/[`FileSnapshotArena`]).
//!   The env-driven store selection ([`build_tier_store`] / [`build_decode_tier_store`],
//!   gated by `ATLAS_SSM_TIER`) picks a backend. The RDMA/peer arms bind to
//!   `spark_storage::RdmaSnapshotArena` — a `connect`-always-errors stub today
//!   (so a requested RDMA tier degrades to host-RAM), whose real verbs data path
//!   lands with the SSM-snapshot spill wiring in a follow-up PR.

mod arena_store;
mod capability;
mod fingerprint;
mod selectors;
mod store;
mod transport;
mod unified;

pub(crate) use arena_store::{ArenaSnapshotStore, PagingSnapshotStore, RdmaSnapshotStore};
pub(crate) use capability::ensure_ssm_tier_capability;
pub(crate) use fingerprint::ModelFingerprint;
pub(crate) use selectors::{build_decode_tier_store, build_tier_store, ssm_tier_enabled};
pub(crate) use store::{BlobStoreStats, MemBlobStore, SnapshotBlobStore};
pub(crate) use transport::{
    FileSnapshotArena, MockSnapshotTransport, PagingTransport, SnapshotTransport,
};
pub(crate) use unified::{UnifiedSnapshotStore, ssm_tier_unified};
