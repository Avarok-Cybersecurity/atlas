// SPDX-License-Identifier: AGPL-3.0-only
//
// Peer-side paging core for the SSM-snapshot spill tier.
//
// Turns the paging peer's fixed RDMA arena into a bounded page-cache over an
// UNBOUNDED lower tier (an NVMe swap file) — infinite depth. The peer owns the
// residency map (so all fleet clients SHARE one warm cache instead of each
// owning a colliding client-side allocator), and the stable per-rail arena MR
// is NEVER re-registered — bytes swap under the fixed rkey, driven by a TCP
// control channel.
//
// The GENERIC half of this module — the `SlotArena`/`SwapStore` seams, the
// `Residency` page table and the O_DIRECT
// `DirectSwapFile` — lives in the CUDA-/verbs-free `atlas-tier` crate and is
// re-exported below, so consumers keep their `crate::snapshot_swap::*` paths
// unchanged. What REMAINS here is the peer-specific half: the TCP control
// protocol (byte-frozen, golden-pinned — what the fleet peer binary speaks),
// the paging loops, the client codec, and the `MmapSlotArena` over the peer's
// RDMA-registered mmap MR — split into the `wire` and `mmap_arena` sub-modules.

#![allow(dead_code)]

/// The generic paging core, lifted to `atlas-tier` (CUDA- and verbs-free).
/// Re-exported under the `atlas_tier` names (no historical aliases).
pub use atlas_tier::{
    DirectSwapFile, MemSwapStore, Residency, SlotArena, SwapStats, SwapStore, VecSlotArena,
};

mod mmap_arena;
mod wire;

pub use mmap_arena::MmapSlotArena;
pub use wire::*;
