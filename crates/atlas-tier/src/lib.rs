// SPDX-License-Identifier: AGPL-3.0-only
#![deny(warnings)]
#![deny(clippy::all)]

//! The generic tiered-cache core: the CUDA-free, verbs-free paging foundation
//! that the peer daemons build on.
//!
//! One mechanism, three seams:
//!   * [`SlotArena`]  — the bounded HOT tier: `num_slots` fixed-size byte slots
//!     (a peer's mmap'd RDMA MR, a host-RAM `Vec`, …). No CUDA/HBM impl may
//!     live in this crate — that belongs to consumer crates.
//!   * [`SwapStore`]  — the unbounded COLD tier: a fixed-stride record store
//!     ([`DirectSwapFile`] on NVMe, [`MemSwapStore`] in RAM, …).
//!   * [`Residency`]  — the page table over both: opaque `u64` key →
//!     byte-agnostic fixed-size blob, two-level LRU (RAM `lru` above
//!     `disk_lru`), read-pins so a slot is never reused mid-read, and
//!     NEVER-reject puts (a full arena spills the coldest resident to disk; a
//!     capped disk drops its coldest key → clean later miss).
//!
//! This crate is CPU/disk-only: deps are `anyhow` + `libc` (unix), so it is
//! fully unit-testable without RDMA or a GPU and lets the downstream peer
//! daemons build CUDA-free. The peer wire protocol (paging loops, client
//! codec) and the concrete RDMA/HBM tier implementations deliberately do NOT
//! live here — they belong to the consumer crates that build on this core.

mod direct_swap;
mod mem;
mod residency;
mod traits;

pub mod entropy;
pub mod hash;

pub use direct_swap::DirectSwapFile;
pub use mem::{MemSwapStore, VecSlotArena};
pub use residency::Residency;
pub use traits::{SlotArena, SwapStats, SwapStore};
