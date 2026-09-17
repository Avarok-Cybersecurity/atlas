// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

//! avarok-rdma: the one-sided RDMA verbs primitive shared by every Avarok RDMA
//! tier (experts / KV overflow / weight staging / LoRA / SSM snapshots).
//!
//! The C shim's QP/RTR/RTS attribute constants and the handshake wire codecs
//! are a frozen external contract: they must stay byte-compatible with already
//! deployed peers, so the shim is treated as byte-stable.
//!
//! CUDA-free by hard constraint (see Cargo.toml): both the non-CUDA peer
//! daemons and the CUDA client tiers link this.
//!
//! `cfg(avarok_rdma_verbs)` is decided by build.rs — ON when `target_os` is not
//! macOS and `AVAROK_SKIP_BUILD`/`SKIP_AVAROK_BUILD` is unset (there is no
//! rdma-core probe; if the headers are missing there, `cc` fails the build
//! loudly) — and published to direct dependents via the `links` metadata
//! `DEP_AVAROK_RDMA_SHIM_HAS_VERBS`. See build.rs for the full story.

/// Rail-env resolution helpers (`first_set` / `first_nonempty` /
/// `first_set_u32`) — un-gated, pure `std::env`.
pub mod env;
/// Client handshake steps as pure `Read`/`Write` byte functions (identity
/// tuples, no `Verbs`) — un-gated so the transcript goldens run everywhere.
pub mod handshake;
/// The golden handshake wire codecs (both server-param dialects, rails
/// framing, mode/status bytes) — un-gated, a frozen external wire contract.
pub mod wire;

// Pure cardinality policy used by the verbs-gated RailSet. Kept un-gated so
// its no-NIC regression checks execute on every development platform.
#[cfg(any(avarok_rdma_verbs, test))]
mod rail_count;

/// Safe-ish wrapper over the C shim: one [`verbs::Verbs`] == one RC QP.
/// Compiled only where the shim is (`cfg(avarok_rdma_verbs)`).
#[cfg(avarok_rdma_verbs)]
pub mod verbs;

/// RailSet — the one client-side rail bring-up. Gated with the shim: it
/// creates and connects real QPs.
#[cfg(avarok_rdma_verbs)]
pub mod railset;

#[cfg(avarok_rdma_verbs)]
pub use railset::{Rail, RailSet, RailSpec};
#[cfg(avarok_rdma_verbs)]
pub use verbs::{Gid, MrKeys, Verbs};
pub use wire::{
    CacheServerParams, MODE_TCP, MODE_VERBS, RemoteQp, STATUS_ERR, STATUS_OK, VerbsClientParams,
    VerbsServerParams,
};

/// `true` iff this build of avarok-rdma compiled the verbs shim (i.e. the
/// `avarok_rdma_verbs` cfg was emitted by build.rs). A permanent, always-
/// compiled witness: tests assert it so a silent cfg evaporation fails
/// `cargo test` on verbs hosts instead of green-building an empty crate.
pub const fn verbs_enabled() -> bool {
    cfg!(avarok_rdma_verbs)
}
