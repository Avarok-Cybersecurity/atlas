// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the port sprint's (`port/consumer-opt`) `spark serve` flags,
//! flattened into `ServeArgs` as its `port_sprint` field (`ServeArgs::port` is
//! the HTTP listen port).
//!
//! Its own struct so the lanes' flags never touch `serve_args.rs` (allow-listed
//! over the cap). Every flag is an `Option` or has a default that changes
//! nothing: an absent flag publishes nothing, so the row's environment
//! variable and then the target's declaration decide, exactly as without the
//! flag. Each flag is published by one line in `main_modules/port_apply.rs`.
//! Lines are inserted immediately BEFORE the marker below by the coordinator's
//! fragment apply pass.

/// The port lanes' serve flags.
#[derive(clap::Args, Debug, Clone, Default, PartialEq)]
pub struct PortArgs {
    /// Pinned host pool (lane L25): MiB page-locked per `cuMemHostRegister`
    /// call, 1 to 4096 (default 64). Each pool chunk is pinned by its own call,
    /// on a background thread after the first chunks; a run never spans two
    /// chunks, so this is also the largest run, and the batched KV swap's
    /// staging (two halves) never exceeds one chunk.
    #[arg(long = "pinned-pool-chunk-mb", value_name = "MIB")]
    pub pinned_pool_chunk_mb: Option<u32>,
    // @port:fields
}
