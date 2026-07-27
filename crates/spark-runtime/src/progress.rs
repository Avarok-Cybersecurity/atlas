// SPDX-License-Identifier: AGPL-3.0-only

//! Structured startup-progress events for the Atlas TUI.
//!
//! Call sites next to the existing human-readable log lines emit ADDITIONAL
//! `debug!`-level events under the dedicated [`TARGET`]. They are invisible on
//! the plain-log path at the default `info` filter — the grepped output the
//! benchmark rigs depend on does not change by a byte — while the TUI's
//! capture layer enables this target unconditionally and decodes the typed
//! fields.
//!
//! Discriminated by the `ev` field:
//!   `phase`       — `phase` (0..=11), `name`
//!   `preflight`   — `disk_gb`, `free_gb`
//!   `shard_start` — `shard` (1-based), `total`, `name`
//!   `shard_done`  — `shard`, `total`, `used_gb`, `free_gb`
//!   `layer`       — `layer` (0-based highest loaded), `total`
//!   `ready`       — `port`

/// Tracing target the TUI capture layer filters on. Keep in sync with
/// `spark-server/src/tui/capture_layer.rs`.
pub const TARGET: &str = "atlas::tui::progress";

/// A named startup phase has begun. `phase` is the serve() phase index.
pub fn phase(phase: u8, name: &str) {
    tracing::debug!(target: "atlas::tui::progress", ev = "phase", phase, name);
}

/// Weight-load preflight: total on-disk GB (the overall-bar denominator).
pub fn preflight(disk_gb: f64, free_gb: f64) {
    tracing::debug!(target: "atlas::tui::progress", ev = "preflight", disk_gb, free_gb);
}

/// A safetensors shard load has started.
pub fn shard_start(shard: usize, total: usize, name: &str) {
    tracing::debug!(target: "atlas::tui::progress", ev = "shard_start", shard, total, name);
}

/// A shard finished; GPU memory snapshot alongside.
pub fn shard_done(shard: usize, total: usize, used_gb: f64, free_gb: f64) {
    tracing::debug!(
        target: "atlas::tui::progress",
        ev = "shard_done",
        shard,
        total,
        used_gb,
        free_gb
    );
}

/// Layer-build progress (sampled by the weight loaders).
pub fn layer(layer: usize, total: usize) {
    tracing::debug!(target: "atlas::tui::progress", ev = "layer", layer, total);
}

/// The server is listening; startup is complete.
pub fn ready(port: u16) {
    tracing::debug!(target: "atlas::tui::progress", ev = "ready", port);
}
