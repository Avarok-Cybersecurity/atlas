// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the root of the port sprint's (`port/consumer-opt`) new
//! top-level spark-model modules.
//!
//! Declared from `lib.rs` rather than `layers/mod.rs`, which is at the
//! 500-line cap. Each lane's tree lives under `port/<name>/` and is declared
//! here; lines are inserted immediately BEFORE the marker below by the
//! coordinator's fragment apply pass.

/// Per-model state the port lanes add (`ModelPortState`), one field per lane.
pub mod state;
// @port:mods
