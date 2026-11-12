// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the port sprint's (`port/consumer-opt`) SERVING rows of
//! `kernels/<hw>/HARDWARE.toml` `[defaults]`, as the runtime type baked into
//! [`crate::TargetDefaults::port`].
//!
//! A sub-table of [`crate::TargetDefaults`] rather than more fields on it, so
//! the sprint's rows land in files of their own: `build_defaults_port.rs`
//! parses them, `build.rs` bakes them into the same generated
//! `TARGET_DEFAULTS` constant (one generated file, one `include!`, one content
//! hash), and `spark_model::layers::ops::target_defaults_port` resolves them
//! against the environment.
//!
//! Every row is OFF (or its baseline string or number) on every target when it
//! lands. A row-off path is byte-identical to the behavior before the row
//! existed, so GB10, Hopper, B200, B300, Metal and Strix serve exactly as they
//! did.
//!
//! Rows arrive through the coordinator's fragment apply pass: each `@port:`
//! marker below is an insertion point, and text is inserted immediately BEFORE
//! the marker line.

/// The port sprint's rows of one compiled target's `[defaults]` table.
///
/// `Copy` and entirely `'static`, like [`crate::TargetDefaults`], of which it
/// is the `port` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PortDefaults {
    // @port:fields
}

/// What a target that declares no port row gets: every row's baseline.
///
/// Kept in agreement with `build_defaults_port.rs`'s `Default for PortFields`
/// by the row templates (one baseline per row, written into both files in the
/// same apply pass).
pub const PORT_BASELINE: PortDefaults = PortDefaults {
    // @port:baseline
};
