// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the port sprint's (`port/consumer-opt`) model-level entry
//! points, as one extension trait reached through `Model::port_ext`.
//!
//! A separate trait rather than more `Model` methods, so the lanes' methods
//! never touch `traits/model.rs` (allow-listed over the cap) and every other
//! `Model` implementor (the NLLB model, the test mocks) compiles unchanged:
//! `Model::port_ext` returns `None` by default, and only `TransformerModel`
//! overrides it (`model/trait_impl/mod.rs`) with its implementation in
//! `model/trait_impl/port_ext.rs`.
//!
//! Every method has a default body, so a lane's method is inert for any model
//! that does not implement it. Lines are inserted immediately BEFORE the marker
//! below by the coordinator's fragment apply pass.

/// The port lanes' model-level entry points. Every method has a default body.
pub trait ModelPortExt {
    // @port:methods
}
