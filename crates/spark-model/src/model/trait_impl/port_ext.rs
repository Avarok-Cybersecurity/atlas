// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: `impl ModelPortExt for TransformerModel`, the port sprint's
//! (`port/consumer-opt`) model-level entry points.
//!
//! One delegating line per method, each calling the owning lane's own
//! function. `Model::port_ext` returns `Some(self)` for `TransformerModel`
//! (the override sits in the one `impl Model` block in `mod.rs`, because a
//! trait method override cannot live in a second file). Lines are inserted
//! immediately BEFORE the marker below by the coordinator's fragment apply
//! pass.

use super::super::types::TransformerModel;
use crate::traits::port_ext::ModelPortExt;

impl ModelPortExt for TransformerModel {
    // @port:impls
}
