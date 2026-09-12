// SPDX-License-Identifier: AGPL-3.0-only

//! One decoder layer: mixer (KDA|MLA) + MLP (dense|LatentMoE) + AttnRes.
//!
//! GPU `TransformerLayer` bind stays bailed in `KimiK3WeightLoader` until C1.

pub use atlas_core::kimi_k3::layer::*;
