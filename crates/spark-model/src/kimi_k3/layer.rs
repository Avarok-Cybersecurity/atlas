// SPDX-License-Identifier: AGPL-3.0-only

//! One decoder layer: mixer (KDA|MLA) + MLP (dense|LatentMoE) + AttnRes.
//!
//! BF16 twin bind is C1; GPU decode still bails (CPU greedy is atlas-core).

pub use atlas_core::kimi_k3::layer::*;
