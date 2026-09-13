// SPDX-License-Identifier: AGPL-3.0-only

//! One decoder layer: mixer (KDA|MLA) + MLP (dense|LatentMoE) + AttnRes.
//!
//! BF16 twin bind is C1. GPU decode is a CPU fallback wrapper (copy-out /
//! `atlas_core::kimi_k3` mixer+MLP+AttnRes / copy-in), not CUDA KDA.

pub use atlas_core::kimi_k3::layer::*;
