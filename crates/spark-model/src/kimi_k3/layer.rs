// SPDX-License-Identifier: AGPL-3.0-only

//! One decoder layer: mixer (KDA|MLA) + MLP (dense|LatentMoE) + AttnRes.
//!
//! BF16 twin bind is C1. GPU decode copies out, runs mixer+MLP+AttnRes, copies
//! in. Default mixer is CPU. `K3_CUDA_KDA=1` runs the KDA core on CUDA.

pub use atlas_core::kimi_k3::layer::*;
