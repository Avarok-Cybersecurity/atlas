// SPDX-License-Identifier: AGPL-3.0-only

//! K3 KDA CPU ref (twin 8×32 conv-4; prod 96×128 bound −5).
//!
//! Not a GDN/Mamba reuse. CUDA: [`super::kda_cuda`] (`K3_CUDA_KDA=1`).

pub use atlas_core::kimi_k3::kda::*;
