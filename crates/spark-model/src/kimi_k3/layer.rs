// SPDX-License-Identifier: AGPL-3.0-only

//! One decoder layer: mixer (KDA|MLA) + MLP (dense|LatentMoE) + AttnRes.
//!
//! BF16 twin bind is C1. GPU decode copies out, runs mixer+MLP+AttnRes, copies
//! in. LinearAttention KDA core is CUDA unless `K3_CUDA_KDA=0`. FullAttention
//! MLA core is CUDA unless `K3_CUDA_MLA=0`. Packed LatentMoE experts
//! launch DSV4 `moe_w4a16_grouped_gemm_ptrtable_e8m0`.

pub use atlas_core::kimi_k3::layer::*;
