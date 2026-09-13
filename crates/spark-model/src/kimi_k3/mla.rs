// SPDX-License-Identifier: AGPL-3.0-only

//! Gated NoPE MLA CPU ref. Do not reuse `qwen3_attention` blindly.
//! CUDA: [`super::mla_cuda`] (default; `K3_CUDA_MLA=0` CPU).

pub use atlas_core::kimi_k3::mla::*;
