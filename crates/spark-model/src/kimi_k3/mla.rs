// SPDX-License-Identifier: AGPL-3.0-only

//! Gated NoPE MLA CPU ref. Do not reuse `qwen3_attention` blindly.
//! CUDA: [`super::mla_cuda`] (`K3_CUDA_MLA=1` opt-in).

pub use atlas_core::kimi_k3::mla::*;
