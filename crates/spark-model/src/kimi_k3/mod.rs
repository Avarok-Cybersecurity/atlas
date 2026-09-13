// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 host graph (S1). CPU refs + layer enum + hybrid cache structs.
//!
//! K3-DECISION: KDA is a new backend. Do **not** copy GDN / Mamba-2 /
//! Qwen3-Next / glm5next_kda kernels into `kernels/gb10/kimi-k3/`. Math lives
//! in `atlas_core::kimi_k3` so Mac unit tests compile without spark-storage.

pub mod attnres;
pub mod bound;
pub mod cache;
mod host_decode;
pub mod kda;
pub mod kda_cuda;
pub mod latent_moe;
pub mod layer;
pub mod mla;
pub mod mla_cuda;
pub mod situ;
pub mod state;

pub use atlas_core::kimi_k3::{
    Ablation, HybridCache, K3CpuModel, K3Graph, K3LayerSpec, KdaConfig, KdaState, LatentMoeConfig,
    LayerCache, MixerKind, MlaConfig, MlpKind, attnres_blend, attnres_softmax_mix,
    cuda_kda_enabled, cuda_mla_enabled, gated_mla_attend, greedy_decode, kda_decode_token,
    latent_moe_forward, mla_decode_token, sigmoid_topk, situ_glu, situ_glu_vec, softcap,
};
pub use kda_cuda::{K3KdaDecodeKernels, launch_k3_kda_decode_token};
pub use mla_cuda::{K3MlaDecodeKernels, launch_k3_mla_decode_token};
pub use state::K3CpuFallbackState;

#[cfg(test)]
mod host_decode_kda;
