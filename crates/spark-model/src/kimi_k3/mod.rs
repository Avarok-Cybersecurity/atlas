// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 host graph (S1). CPU refs + layer enum + hybrid cache structs.
//!
//! K3-DECISION: KDA is a new backend. Do **not** copy GDN / Mamba-2 /
//! Qwen3-Next / glm5next_kda kernels into `kernels/gb10/kimi-k3/`. Math lives
//! in `atlas_core::kimi_k3` so Mac unit tests compile without spark-storage.

pub mod attnres;
pub mod cache;
pub mod kda;
pub mod latent_moe;
pub mod layer;
pub mod mla;
pub mod situ;

pub use atlas_core::kimi_k3::{
    HybridCache, K3Graph, K3LayerSpec, KdaConfig, KdaState, LatentMoeConfig, MixerKind, MlaConfig,
    MlpKind, attnres_blend, attnres_softmax_mix, gated_mla_attend, kda_decode_token,
    latent_moe_forward, sigmoid_topk, situ_glu, situ_glu_vec, softcap,
};
