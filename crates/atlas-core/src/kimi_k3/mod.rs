// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 host graph. This slice is **KDA only**.
//!
//! K3-DECISION: KDA is a new backend. Do not copy GDN / Mamba-2 /
//! Qwen3-Next / glm5next_kda kernels into `kernels/gb10/kimi-k3/`.
//! CUDA: `kda_decode.cu` (default; `K3_CUDA_KDA=0` CPU escape).

pub mod attnres;
pub mod cache;
pub mod kda;
pub mod latent_moe;
pub mod layer;
pub mod mla;
pub mod situ;

pub use attnres::{AttnResHub, attnres_blend, attnres_mix, attnres_softmax_mix};
pub use cache::{HybridCache, LayerCache, MlaKv};
pub use kda::{KDA_L2_EPS, KdaConfig, KdaState, cuda_kda_enabled, kda_decode_token, kda_from};
pub use latent_moe::{
    LatentMoeConfig, latent_moe_forward, mix_routed_experts, moe_from, sigmoid_topk,
};
pub use layer::{K3Graph, K3LayerSpec, MixerKind, MlpKind};
pub use mla::{MlaConfig, cuda_mla_enabled, gated_mla_attend, mla_decode_token, mla_from};
pub use situ::{situ_glu, situ_glu_vec, softcap};
