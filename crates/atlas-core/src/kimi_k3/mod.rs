// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 host graph: CPU references for KDA, gated NoPE MLA, AttnRes,
//! SiTU-GLU, and LatentMoE.
//!
//! K3-DECISION: KDA is a new backend. These refs do not call GDN / Mamba-2
//! / Qwen3-Next kernels. GPU kernels come later behind goldens.

pub mod attnres;
pub mod cache;
pub mod cpu_forward;
pub mod cpu_weights;
pub mod greedy;
pub mod kda;
pub mod latent_moe;
pub mod layer;
pub mod mla;
pub mod ops;
pub mod situ;

pub use attnres::{attnres_blend, attnres_mix, attnres_softmax_mix};
pub use cache::HybridCache;
pub use cpu_weights::{Ablation, K3CpuModel};
pub use greedy::greedy_decode;
pub use kda::{KdaConfig, KdaState, kda_decode_token};
pub use latent_moe::{LatentMoeConfig, latent_moe_forward, sigmoid_topk};
pub use layer::{K3Graph, K3LayerSpec, MixerKind, MlpKind};
pub use mla::{MlaConfig, gated_mla_attend};
pub use situ::{situ_glu, situ_glu_vec, softcap};

#[cfg(test)]
mod c1;
