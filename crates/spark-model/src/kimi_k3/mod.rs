// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 host graph. This slice is **KDA only**.
//!
//! Math lives in `avarok_core::kimi_k3` so Mac unit tests compile without
//! spark-storage. CUDA launch: [`kda_cuda`].

pub mod device_cache;
pub mod kda;
pub mod kda_cuda;
pub mod latent_moe;
pub mod mla;
pub mod mla_cuda;
pub mod moe_cuda;
pub mod tp;

pub use avarok_core::kimi_k3::{
    AttnResHub, HybridCache, K3Graph, KDA_L2_EPS, KdaConfig, KdaState, LatentMoeConfig, LayerCache,
    MixerKind, MlaConfig, MlaKv, MlpKind, attnres_blend, attnres_mix, attnres_softmax_mix,
    cuda_kda_enabled, cuda_mla_enabled, gated_mla_attend, kda_decode_token, kda_from,
    latent_moe_forward, mix_routed_experts, mla_decode_token, mla_from, moe_from, sigmoid_topk,
    situ_glu, situ_glu_vec, softcap,
};
pub use device_cache::{DeviceHybridCache, DeviceLayerCache};
pub use kda_cuda::{
    K3KdaDecodeKernels, KdaDeviceState, launch_k3_kda_decode_token,
    launch_k3_kda_decode_token_on_device,
};
pub use mla_cuda::{
    K3MlaDecodeKernels, MlaDeviceKv, launch_k3_mla_decode_token,
    launch_k3_mla_decode_token_on_device,
};
pub use moe_cuda::{K3MoeGemmKernels, launch_k3_latent_moe_experts};
pub use tp::{supports_tp, tensor_plan};
