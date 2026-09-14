// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 host graph. This slice is **KDA only**.
//!
//! Math lives in `atlas_core::kimi_k3` so Mac unit tests compile without
//! spark-storage. CUDA launch: [`kda_cuda`].

pub mod device_cache;
pub mod kda;
pub mod kda_cuda;
pub mod mla;
pub mod mla_cuda;

pub use atlas_core::kimi_k3::{
    AttnResHub, HybridCache, K3Graph, KDA_L2_EPS, KdaConfig, KdaState, LayerCache, MixerKind,
    MlaConfig, MlaKv, MlpKind, attnres_blend, attnres_mix, attnres_softmax_mix, cuda_kda_enabled,
    cuda_mla_enabled, gated_mla_attend, kda_decode_token, kda_from, mla_decode_token, mla_from,
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
