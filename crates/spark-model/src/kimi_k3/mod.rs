// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 host graph. This slice is **KDA only**.
//!
//! Math lives in `atlas_core::kimi_k3` so Mac unit tests compile without
//! spark-storage. CUDA launch: [`kda_cuda`].

pub mod kda;
pub mod kda_cuda;

pub use atlas_core::kimi_k3::{
    KDA_L2_EPS, KdaConfig, KdaState, cuda_kda_enabled, kda_decode_token, kda_from,
};
pub use kda_cuda::{K3KdaDecodeKernels, launch_k3_kda_decode_token};
