// SPDX-License-Identifier: AGPL-3.0-only

//! K3 KDA CPU ref (head_dim 128, conv 4, full-rank gate, bound −5).
//!
//! Not a GDN/Mamba reuse. GPU kernels are a later slice, behind goldens.

pub use atlas_core::kimi_k3::kda::*;
