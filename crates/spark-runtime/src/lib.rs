// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

pub mod buffers;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(unix)]
pub mod fast_weights;
pub mod gpu;
pub mod kernel_args;
pub mod kernel_audit;
pub mod kv_cache;
pub mod kv_dequant;
pub mod kv_spill;
#[cfg(feature = "metal")]
pub mod metal_backend;
pub mod prefix_cache;
pub mod radix_tree;
pub mod sampler;
pub mod weights;
