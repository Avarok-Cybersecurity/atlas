// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen3 full attention layer.
//!
//! Q/K/V projection -> Q/K norms -> RoPE -> KV cache write ->
//! paged decode attention -> O projection, then MoE FFN.
//!
//! Split into submodules:
//!   - `types`: `MlaWeights` + `Qwen3AttentionLayer` struct definitions
//!   - `init`: `new`, `new_ungated`, `new_with_gating` (kernel loading)
//!   - `helpers`: setters + `apply_layer_scalar` + `effective_attn_scale`
//!   - `prefill_weights`: prefill weight setup + W4A16 M128 dispatcher
//!   - `decode`: single-token attention forward + KV cache helpers
//!   - `prefill`: batched prefill with paged attention
//!   - `trait_impl`: `TransformerLayer` trait implementation

mod decode;
mod helpers;
mod init;
mod init_kernel_dispatch;
// `innerq_driver` calls the CUDA Driver API directly via `atlas_core::registry`,
// which is itself gated on the `cuda` feature. Mirror that gate here so the
// metal-only build of spark-model (`--no-default-features --features metal`)
// compiles on Apple Silicon without dragging in `atlas_core::registry`.
#[cfg(feature = "cuda")]
pub mod innerq_driver;
mod prefill;
mod prefill_weights;
mod trait_impl;
mod types;

#[cfg(feature = "cuda")]
pub use innerq_driver::InnerQDriver;
pub use types::{MlaWeights, Qwen3AttentionLayer};

#[cfg(feature = "cuda")]
use std::sync::OnceLock;

// Process-wide handle, populated at serve startup when `TURBO_INNERQ=N`
// is set. `OnceLock` matches Atlas's pattern for other singletons (kernel
// registry, EP comm). Kept here next to the driver itself so server code
// just does `qwen3_attention::INNERQ.get()`.
#[cfg(feature = "cuda")]
pub static INNERQ: OnceLock<InnerQDriver> = OnceLock::new();
