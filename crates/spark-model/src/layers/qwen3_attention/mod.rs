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
pub mod innerq_driver;
mod prefill;
mod prefill_weights;
mod trait_impl;
mod types;

pub use innerq_driver::InnerQDriver;
pub use types::{MlaWeights, Qwen3AttentionLayer};

use std::sync::OnceLock;

// Process-wide handle, populated at serve startup when `TURBO_INNERQ=N`
// is set. `OnceLock` matches Atlas's pattern for other singletons (kernel
// registry, EP comm). Kept here next to the driver itself so server code
// just does `qwen3_attention::INNERQ.get()`.
pub static INNERQ: OnceLock<InnerQDriver> = OnceLock::new();
