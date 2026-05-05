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
mod prefill;
mod prefill_weights;
mod trait_impl;
mod types;

pub use types::{MlaWeights, Qwen3AttentionLayer};
