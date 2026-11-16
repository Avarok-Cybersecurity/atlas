// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The
// HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors
// to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas. Layer structure
// after `Lfm3DecoderLayer.forward` (sandwich norms) and vLLM `short_conv.py` (the conv layer as a
// recurrent layer with a [hidden, 6] state).

//! Responsibility: the LFM3 gated short-convolution layer (`Lfm3ConvLayer`):
//! its configuration and weights, the per-sequence state it owns, and the
//! module tree that implements it (port lane L38).
//!
//! One LFM3 conv layer computes, per token (sandwich norms, `use_sandwich_norm`):
//!
//! ```text
//! a   = operator_norm(h)                     RMSNorm, weight * x (no 1 + w), eps 1e-5
//! BCx = in_proj(a)                           [3 * hidden], chunk order B | C | x
//! y   = C * conv7(B * x)                     the kernels of kernels/ada/common/lfm3_shortconv_ada.cu
//! h   = h + post_operator_norm(out_proj(y))
//! h   = h + post_ffn_norm(ffn(ffn_norm(h)))  dense SwiGLU (FfnComponent::Dense)
//! ```
//!
//! STATE. Each sequence owns one `[hidden, 6]` BF16 window per conv layer
//! (`Lfm3ConvState`, 24 KB at hidden 2048): the last six `u = B * x` values,
//! oldest first. It is NOT GDN-shaped, so the layer opts out of the SSM pool
//! (`uses_ssm_pool() == false`) and allocates its own state; the model reports
//! it as `LayerType::LinearAttention` (L37's config mapping), and pool-shaped
//! model loops must skip it (REQUESTS.md of lane L38).
//!
//! GRAPHS. Decode issues no host sync and passes the state pointers as kernel
//! parameters, so it is capturable. Because the state is allocated per
//! sequence, a slot's captured graphs bake that sequence's addresses; the layer
//! reports `graph_stale_on_new_sequence()`, and `free_sequence` drops them.
//!
//! SPECULATION. The window is not rewound by lowering the KV cursor and there
//! is no per-token checkpoint ring, so rollback and the batched verify sweep
//! are declined (`decode_rollback_unsupported`, `decode_verify_multi_unsupported`).
//!
//! PREFIX CACHE. `snapshot_aux` / `restore_aux` carry the window with a Marconi
//! snapshot taken at a chunk boundary (`state.rs` has the blob format).

use crate::layers::FfnComponent;
use crate::weight_map::{DenseWeight, Fp8Weight};
use spark_runtime::gpu::DevicePtr;

mod decode;
mod init;
pub mod launch;
mod prefill;
pub mod reference;
pub mod state;
mod trait_impl;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_mock;

pub use launch::ConvKernels;
pub use state::Lfm3ConvState;

/// Taps of the depthwise convolution (`conv_L_cache`).
pub const CONV_WIDTH: usize = 7;
/// Values of the per-channel decode window (`CONV_WIDTH - 1`).
pub const STATE_LEN: usize = CONV_WIDTH - 1;

/// Geometry and numerics of one conv layer, built by the LFM3 loader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lfm3ConvCfg {
    /// Model width (2048 for LFM3-1.7B); in_proj emits `3 * hidden`.
    pub hidden: usize,
    /// Depthwise conv taps; only 7 is implemented (the kernels are 7-tap).
    pub conv_width: usize,
    /// RMSNorm epsilon of all four norms (`norm_eps`, 1e-5).
    pub eps: f32,
    /// Index of this layer in the model (logs and errors only).
    pub layer_idx: usize,
}

impl Lfm3ConvCfg {
    /// The conv geometry of a parsed LFM3 config (`avarok-core` `parsers/lfm3.rs`
    /// stores `conv_L_cache` in `linear_conv_kernel_dim` and `norm_eps` in
    /// `rms_norm_eps`).
    pub fn from_model_config(config: &avarok_core::config::ModelConfig, layer_idx: usize) -> Self {
        Self {
            hidden: config.hidden_size,
            conv_width: config.linear_conv_kernel_dim,
            eps: config.rms_norm_eps as f32,
            layer_idx,
        }
    }

    /// BF16 bytes of one sequence's window (`[hidden, 6]`).
    pub fn state_bytes(&self) -> usize {
        self.hidden * STATE_LEN * 2
    }

    /// Width of the in_proj output row (`B | C | x`).
    pub fn bcx_width(&self) -> usize {
        3 * self.hidden
    }

    /// Refuse a geometry the kernels do not implement.
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.conv_width == CONV_WIDTH,
            "LFM3 conv layer {}: conv width {} is not supported (the kernels are {CONV_WIDTH}-tap)",
            self.layer_idx,
            self.conv_width
        );
        anyhow::ensure!(
            self.hidden > 0 && self.hidden.is_multiple_of(2),
            "LFM3 conv layer {}: hidden {} must be positive and even",
            self.layer_idx,
            self.hidden
        );
        anyhow::ensure!(
            u32::try_from(3 * self.hidden).is_ok(),
            "LFM3 conv layer {}: hidden {} overflows the kernel arguments",
            self.layer_idx,
            self.hidden
        );
        anyhow::ensure!(
            self.eps.is_finite() && self.eps > 0.0,
            "LFM3 conv layer {}: eps {} must be finite and positive",
            self.layer_idx,
            self.eps
        );
        Ok(())
    }
}

/// The checkpoint tensors of one conv layer, device-resident, BF16 unless
/// noted. Names are the HF tensor names under `model.layers.N.`.
pub struct Lfm3ConvWeights {
    /// `operator_norm.weight` `[hidden]` (plain weight, not 1 + w).
    pub operator_norm: DenseWeight,
    /// `conv.in_proj.weight` `[3 * hidden, hidden]`.
    pub in_proj: DenseWeight,
    /// `conv.conv.weight` `[hidden, 1, 7]`, used as `[hidden, 7]`.
    pub conv: DenseWeight,
    /// `conv.out_proj.weight` `[hidden, hidden]`.
    pub out_proj: DenseWeight,
    /// `post_operator_norm.weight` `[hidden]`.
    pub post_operator_norm: DenseWeight,
    /// `ffn_norm.weight` `[hidden]`.
    pub ffn_norm: DenseWeight,
    /// `feed_forward.{w1, w3, w2}` (gate, up, down) as a dense SwiGLU FFN.
    /// Must be `FfnComponent::Dense`.
    pub ffn: FfnComponent,
    /// `post_ffn_norm.weight` `[hidden]`.
    pub post_ffn_norm: DenseWeight,
}

/// Optional FP8 decode copies of the two conv projections (opt-in route of
/// lane L37, row `lfm3_fp8_all`): block-scaled E4M3 weights with FP32
/// `[ceil(N/128), ceil(K/128)]` scales. Decode GEMVs (1 to 16 rows) read them;
/// longer prefill chunks keep the BF16 weights on cuBLASLt.
#[derive(Clone, Copy, Debug)]
pub struct Lfm3ConvFp8 {
    pub in_proj: Fp8Weight,
    pub out_proj: Fp8Weight,
}

/// The LFM3 gated short-convolution layer. Built by [`Lfm3ConvLayer::new`]
/// (`init.rs`), driven through `TransformerLayer` (`trait_impl.rs`).
pub struct Lfm3ConvLayer {
    pub(crate) cfg: Lfm3ConvCfg,
    pub(crate) operator_norm: DenseWeight,
    pub(crate) in_proj: DenseWeight,
    pub(crate) conv: DenseWeight,
    pub(crate) out_proj: DenseWeight,
    pub(crate) post_operator_norm: DenseWeight,
    pub(crate) ffn_norm: DenseWeight,
    pub(crate) ffn: FfnComponent,
    pub(crate) post_ffn_norm: DenseWeight,
    pub(crate) fp8: Option<Lfm3ConvFp8>,
    pub(crate) k: init::LayerKernels,
    /// Write-only window for batch padding rows: the model pads a batched
    /// decode with pool-dummy `SsmLayerState`s for every `LinearAttention`
    /// layer, and those rows must not touch a real sequence's window.
    pub(crate) pad_state: DevicePtr,
}
