// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::error::Result;

/// Normalization backend trait.
#[allow(clippy::too_many_arguments)]
pub trait Normalize {
    /// RMS normalization: y = x * weight / rms(x)
    fn rms_norm(
        &self,
        input_ptr: u64,
        weight_ptr: u64,
        output_ptr: u64,
        num_tokens: u32,
        hidden_size: u32,
        eps: f32,
        stream_ptr: u64,
    ) -> Result<()>;

    /// Gated RMS normalization (for Mamba): y = silu(gate) * rms_norm(x)
    fn gated_rms_norm(
        &self,
        input_ptr: u64,
        gate_ptr: u64,
        weight_ptr: u64,
        output_ptr: u64,
        num_tokens: u32,
        hidden_size: u32,
        eps: f32,
        stream_ptr: u64,
    ) -> Result<()>;
}
