// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::error::Result;

/// Activation function backend trait.
pub trait Activation {
    /// Fused SiLU(gate) * up: output = silu(gate) * up
    fn silu_mul(
        &self,
        gate_ptr: u64,
        up_ptr: u64,
        output_ptr: u64,
        num_elements: u32,
        stream_ptr: u64,
    ) -> Result<()>;

    /// Fused SiLU(gate) * up + FP4 quantize: output = quant(silu(gate) * up)
    #[allow(clippy::too_many_arguments)]
    fn silu_mul_quant(
        &self,
        gate_ptr: u64,
        up_ptr: u64,
        output_ptr: u64,
        scale_ptr: u64,
        num_elements: u32,
        group_size: u32,
        stream_ptr: u64,
    ) -> Result<()>;
}
