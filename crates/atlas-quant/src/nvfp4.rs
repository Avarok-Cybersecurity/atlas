// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::error::Result;
use atlas_core::tensor::TensorRef;

use crate::traits::Quantize;

/// NVFP4 quantization: FP32 activations → E2M1 with FP8 block scales.
///
/// Uses the branchless integer-comparison E2M1 conversion kernel.
/// Each float is converted via 7 unsigned comparisons on IEEE 754 bit patterns.
pub struct NvFp4Quantizer;

impl Quantize for NvFp4Quantizer {
    fn quantize(
        &self,
        _input: &TensorRef,
        _output: &TensorRef,
        _scale: &TensorRef,
        _stream_ptr: u64,
    ) -> Result<()> {
        // TODO: Launch e2m1_branchless.cu kernel via cudarc
        todo!("NVFP4 quantization kernel launch")
    }
}
