// SPDX-License-Identifier: AGPL-3.0-only

use atlas_core::error::Result;
use atlas_core::tensor::TensorRef;

/// Quantize activations from a higher-precision format to a lower one.
pub trait Quantize {
    /// Quantize input tensor in-place or to output tensor.
    /// `scale` is written with the computed per-block scale factors.
    fn quantize(
        &self,
        input: &TensorRef,
        output: &TensorRef,
        scale: &TensorRef,
        stream_ptr: u64,
    ) -> Result<()>;
}

/// Dequantize weights or activations from a lower-precision format.
pub trait Dequantize {
    /// Dequantize input tensor to output tensor using provided scale factors.
    fn dequantize(
        &self,
        input: &TensorRef,
        output: &TensorRef,
        scale: &TensorRef,
        stream_ptr: u64,
    ) -> Result<()>;
}
