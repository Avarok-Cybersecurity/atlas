// SPDX-License-Identifier: AGPL-3.0-only

use crate::dtype::DType;

/// A zero-copy reference to a GPU tensor, typically from PyTorch via data_ptr().
///
/// Atlas never allocates or copies tensor data — it receives raw CUDA device
/// pointers from Python and wraps them for kernel launches.
#[derive(Debug, Clone)]
pub struct TensorRef {
    /// Raw CUDA device pointer (from torch.Tensor.data_ptr())
    pub ptr: u64,

    /// Shape dimensions (e.g., [batch, seq_len, hidden_size])
    pub shape: Vec<usize>,

    /// Strides in elements (not bytes)
    pub strides: Vec<usize>,

    /// Element data type
    pub dtype: DType,
}

impl TensorRef {
    /// Create a new tensor reference from a raw pointer and shape.
    /// Assumes contiguous (row-major) layout.
    pub fn new(ptr: u64, shape: Vec<usize>, dtype: DType) -> Self {
        let strides = Self::contiguous_strides(&shape);
        Self {
            ptr,
            shape,
            strides,
            dtype,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        let bits = self.numel() * self.dtype.element_size_bits();
        bits.div_ceil(8) // round up for sub-byte types
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Compute contiguous (C-order / row-major) strides from shape.
    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Raw pointer cast to a typed device pointer (for kernel launches).
    pub fn as_device_ptr<T>(&self) -> *const T {
        self.ptr as *const T
    }

    /// Mutable raw pointer cast.
    pub fn as_device_ptr_mut<T>(&self) -> *mut T {
        self.ptr as *mut T
    }
}
