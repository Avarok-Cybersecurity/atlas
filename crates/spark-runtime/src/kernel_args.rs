// SPDX-License-Identifier: AGPL-3.0-only

//! Type-safe kernel argument builder for CUDA kernel launches.
//!
//! Replaces manual `Vec<*mut c_void>` construction with a builder
//! pattern that prevents parameter type/order mismatches.
//!
//! # Usage
//!
//! ```ignore
//! KernelLaunch::new(gpu, kernel)
//!     .grid([num_tokens, 1, 1])
//!     .block([256, 1, 1])
//!     .arg_ptr(input)
//!     .arg_u32(hidden_size)
//!     .arg_f32(eps)
//!     .launch(stream)?;
//! ```
//!
//! Values are stored as u64 for uniform alignment. On little-endian
//! systems (all CUDA targets), the low bytes of each u64 contain the
//! actual parameter data, which cuLaunchKernel reads correctly.

use anyhow::Result;

use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Builder for type-safe CUDA kernel launches.
///
/// Accumulates grid dimensions, block dimensions, and typed kernel
/// arguments. The `launch()` method constructs the raw `*mut c_void`
/// parameter array and calls `GpuBackend::launch()`.
pub struct KernelLaunch<'a> {
    gpu: &'a dyn GpuBackend,
    kernel: KernelHandle,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    /// Backing storage: each parameter value stored as u64.
    /// Pointers into this vec remain stable because we never
    /// reallocate after the initial capacity reservation.
    storage: Vec<u64>,
}

impl<'a> KernelLaunch<'a> {
    pub fn new(gpu: &'a dyn GpuBackend, kernel: KernelHandle) -> Self {
        Self {
            gpu,
            kernel,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_mem: 0,
            storage: Vec::with_capacity(16),
        }
    }

    pub fn grid(mut self, grid: [u32; 3]) -> Self {
        self.grid = grid;
        self
    }

    pub fn block(mut self, block: [u32; 3]) -> Self {
        self.block = block;
        self
    }

    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Add a DevicePtr (u64) argument.
    pub fn arg_ptr(mut self, p: DevicePtr) -> Self {
        self.storage.push(p.0);
        self
    }

    /// Add a u32 argument.
    pub fn arg_u32(mut self, v: u32) -> Self {
        self.storage.push(v as u64);
        self
    }

    /// Add a u64 argument.
    pub fn arg_u64(mut self, v: u64) -> Self {
        self.storage.push(v);
        self
    }

    /// Add an i32 argument.
    pub fn arg_i32(mut self, v: i32) -> Self {
        // Store as u64, preserving the i32 bits in the low 4 bytes.
        self.storage.push(v as u32 as u64);
        self
    }

    /// Add an f32 argument.
    pub fn arg_f32(mut self, v: f32) -> Self {
        self.storage.push(f32::to_bits(v) as u64);
        self
    }

    /// Execute the kernel launch.
    ///
    /// Builds the raw parameter pointer array from stored values,
    /// then calls `GpuBackend::launch()`. The storage vec is not
    /// reallocated between building pointers and launching, so
    /// all pointers remain valid.
    pub fn launch(self, stream: u64) -> Result<()> {
        let mut params: Vec<*mut std::ffi::c_void> = self
            .storage
            .iter()
            .map(|v| v as *const u64 as *mut std::ffi::c_void)
            .collect();
        self.gpu.launch(
            self.kernel,
            self.grid,
            self.block,
            self.shared_mem,
            stream,
            &mut params,
        )
    }
}

/// Convenience: divide and round up.
pub fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;

    #[test]
    fn test_kernel_launch_builder() {
        let gpu = MockGpuBackend::new();
        let kernel = gpu.kernel("test", "test_kernel").unwrap();

        let result = KernelLaunch::new(&gpu, kernel)
            .grid([4, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(DevicePtr(0x1000))
            .arg_u32(42)
            .arg_f32(1.5)
            .launch(0);

        assert!(result.is_ok());
        assert_eq!(gpu.launch_count(), 1);
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(div_ceil(10, 3), 4);
        assert_eq!(div_ceil(9, 3), 3);
        assert_eq!(div_ceil(1, 256), 1);
        assert_eq!(div_ceil(256, 256), 1);
        assert_eq!(div_ceil(257, 256), 2);
    }
}
