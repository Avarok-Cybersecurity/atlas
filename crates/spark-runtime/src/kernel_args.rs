// SPDX-License-Identifier: AGPL-3.0-only

//! Type-safe kernel argument builder for CUDA + Metal kernel launches.
//!
//! Replaces manual `Vec<*mut c_void>` construction with a builder
//! pattern that prevents parameter type/order mismatches AND records
//! per-arg type information so the metal backend can dispatch buffer
//! args via `setBuffer:offset:atIndex:` and scalar args via
//! `setBytes:length:atIndex:` (cuda's untyped `cuLaunchKernel` cannot
//! distinguish the two; metal cannot conflate them).
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
//! Internally, every arg is recorded with its kind (Buffer / Scalar)
//! and its native byte width. `launch()` materializes a typed
//! `KernelArg` slice and calls `GpuBackend::launch_typed`. The
//! cuda backend's default `launch_typed` impl flattens that back
//! into the legacy `void**` shape; the metal backend overrides
//! `launch_typed` to thread the type info through to the encoder.

use anyhow::Result;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

/// Per-arg metadata: which slot of `storage` it lives at, and how
/// many native bytes it occupies. Buffer args set `is_buffer = true`
/// and ignore `byte_len`. Scalar args set `is_buffer = false` and
/// record `byte_len = sizeof::<T>()`.
struct ArgKind {
    is_buffer: bool,
    /// Byte count for scalar args (4 for u32/i32/f32, 8 for u64).
    /// Unused when `is_buffer` is true.
    byte_len: u8,
}

/// Builder for type-safe kernel launches across CUDA + Metal.
///
/// Accumulates grid dimensions, block dimensions, and typed kernel
/// arguments. `launch()` packages the args as `&[KernelArg]` and
/// calls `GpuBackend::launch_typed`.
pub struct KernelLaunch<'a> {
    gpu: &'a dyn GpuBackend,
    kernel: KernelHandle,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    /// Backing storage: each parameter's bytes stored in a u64 slot
    /// (LE-packed for scalars; raw u64 GPU address for pointers).
    /// Pointers into this vec remain stable because we never
    /// reallocate after the initial capacity reservation.
    storage: Vec<u64>,
    /// Parallel array recording per-arg kind so `launch()` can build
    /// a typed `KernelArg` slice.
    kinds: Vec<ArgKind>,
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
            kinds: Vec::with_capacity(16),
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
        self.kinds.push(ArgKind {
            is_buffer: true,
            byte_len: 0,
        });
        self
    }

    /// Add a u32 argument.
    pub fn arg_u32(mut self, v: u32) -> Self {
        self.storage.push(v as u64);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 4,
        });
        self
    }

    /// Add a u64 argument.
    pub fn arg_u64(mut self, v: u64) -> Self {
        self.storage.push(v);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 8,
        });
        self
    }

    /// Add an i32 argument.
    pub fn arg_i32(mut self, v: i32) -> Self {
        // Store as u64, preserving the i32 bits in the low 4 bytes.
        self.storage.push(v as u32 as u64);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 4,
        });
        self
    }

    /// Add an f32 argument.
    pub fn arg_f32(mut self, v: f32) -> Self {
        self.storage.push(f32::to_bits(v) as u64);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 4,
        });
        self
    }

    /// Execute the kernel launch via `GpuBackend::launch_typed`.
    ///
    /// Builds a typed `KernelArg` slice from the recorded storage +
    /// kinds. The cuda backend's default `launch_typed` flattens this
    /// back into the legacy `void**` shape; the metal backend
    /// overrides `launch_typed` to use `setBuffer:` / `setBytes:` per
    /// arg. The storage vec is not reallocated between building the
    /// args and launching, so all byte slices remain valid.
    pub fn launch(self, stream: u64) -> Result<()> {
        // Build typed args. The `&[u8]` slices borrow from `self.storage`
        // (specifically the low N bytes of each u64 slot, LE-packed).
        let mut args: Vec<KernelArg<'_>> = Vec::with_capacity(self.kinds.len());
        for (idx, kind) in self.kinds.iter().enumerate() {
            let slot = &self.storage[idx];
            if kind.is_buffer {
                args.push(KernelArg::Buffer(DevicePtr(*slot)));
            } else {
                // SAFETY: slot is a valid u64 in self.storage; we slice
                // its first `byte_len` bytes (LE) and the slice's
                // lifetime is bounded by the borrow of self.storage,
                // which lives until the end of this function.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        slot as *const u64 as *const u8,
                        kind.byte_len as usize,
                    )
                };
                args.push(KernelArg::Bytes(bytes));
            }
        }
        let r = self.gpu.launch_typed(
            self.kernel,
            self.grid,
            self.block,
            self.shared_mem,
            stream,
            &args,
        );
        // ATLAS_DEBUG_SYNC_KERNELS (PCND, default-off): synchronize after
        // each launch so an async CUDA fault surfaces AT the culprit launch
        // (with grid/block) instead of at a later, unrelated sync point.
        // Diagnostic only — leave unset in production (one stream sync per
        // launch is a large slowdown). With RUST_BACKTRACE=1 the propagated
        // error pinpoints the calling op.
        if r.is_ok() && debug_sync_kernels() {
            self.gpu.synchronize(stream).map_err(|e| {
                let bt = std::backtrace::Backtrace::force_capture();
                anyhow::anyhow!(
                    "ATLAS_DEBUG_SYNC_KERNELS: async GPU fault immediately after kernel launch \
                     grid={:?} block={:?} shared_mem={}: {e}\nLAUNCH BACKTRACE:\n{bt}",
                    self.grid,
                    self.block,
                    self.shared_mem
                )
            })?;
        }
        r
    }
}

/// One-time read of `ATLAS_DEBUG_SYNC_KERNELS` (PCND opt-in). When `1`,
/// `KernelLaunch::launch` synchronizes the stream after every launch so an
/// asynchronous illegal-address fault is reported at the exact kernel that
/// caused it (rather than surfacing at a later sync). Default-off.
fn debug_sync_kernels() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("ATLAS_DEBUG_SYNC_KERNELS").as_deref() == Ok("1"))
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
