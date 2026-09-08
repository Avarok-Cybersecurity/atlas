// SPDX-License-Identifier: AGPL-3.0-only

//! GPU handle types (`DevicePtr`, `KernelHandle`, `GraphHandle`,
//! `KernelArg`) and the free-memory baseline accessors. Split out of `gpu.rs`
//! for the ≤500 LoC cap; `gpu.rs` re-exports everything here, so the public
//! paths (`spark_runtime::gpu::DevicePtr` etc.) are unchanged.

use std::fmt;
use std::sync::atomic::Ordering;

// The free-memory baseline is a field of the single run mailbox,
// `crate::run_metrics::RunMetrics`: it is read by the dashboard and by KV
// sizing from threads with no carrier, and it is cleared at run start so a
// second model measures against its own baseline rather than the first
// model's pre-load free memory.

/// Record the free-memory baseline at GPU-context init. Call once, early,
/// before weight loading. Idempotent-last-write; intended to be set exactly once.
pub fn set_baseline_free_bytes(bytes: usize) {
    crate::run_metrics::metrics()
        .baseline_free_bytes
        .store(bytes, Ordering::Relaxed);
}

/// The free-memory baseline captured at context init, or `None` if never set.
pub fn baseline_free_bytes() -> Option<usize> {
    match crate::run_metrics::metrics()
        .baseline_free_bytes
        .load(Ordering::Relaxed)
    {
        0 => None,
        v => Some(v),
    }
}

/// Opaque device pointer wrapping a CUDA CUdeviceptr (u64).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    pub const NULL: Self = Self(0);

    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Byte offset from this pointer.
    pub fn offset(self, bytes: usize) -> Self {
        Self(self.0 + bytes as u64)
    }
}

/// Handle to a loaded CUDA kernel function.
#[derive(Debug, Clone, Copy)]
pub struct KernelHandle(pub u64);

/// Handle to an instantiated CUDA graph (CUgraphExec).
#[derive(Debug, Clone, Copy)]
pub struct GraphHandle(pub u64);

/// Typed kernel argument, used by `launch_typed`.
///
/// CUDA's `cuLaunchKernel` is type-blind — every arg is `void*` and the
/// driver interprets bytes by kernel signature. Metal's
/// `MTLComputeCommandEncoder` is not: buffer arguments require
/// `setBuffer:offset:atIndex:` (the encoder tracks the resource) while
/// scalar/struct args require `setBytes:length:atIndex:`. `KernelArg`
/// preserves that distinction so both backends can dispatch correctly.
#[derive(Debug, Clone, Copy)]
pub enum KernelArg<'a> {
    /// A device buffer at this base GPU address. The metal backend
    /// resolves it to its owning `MTLBuffer` + offset via the alloc
    /// registry; the cuda backend forwards the raw `u64` to the driver.
    Buffer(DevicePtr),
    /// Inline scalar/struct bytes, e.g. a `u32` count or an `f32` eps.
    /// Length is forwarded to Metal's `setBytes:length:`; the cuda
    /// backend zero-pads up to 8 bytes per slot.
    Bytes(&'a [u8]),
}

impl fmt::Display for DevicePtr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DevicePtr(0x{:x})", self.0)
    }
}
