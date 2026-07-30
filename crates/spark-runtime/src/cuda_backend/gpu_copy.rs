// SPDX-License-Identifier: AGPL-3.0-only

//! The synchronous host/device copy family, as inherent methods.
//!
//! `impl GpuBackend for AtlasCudaBackend` cannot be split across files —
//! Rust requires one block per impl — so the trait body in `gpu_impl.rs`
//! delegates these four to the implementations here. They carry an `_impl`
//! suffix so an inherent method can never silently shadow the trait method
//! at a concrete-typed call site. They are the natural
//! group to move: same shape, same error handling, and each is a bounds
//! check plus one driver call.
//!
//! The safety contract is the one documented in `gpu_impl.rs`: a primary
//! context is current on the calling thread, every `DevicePtr` came from a
//! live allocation, and byte counts are exact.

use std::ffi::c_void;

use anyhow::{Result, bail};
use atlas_core::registry::cuda_error_text;

use super::{
    AtlasCudaBackend, cuMemcpyDtoDAsync_v2, cuMemcpyDtoHAsync_v2, cuMemcpyHtoDAsync_v2,
    cuStreamSynchronize,
};
use crate::gpu::DevicePtr;

impl AtlasCudaBackend {
    pub(crate) fn copy_h2d_impl(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        let status = unsafe {
            cuMemcpyHtoDAsync_v2(
                dst.0,
                src.as_ptr() as *const c_void,
                src.len(),
                self.default_stream,
            )
        };
        if status != 0 {
            bail!("cuMemcpyHtoDAsync_v2 failed: status {status}");
        }
        // Synchronize to ensure the copy completes before host buffer is freed.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after H2D failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    pub(crate) fn copy_d2h_impl(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr() as *mut c_void,
                src.0,
                dst.len(),
                self.default_stream,
            )
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2H failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    pub(crate) fn copy_d2h_on_stream_impl(
        &self,
        src: DevicePtr,
        dst: &mut [u8],
        stream: u64,
    ) -> Result<()> {
        // Enqueue the copy on the caller's stream so CUDA orders it after
        // any prior kernel launches on the same stream. Without this, the
        // copy may run on the default stream concurrently with kernels on
        // `stream` and read torn bytes (HSS Turbo8 race, 2026-04-28).
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(dst.as_mut_ptr() as *mut c_void, src.0, dst.len(), stream)
        };
        if status != 0 {
            bail!("cuMemcpyDtoHAsync_v2 (on_stream) failed: status {status}");
        }
        let sync = unsafe { cuStreamSynchronize(stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2H on_stream failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }

    pub(crate) fn copy_d2d_impl(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        let status = unsafe { cuMemcpyDtoDAsync_v2(dst.0, src.0, bytes, self.default_stream) };
        if status != 0 {
            bail!("cuMemcpyDtoDAsync_v2 failed: status {status}");
        }
        // Synchronize to ensure copy completes before kernels on other streams read it.
        let sync = unsafe { cuStreamSynchronize(self.default_stream) };
        if sync != 0 {
            bail!(
                "cuStreamSynchronize after D2D failed: {}",
                cuda_error_text(sync)
            );
        }
        Ok(())
    }
}
