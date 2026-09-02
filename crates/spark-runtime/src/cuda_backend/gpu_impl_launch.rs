// SPDX-License-Identifier: AGPL-3.0-only

//! Cooperative kernel launch, the dynamic-smem attribute raise, the
//! stream-capture probe and the pitched D2D copy for [`AtlasCudaBackend`].
//!
//! Split out of `gpu_impl.rs` to keep both files under the repo's 500-LoC cap,
//! the same way `gpu_impl_graph.rs` was: these are the inherent bodies
//! (suffixed `_cu` so a delegator can never self-recurse) and the
//! `GpuBackend` impl next door is a one-line delegator to each.
//!
//! The `unsafe` safety contract documented at the top of `gpu_impl.rs` applies
//! verbatim to every driver call here.

use std::ffi::c_void;

use anyhow::{Result, bail};
use atlas_core::registry::cuda_error_text;

use super::AtlasCudaBackend;
use crate::gpu::{DevicePtr, KernelHandle};

impl AtlasCudaBackend {
    pub(super) fn launch_cooperative_cu(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        // SCALE's libcuda does not export cuLaunchCooperativeKernel (see the
        // extern block in cuda_backend.rs); refusing is correct — a fallback
        // to cuLaunchKernel would let the kernel's grid.sync() deadlock.
        #[cfg(atlas_scale)]
        {
            let _ = (func, grid, block, shared_mem, stream, params);
            bail!("launch_cooperative: not available under SCALE (gfx1151)");
        }
        #[cfg(not(atlas_scale))]
        {
            let status = unsafe {
                super::cuLaunchCooperativeKernel(
                    func.0 as *mut c_void,
                    grid[0],
                    grid[1],
                    grid[2],
                    block[0],
                    block[1],
                    block[2],
                    shared_mem,
                    stream,
                    params.as_mut_ptr(),
                )
            };
            if status != 0 {
                let msg = format!(
                    "cuLaunchCooperativeKernel failed: {} (grid={:?}, block={:?}, \
                     shared_mem={shared_mem}) — a too-large grid (blocks exceed what \
                     co-residency allows) or an un-raised dynamic-smem cap (see \
                     set_kernel_max_dynamic_smem) both land here",
                    cuda_error_text(status),
                    grid,
                    block,
                );
                // Same probe-and-latch as `launch`: a failed launch may have
                // destroyed the context, and the caller's error string alone
                // cannot say so.
                super::fault_probe::note_failure("cooperative kernel launch", &msg);
                bail!(msg);
            }
            Ok(())
        }
    }

    pub(super) fn set_kernel_max_dynamic_smem_cu(
        &self,
        kernel: KernelHandle,
        bytes: usize,
    ) -> Result<()> {
        const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
        let status = unsafe {
            super::cuFuncSetAttribute(
                kernel.0 as *mut c_void,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(bytes)
                    .map_err(|_| anyhow::anyhow!("dynamic smem request {bytes} overflows i32"))?,
            )
        };
        if status != 0 {
            bail!(
                "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={bytes}) failed: {}",
                cuda_error_text(status)
            );
        }
        Ok(())
    }

    pub(super) fn stream_is_capturing_cu(&self, stream: u64) -> bool {
        // SCALE's libcuda does not export cuStreamIsCapturing; report
        // not-capturing there (gfx1151 telemetry taps then sample eagerly —
        // acceptable for a default-off measurement knob).
        #[cfg(atlas_scale)]
        {
            let _ = stream;
            false
        }
        #[cfg(not(atlas_scale))]
        {
            let mut status: u32 = 0;
            // CU_STREAM_CAPTURE_STATUS_NONE = 0; treat query failure as
            // capturing (conservative: the tap skips its sample).
            let rc = unsafe { super::cuStreamIsCapturing(stream, &mut status) };
            rc != 0 || status != 0
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn copy_d2d_2d_async_cu(
        &self,
        src: DevicePtr,
        src_pitch: usize,
        dst: DevicePtr,
        dst_pitch: usize,
        width_bytes: usize,
        height: usize,
        stream: u64,
    ) -> Result<()> {
        // One pitched copy (cudaMemcpyDeviceToDevice = 3) on the caller's stream,
        // replacing a per-row copy_d2d_async loop. cudart is linked (cutlass/
        // flashinfer use the runtime API); a CUstream handle is a valid
        // cudaStream_t.
        unsafe extern "C" {
            fn cudaMemcpy2DAsync(
                dst: *mut c_void,
                dpitch: usize,
                src: *const c_void,
                spitch: usize,
                width: usize,
                height: usize,
                kind: i32,
                stream: u64,
            ) -> i32;
        }
        let status = unsafe {
            cudaMemcpy2DAsync(
                dst.0 as *mut c_void,
                dst_pitch,
                src.0 as *const c_void,
                src_pitch,
                width_bytes,
                height,
                3,
                stream,
            )
        };
        if status != 0 {
            bail!("cudaMemcpy2DAsync failed: status {status}");
        }
        Ok(())
    }
}
