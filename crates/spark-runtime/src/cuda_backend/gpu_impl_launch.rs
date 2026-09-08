// SPDX-License-Identifier: AGPL-3.0-only

//! Cooperative kernel launch and the dynamic-smem attribute raise for
//! [`AtlasCudaBackend`].
//!
//! Trimmed to these two on import: this branch's `gpu_impl.rs` already
//! implements the stream-capture probe and the pitched D2D copy inline, and
//! a second copy here would be dead code the crate's deny(warnings) rejects.
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
use crate::gpu::KernelHandle;

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
}
