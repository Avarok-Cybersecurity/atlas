// SPDX-License-Identifier: AGPL-3.0-only
//
// Pitched (2-D) device-to-device copy, extracted from `gpu_impl.rs`.
//
// It lives in its own module for two reasons. It is the ONLY method on the
// backend whose implementation forks on the GPU vendor — NVIDIA issues a
// single `cudaMemcpy2DAsync` through the cudart runtime, while the AMD
// (SCALE / native-HIP) targets have no cudart linked and must fall back to a
// per-row driver-API loop. Keeping both arms here leaves the trait body in
// `gpu_impl.rs` uniform, and keeps that file under the repo's 500-LoC cap.
//
// The safety contract is the one documented at the top of `gpu_impl.rs`:
// context bound, pointers from a prior successful allocation, byte-exact
// sizes, stream owned by `Self`.

use anyhow::Result;

use super::AtlasCudaBackend;
use crate::gpu::DevicePtr;

/// NVIDIA: one pitched copy (`cudaMemcpyDeviceToDevice` = 3) on the caller's
/// stream via the cudart runtime, replacing a per-row `copy_d2d_async` loop.
/// cudart is linked (cutlass/flashinfer use the runtime API); a `CUstream`
/// handle is a valid `cudaStream_t`.
#[cfg(not(atlas_scale))]
#[allow(clippy::too_many_arguments)] // mirrors the GpuBackend trait method's arity
pub(super) fn copy_d2d_2d_async(
    _backend: &AtlasCudaBackend,
    src: DevicePtr,
    src_pitch: usize,
    dst: DevicePtr,
    dst_pitch: usize,
    width_bytes: usize,
    height: usize,
    stream: u64,
) -> Result<()> {
    use std::ffi::c_void;

    use anyhow::bail;

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

/// strix/SCALE and native-HIP: no cudart runtime is linked (SCALE's libcuda is
/// driver-only, and the HIP shim exports the driver surface), so fall back to
/// the per-row driver-API loop this pitched copy replaced. `copy_d2d_async`
/// uses `cuMemcpyDtoDAsync`, which both SCALE and the HIP shim provide.
#[cfg(atlas_scale)]
#[allow(clippy::too_many_arguments)] // mirrors the GpuBackend trait method's arity
pub(super) fn copy_d2d_2d_async(
    backend: &AtlasCudaBackend,
    src: DevicePtr,
    src_pitch: usize,
    dst: DevicePtr,
    dst_pitch: usize,
    width_bytes: usize,
    height: usize,
    stream: u64,
) -> Result<()> {
    use crate::gpu::GpuBackend;

    for row in 0..height {
        let s = DevicePtr(src.0 + (row * src_pitch) as u64);
        let d = DevicePtr(dst.0 + (row * dst_pitch) as u64);
        backend.copy_d2d_async(s, d, width_bytes, stream)?;
    }
    Ok(())
}
