// SPDX-License-Identifier: AGPL-3.0-only

//! TMA descriptors (`CUtensorMap`) for `cp.async.bulk.tensor` loads.
//!
//! A TMA descriptor moves address generation, tiling and BOUNDS CHECKING for a
//! global→shared copy into hardware: the kernel supplies tile coordinates and
//! the copy engine does the rest, with out-of-range elements zero-filled instead
//! of masked by hand. That removes the per-row index arithmetic, the tail
//! predication and the register round-trip that `cp.async` still pays.
//!
//! ## Why this exists
//!
//! `kernels/gb10/common/gated_delta_rule_fla.cu` claimed "GB10 sm_121 has
//! cp.async.cg (NO TMA)". That is false: `cp.async.bulk.tensor` + `mbarrier`
//! compile for `sm_121a` under CUDA 13.0, and the FlashQLA GDN spine measured on
//! a GB10 at 15.0 ms uses exactly that against our 105.7 ms. Nothing in
//! `kernels/` used TMA before this module.
//!
//! ## Contract (the parts that bite)
//!
//! * The descriptor is **128 bytes aligned to 64** — hence `repr(C, align(64))`.
//!   It is passed BY VALUE to a `__grid_constant__ const CUtensorMap` parameter,
//!   which is why `KernelLaunch::arg_tensormap` has to occupy 16 slots.
//! * `global_strides` carries `rank - 1` entries **in bytes**. The innermost
//!   stride is implicit and must equal the element size, so a tensor whose fast
//!   axis is not contiguous cannot be described.
//! * Every stride must be 16-byte aligned, and so must `global_address`.
//! * Dimensions are in ELEMENTS and are ordered fastest-varying FIRST — the
//!   opposite of the row-major `[rows][cols]` we write everywhere else. Getting
//!   this backwards does not error; it silently transposes the load.

use anyhow::{Result, bail};

use crate::gpu::DevicePtr;

/// Verified against `/usr/local/cuda/include/cuda.h` (CUDA 13.0). Note
/// `BFLOAT16 = 9`: `FLOAT64` sits at 8, ahead of it, and `FLOAT32_FTZ` is 10 —
/// the order most references get wrong. An incorrect dtype here does not fail,
/// it reinterprets the bytes.
const CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: u32 = 9;
const CU_TENSOR_MAP_INTERLEAVE_NONE: u32 = 0;
const CU_TENSOR_MAP_SWIZZLE_NONE: u32 = 0;
const CU_TENSOR_MAP_L2_PROMOTION_L2_128B: u32 = 2;
const CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE: u32 = 0;

/// `cuTensorMapEncodeTiled`, looked up once in the already-loaded libcuda.
///
/// `RTLD_DEFAULT` searches the global scope, so this finds the driver the
/// process has already loaded rather than opening a second copy. Cached in a
/// `OnceLock`: the lookup is cheap but not free, and descriptor construction sits
/// on the prefill path.
type EncodeTiledFn = unsafe extern "C" fn(
    *mut u8,
    u32,
    u32,
    u64,
    *const u64,
    *const u64,
    *const u32,
    *const u32,
    u32,
    u32,
    u32,
    u32,
) -> i32;

fn tensor_map_encode_tiled() -> Option<EncodeTiledFn> {
    static CACHED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let addr = (*CACHED.get_or_init(|| {
        // `dlsym` exists only on unix — on Windows this symbol has no libc
        // provider and the build DIED AT LINK (LNK2019 unresolved external
        // `dlsym`, seen on the windows-x86_64 release-matrix builds
        // 2026-08-22). TMA is opt-in (`ATLAS_GDN_TMA=1`), measured neutral,
        // and unresolved here just routes callers to their non-TMA path — so
        // Windows reporting "unavailable" is the honest and cheap behaviour,
        // not a loss of function.
        #[cfg(unix)]
        {
            unsafe extern "C" {
                fn dlsym(handle: *mut std::ffi::c_void, symbol: *const i8)
                -> *mut std::ffi::c_void;
            }
            let name = c"cuTensorMapEncodeTiled";
            let p = unsafe { dlsym(std::ptr::null_mut(), name.as_ptr().cast::<i8>()) };
            if p.is_null() { None } else { Some(p as usize) }
        }
        #[cfg(not(unix))]
        {
            None
        }
    }))?;
    // SAFETY: the symbol resolved from libcuda has exactly this signature; it is
    // the documented prototype for cuTensorMapEncodeTiled.
    Some(unsafe { std::mem::transmute::<usize, EncodeTiledFn>(addr) })
}

/// A 128-byte TMA descriptor, aligned as the driver requires.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TensorMap([u8; 128]);

impl std::fmt::Debug for TensorMap {
    /// The 128 bytes are an opaque driver-owned blob; printing them is noise.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TensorMap(<128 bytes>)")
    }
}

impl TensorMap {
    /// The raw bytes, for `KernelLaunch::arg_tensormap`.
    pub fn bytes(&self) -> &[u8; 128] {
        &self.0
    }

    /// Describe a row-major bf16 matrix as tiles of `box_rows x box_cols`.
    ///
    /// `rows`/`cols` and `row_stride_elems` are in ELEMENTS, in the row-major
    /// terms the rest of the codebase uses; the fastest-varying-first ordering
    /// the driver wants is applied here so callers never have to think about it.
    /// `row_stride_elems` is the distance between consecutive rows and may
    /// exceed `cols` (a view into a wider tensor).
    ///
    /// Loads of tiles that hang off the end are ZERO-FILLED by hardware, so the
    /// caller does not predicate the tail.
    pub fn tiled_2d_bf16(
        global: DevicePtr,
        rows: u64,
        cols: u64,
        row_stride_elems: u64,
        box_rows: u32,
        box_cols: u32,
    ) -> Result<Self> {
        // Fail fast on the alignment rules rather than let the driver return a
        // generic invalid-argument, or worse, succeed and mis-address.
        if !global.0.is_multiple_of(16) {
            bail!("TMA global address {:#x} is not 16-byte aligned", global.0);
        }
        let row_stride_bytes = row_stride_elems * 2;
        if !row_stride_bytes.is_multiple_of(16) {
            bail!(
                "TMA row stride {row_stride_elems} elems ({row_stride_bytes} B) is not \
                 16-byte aligned; bf16 needs a stride that is a multiple of 8 elements"
            );
        }
        if row_stride_elems < cols {
            bail!("TMA row stride {row_stride_elems} is narrower than cols {cols}");
        }
        if box_rows == 0 || box_cols == 0 {
            bail!("TMA box dims must be non-zero, got {box_rows}x{box_cols}");
        }

        // Fastest-varying axis FIRST: for row-major [rows][cols] that is cols.
        let global_dim: [u64; 2] = [cols, rows];
        // rank-1 strides, in bytes, skipping the implicit innermost one.
        let global_strides: [u64; 1] = [row_stride_bytes];
        let box_dim: [u32; 2] = [box_cols, box_rows];
        let element_strides: [u32; 2] = [1, 1];

        // ★ RESOLVED AT RUNTIME, NOT LINKED. `cuTensorMapEncodeTiled` is a CUDA
        // 12.0+ driver entry point, and a link-time `extern "C"` makes the whole
        // workspace fail to LINK anywhere the available libcuda stub predates it:
        //
        //   rust-lld: error: undefined symbol: cuTensorMapEncodeTiled
        //
        // which is what CI hit while a local build with CUDA 13.0 linked fine.
        // `dlsym` also degrades honestly — a driver without the symbol yields a
        // clear error here and the caller falls back to its non-TMA path, rather
        // than the binary refusing to build for everyone.
        let f = tensor_map_encode_tiled().ok_or_else(|| {
            anyhow::anyhow!(
                "cuTensorMapEncodeTiled not present in libcuda — TMA needs a CUDA 12.0+ driver"
            )
        })?;
        let mut map = TensorMap([0u8; 128]);
        let rc = unsafe {
            f(
                map.0.as_mut_ptr(),
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                2,
                global.0,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                element_strides.as_ptr(),
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        if rc != 0 {
            bail!(
                "cuTensorMapEncodeTiled failed: CUresult {rc} \
                 (rows={rows} cols={cols} stride={row_stride_elems} box={box_rows}x{box_cols})"
            );
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The alignment guards must reject BEFORE calling the driver — these run on
    /// a box with no CUDA context, which is exactly the point: a misaligned
    /// stride is a caller bug, not a driver outcome.
    #[test]
    fn misaligned_stride_is_rejected_without_touching_the_driver() {
        // 5 bf16 elements = 10 bytes: not a multiple of 16.
        let e = TensorMap::tiled_2d_bf16(DevicePtr(0x1000), 4, 5, 5, 2, 5).unwrap_err();
        assert!(
            e.to_string().contains("not 16-byte aligned"),
            "expected a stride-alignment error, got: {e}"
        );
    }

    #[test]
    fn misaligned_address_is_rejected() {
        let e = TensorMap::tiled_2d_bf16(DevicePtr(0x1004), 4, 8, 8, 2, 8).unwrap_err();
        assert!(e.to_string().contains("not 16-byte aligned"), "got: {e}");
    }

    #[test]
    fn a_stride_narrower_than_the_row_is_rejected() {
        let e = TensorMap::tiled_2d_bf16(DevicePtr(0x1000), 4, 64, 32, 2, 64).unwrap_err();
        assert!(e.to_string().contains("narrower than cols"), "got: {e}");
    }

    /// 128 bytes at 64-byte alignment is a hard driver requirement, and getting
    /// it wrong is an invalid-argument at encode time or corruption at launch.
    #[test]
    fn the_descriptor_has_the_layout_the_driver_requires() {
        assert_eq!(std::mem::size_of::<TensorMap>(), 128);
        assert_eq!(std::mem::align_of::<TensorMap>(), 64);
    }
}
