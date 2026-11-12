// SPDX-License-Identifier: AGPL-3.0-only

//! Launch-size floors for kernels that keep their tiles in DYNAMIC shared
//! memory and say how much they need.
//!
//! WHY. ptxas caps static `__shared__` at 48 KiB for every arch string without
//! an `a` suffix (docs/HARDWARE.md, "PTX arch suffix and the static
//! shared-memory ceiling"). A target with no `a` form, such as `kernels/ada`
//! (sm_89), therefore holds its 66-90 KB prefill attention tiles in dynamic
//! shared memory instead, and dynamic shared memory is sized by the LAUNCH. The
//! dispatch sites that launch those kernels pass no shared-memory size today,
//! and teaching each of them a per-target tile size would put a second copy of
//! every kernel's layout on the host.
//!
//! HOW. A kernel that needs a floor exports it beside its entry point:
//!
//! ```text
//! extern "C" __device__ const unsigned int <entry>_dyn_smem_bytes = sizeof(Tiles);
//! ```
//!
//! (`kernels/ada/common/prefill_dyn_smem.cuh`). When the registry resolves
//! `<entry>` it looks for that symbol in the same module; if present, the value
//! becomes the function's floor, and every launch of that function is raised to
//! at least that many dynamic bytes. The size has one source, the kernel's own
//! `sizeof`, and a kernel that exports nothing launches exactly as before: on a
//! module set with no floors the launch path pays one atomic load.
//!
//! FAIL-CLOSED. A symbol that exists but cannot be read is an error at
//! resolution, never a silent zero: launching a dynamic-tile kernel with no
//! dynamic memory is an out-of-bounds shared-memory access, not a slowdown.

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::LaunchConfig;

use crate::error::{AvarokError, Result};
use crate::registry::{RawCudaFunc, cuda_error_text};

unsafe extern "C" {
    fn cuModuleGetGlobal_v2(
        dptr: *mut u64,
        bytes: *mut usize,
        hmod: *mut c_void,
        name: *const i8,
    ) -> i32;
    fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> i32;
}

/// `CUDA_ERROR_NOT_FOUND`: the module has no such symbol, which for a floor
/// symbol is the ordinary case (the kernel keeps its tiles static).
const CUDA_ERROR_NOT_FOUND: i32 = 500;

/// The suffix a kernel's floor symbol carries after its entry-point name.
/// SSOT shared with `AVAROK_DYN_SMEM_SYMBOL` in
/// `kernels/ada/common/prefill_dyn_smem.cuh`.
pub const DYN_SMEM_SUFFIX: &str = "_dyn_smem_bytes";

/// The floor symbol for one entry point.
pub fn floor_symbol(func_name: &str) -> String {
    format!("{func_name}{DYN_SMEM_SUFFIX}")
}

/// The launch size `cfg` must use given a function's floor.
///
/// Never lowers what the caller asked for: a dispatch site that already passes
/// a larger dynamic size (tiles plus scratch, say) keeps it.
pub fn raised(cfg: LaunchConfig, floor: Option<u32>) -> LaunchConfig {
    match floor {
        Some(f) if cfg.shared_mem_bytes < f => LaunchConfig {
            shared_mem_bytes: f,
            ..cfg
        },
        _ => cfg,
    }
}

/// Per-registry map of CUfunction -> dynamic shared-memory floor.
#[derive(Default)]
pub(crate) struct DynSmemFloors {
    /// Set once the first floor is recorded, so a module set that declares
    /// none never takes the lock on the launch path.
    any: AtomicBool,
    by_func: RwLock<HashMap<usize, u32>>,
}

impl DynSmemFloors {
    pub(crate) fn record(&self, func: RawCudaFunc, bytes: u32) {
        self.by_func
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(func.0 as usize, bytes);
        self.any.store(true, Ordering::Release);
    }

    pub(crate) fn floor(&self, func: RawCudaFunc) -> Option<u32> {
        if !self.any.load(Ordering::Acquire) {
            return None;
        }
        self.by_func
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(func.0 as usize))
            .copied()
    }

    /// `cfg` raised to `func`'s floor, if it has one.
    pub(crate) fn apply(&self, func: RawCudaFunc, cfg: LaunchConfig) -> LaunchConfig {
        raised(cfg, self.floor(func))
    }

    /// Look for `<func_name>_dyn_smem_bytes` in `raw_mod` and record it as
    /// `func`'s floor. Absent is `Ok` (no floor); present but unreadable, or
    /// not a 4-byte value, is an error.
    ///
    /// # Safety
    /// `raw_mod` must be a loaded CUmodule of the current context and `func`
    /// a function resolved from it.
    pub(crate) unsafe fn discover(
        &self,
        raw_mod: *mut c_void,
        module_name: &str,
        func_name: &str,
        func: RawCudaFunc,
    ) -> Result<()> {
        let sym = floor_symbol(func_name);
        let c_sym = CString::new(sym.as_str())
            .map_err(|e| AvarokError::ModuleLoad(format!("{module_name}::{sym}: CString: {e}")))?;
        let mut dptr: u64 = 0;
        let mut bytes: usize = 0;
        let status =
            unsafe { cuModuleGetGlobal_v2(&mut dptr, &mut bytes, raw_mod, c_sym.as_ptr().cast()) };
        if status == CUDA_ERROR_NOT_FOUND {
            return Ok(());
        }
        if status != 0 {
            return Err(AvarokError::ModuleLoad(format!(
                "{module_name}::{sym}: cuModuleGetGlobal_v2 failed: {}",
                cuda_error_text(status)
            )));
        }
        if bytes != std::mem::size_of::<u32>() {
            return Err(AvarokError::ModuleLoad(format!(
                "{module_name}::{sym} is {bytes} bytes; a launch-size floor is one u32"
            )));
        }
        let mut value: u32 = 0;
        let status = unsafe { cuMemcpyDtoH_v2((&mut value as *mut u32).cast(), dptr, bytes) };
        if status != 0 {
            return Err(AvarokError::ModuleLoad(format!(
                "{module_name}::{sym}: reading the launch-size floor failed: {}",
                cuda_error_text(status)
            )));
        }
        self.record(func, value);
        Ok(())
    }
}

#[cfg(test)]
#[path = "registry_dyn_smem_tests.rs"]
mod tests;
