// SPDX-License-Identifier: AGPL-3.0-only

//! Diagnostic-only NVTX range instrumentation (Stage 2b kernel-attribution
//! profiling). Gated behind `ATLAS_NVTX_DIAG=1` (PCND, default-off) so it has
//! zero cost/dependency in normal builds and runs.
//!
//! `libnvToolsExt` is dlopen'd lazily (via `libloading`, not linked at build
//! time) so a missing library on non-profiling machines just disables
//! ranges instead of failing the build or the run. Call sites push a label
//! immediately around the one GPU-launch/API call they want attributed;
//! nsys's NVTX+CUDA-API correlation (via `--trace=cuda,nvtx,osrt` and the
//! `nvtx_kern_sum` report) then attributes the *actually launched* kernel
//! time to that label even though the kernel executes asynchronously after
//! the host-side push/pop window closes.

use libloading::{Library, Symbol};
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::sync::OnceLock;

type PushFn = unsafe extern "C" fn(*const c_char) -> c_int;
type PopFn = unsafe extern "C" fn() -> c_int;

struct NvtxFns {
    push: PushFn,
    pop: PopFn,
    // Keep the library mapped for the process lifetime; never unloaded.
    _lib: Library,
}

// SAFETY: raw fn pointers into a dlopen'd shared library that is never
// unloaded (owned for 'static via the OnceLock); the underlying NVTX ABI
// is thread-safe (it's the same library the C++ NVTX3 header calls into).
unsafe impl Send for NvtxFns {}
unsafe impl Sync for NvtxFns {}

fn enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("ATLAS_NVTX_DIAG").as_deref() == Ok("1"))
}

fn nvtx() -> Option<&'static NvtxFns> {
    static LIB: OnceLock<Option<NvtxFns>> = OnceLock::new();
    LIB.get_or_init(|| {
        if !enabled() {
            return None;
        }
        // Resolved via the dynamic linker search path (LD_LIBRARY_PATH) —
        // the profiling harness is responsible for pointing that at a
        // directory containing one of these sonames.
        for candidate in ["libnvToolsExt.so.1", "libnvToolsExt.so"] {
            let lib = unsafe { Library::new(candidate) };
            let Ok(lib) = lib else { continue };
            let push: Symbol<PushFn> = match unsafe { lib.get(b"nvtxRangePushA\0") } {
                Ok(s) => s,
                Err(_) => continue,
            };
            let pop: Symbol<PopFn> = match unsafe { lib.get(b"nvtxRangePop\0") } {
                Ok(s) => s,
                Err(_) => continue,
            };
            let push = *push;
            let pop = *pop;
            tracing::info!("ATLAS_NVTX_DIAG: loaded {candidate}");
            return Some(NvtxFns {
                push,
                pop,
                _lib: lib,
            });
        }
        tracing::warn!(
            "ATLAS_NVTX_DIAG=1 set but no libnvToolsExt found — NVTX ranges disabled"
        );
        None
    })
    .as_ref()
}

/// RAII NVTX range: push on construction, pop on drop. No-op unless
/// `ATLAS_NVTX_DIAG=1` and libnvToolsExt was found.
pub struct NvtxRange {
    active: bool,
}

impl NvtxRange {
    #[inline]
    pub fn new(label: &str) -> Self {
        let Some(n) = nvtx() else {
            return Self { active: false };
        };
        if let Ok(c) = CString::new(label) {
            unsafe { (n.push)(c.as_ptr()) };
        }
        Self { active: true }
    }
}

impl Drop for NvtxRange {
    #[inline]
    fn drop(&mut self) {
        if self.active {
            if let Some(n) = nvtx() {
                unsafe { (n.pop)() };
            }
        }
    }
}
