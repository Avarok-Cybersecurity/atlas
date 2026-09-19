// SPDX-License-Identifier: AGPL-3.0-only

//! What a CUresult means.
//!
//! The driver reports every failure as a bare `i32`. This module is the one
//! place that turns one into text, and the one place that decides which codes
//! mean "the context is already gone, nothing to do".

// The two driver entry points that describe a status code. The rest of the
// raw CUDA API the registry calls stays with the calls themselves.
unsafe extern "C" {
    fn cuGetErrorName(error: i32, pStr: *mut *const i8) -> i32;
    fn cuGetErrorString(error: i32, pStr: *mut *const i8) -> i32;
}

/// Resolve a CUresult status code into `"<NAME>: <description>"` via
/// cuGetErrorName + cuGetErrorString. Returns "CUDA_UNKNOWN" / "(no message)"
/// if the driver doesn't recognize the code.
pub fn cuda_error_text(status: i32) -> String {
    use std::ffi::CStr;
    let mut name_ptr: *const i8 = std::ptr::null();
    let mut msg_ptr: *const i8 = std::ptr::null();
    let name = unsafe {
        if cuGetErrorName(status, &mut name_ptr) == 0 && !name_ptr.is_null() {
            CStr::from_ptr(name_ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        } else {
            "CUDA_UNKNOWN".to_string()
        }
    };
    let msg = unsafe {
        if cuGetErrorString(status, &mut msg_ptr) == 0 && !msg_ptr.is_null() {
            CStr::from_ptr(msg_ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        } else {
            "(no message)".to_string()
        }
    };
    format!("{name} ({status}): {msg}")
}

/// `CUDA_ERROR_DEINITIALIZED`. The driver tears the primary context down in its
/// own `atexit` handler, which can run before our `Drop` impls do. Every
/// module unload and every host free then reports this code.
///
/// It is **not a failure**: a module cannot leak out of a context that no
/// longer exists, and the memory it occupied went with it. Reporting 158 of
/// them at exit is pure noise that buries anything real.
pub const CUDA_ERROR_DEINITIALIZED: i32 = 4;

/// Whether a CUresult means "the context is already gone, nothing to do".
///
/// Also covers `CUDA_ERROR_INVALID_CONTEXT` (201) and
/// `CUDA_ERROR_CONTEXT_IS_DESTROYED` (709), which arrive by the same route
/// depending on how far the driver got before we ran.
pub fn is_teardown_noop(status: i32) -> bool {
    matches!(status, CUDA_ERROR_DEINITIALIZED | 201 | 709)
}
