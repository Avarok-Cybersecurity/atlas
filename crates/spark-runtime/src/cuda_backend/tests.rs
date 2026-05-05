// SPDX-License-Identifier: AGPL-3.0-only

//! cuda_backend unit tests. Pure CPU — no `cuInit`, no GPU touch — so
//! they run on every CI host.

use std::ffi::c_void;

use atlas_core::registry::RawCudaFunc;

use crate::gpu::{DevicePtr, KernelHandle};

#[test]
fn kernel_handle_roundtrip() {
    // Verify KernelHandle <-> RawCudaFunc pointer conversion is lossless.
    let fake_ptr = 0xDEAD_BEEF_CAFE_u64;
    let handle = KernelHandle(fake_ptr);
    let raw = RawCudaFunc(handle.0 as *mut c_void);
    let back = raw.0 as u64;
    assert_eq!(back, fake_ptr);
}

#[test]
fn null_free_is_noop() {
    // AtlasCudaBackend::free should handle null pointers gracefully.
    // Can't call without GPU, but verify the DevicePtr::is_null logic.
    assert!(DevicePtr::NULL.is_null());
    assert!(!DevicePtr(0x1000).is_null());
}
