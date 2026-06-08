// SPDX-License-Identifier: AGPL-3.0-only
//
// Minimal DLPack tensor types — W7 compatibility shim.
//
// The vendored C++ `xgrammar-rs` re-exported the real `dlpack/dlpack.h`
// structs because its `fill_next_token_bitmask` crossed the FFI
// boundary as a `DLTensor*`. The pure-Rust port has no FFI boundary and
// works on `&mut [i32]` directly, but Atlas's `grammar/state.rs` still
// builds a `DLTensor` describing its bitmask buffer. We provide a
// layout-faithful, pure-Rust re-implementation so that Atlas's
// construction site compiles unchanged; the façade's
// `GrammarMatcher::fill_next_token_bitmask` reads the `data`/`shape`
// fields back out to recover the `&mut [i32]` slice.

use std::ffi::c_void;

/// DLPack device type enum (`DLDeviceType`).
///
/// Only the variants Atlas references are defined; discriminants match
/// the dlpack ABI so a future real-FFI consumer stays compatible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
#[allow(non_camel_case_types)]
pub enum DLDeviceType {
    /// CPU device.
    kDLCPU = 1,
    /// CUDA GPU device.
    kDLCUDA = 2,
}

/// DLPack data type code enum (`DLDataTypeCode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum DLDataTypeCode {
    /// Signed integer.
    kDLInt = 0,
    /// Unsigned integer.
    kDLUInt = 1,
    /// IEEE floating point.
    kDLFloat = 2,
}

/// DLPack device descriptor (`DLDevice`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDevice {
    /// The device type.
    pub device_type: DLDeviceType,
    /// The device index.
    pub device_id: i32,
}

/// DLPack data type descriptor (`DLDataType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DLDataType {
    /// The type code (see [`DLDataTypeCode`]).
    pub code: u8,
    /// Bits per element.
    pub bits: u8,
    /// Number of lanes (vector width).
    pub lanes: u16,
}

/// DLPack tensor view (`DLTensor`) — does not own memory.
///
/// This mirrors the field set Atlas's `grammar/state.rs` populates. The
/// W7 façade only ever reads `data`, `shape` and `ndim` back out, so
/// the remaining fields are carried purely for source compatibility.
#[derive(Debug, Clone, Copy)]
pub struct DLTensor {
    /// Pointer to the underlying buffer.
    pub data: *mut c_void,
    /// The device the buffer lives on.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// The element data type.
    pub dtype: DLDataType,
    /// Pointer to the shape array (`ndim` entries).
    pub shape: *mut i64,
    /// Pointer to the strides array, or null for row-major.
    pub strides: *mut i64,
    /// Byte offset from `data` to the first element.
    pub byte_offset: u64,
}
