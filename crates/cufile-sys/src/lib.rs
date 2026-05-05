// SPDX-License-Identifier: AGPL-3.0-only

//! Minimal raw FFI for NVIDIA's libcufile (GPUDirect Storage / GDS).
//!
//! Loaded via dlopen so a binary can probe whether GDS is available
//! without failing to launch when `libcufile.so` is absent. Only the
//! symbols Atlas actually uses for the high-speed-swap path are bound
//! here; extend as needed.
//!
//! GDS is not currently supported on GB10 hardware (see
//! `docs/adr/0008-nvme-high-speed-swap.md`) — this crate is dormant on
//! that target and will become live when a GDS-capable platform lands.

#![deny(warnings)]
#![deny(clippy::all)]
#![allow(non_camel_case_types, non_snake_case)]

use libloading::{Library, Symbol};
use std::ffi::{c_int, c_void};
use std::os::raw::c_long;

pub const CUFILEOP_BASE_ERR: i32 = 5000;
pub const CU_FILE_SUCCESS: i32 = 0;
pub const CU_FILE_DRIVER_NOT_INITIALIZED: i32 = CUFILEOP_BASE_ERR + 1;
pub const CU_FILE_PLATFORM_NOT_SUPPORTED: i32 = CUFILEOP_BASE_ERR + 7;
pub const CU_FILE_IO_NOT_SUPPORTED: i32 = CUFILEOP_BASE_ERR + 8;
pub const CU_FILE_DEVICE_NOT_SUPPORTED: i32 = CUFILEOP_BASE_ERR + 9;

pub const CU_FILE_HANDLE_TYPE_OPAQUE_FD: c_int = 1;

pub type CUfileHandle_t = *mut c_void;
pub type CUresult = c_int;
pub type CUfileOpError = c_int;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CUfileError_t {
    pub err: CUfileOpError,
    pub cu_err: CUresult,
}

#[repr(C)]
pub struct CUfileDescrHandle {
    pub fd: c_int,
    _pad: [u8; 8 - core::mem::size_of::<c_int>()],
}

impl CUfileDescrHandle {
    pub fn from_fd(fd: c_int) -> Self {
        Self {
            fd,
            _pad: [0; 8 - core::mem::size_of::<c_int>()],
        }
    }
}

#[repr(C)]
pub struct CUfileDescr_t {
    pub type_: c_int,
    pub handle: CUfileDescrHandle,
    pub fs_ops: *const c_void,
}

pub type FnDriverOpen = unsafe extern "C" fn() -> CUfileError_t;
pub type FnDriverClose = unsafe extern "C" fn() -> CUfileError_t;
pub type FnHandleRegister =
    unsafe extern "C" fn(*mut CUfileHandle_t, *mut CUfileDescr_t) -> CUfileError_t;
pub type FnHandleDeregister = unsafe extern "C" fn(CUfileHandle_t);
pub type FnBufRegister = unsafe extern "C" fn(*const c_void, libc::size_t, c_int) -> CUfileError_t;
pub type FnBufDeregister = unsafe extern "C" fn(*const c_void) -> CUfileError_t;
pub type FnRead = unsafe extern "C" fn(
    CUfileHandle_t,
    *mut c_void,
    libc::size_t,
    c_long,
    c_long,
) -> libc::ssize_t;
pub type FnWrite = unsafe extern "C" fn(
    CUfileHandle_t,
    *const c_void,
    libc::size_t,
    c_long,
    c_long,
) -> libc::ssize_t;
pub type FnGetVersion = unsafe extern "C" fn(*mut c_int) -> CUfileError_t;

pub struct CuFile {
    _lib: Library,
    pub driver_open: FnDriverOpen,
    pub driver_close: FnDriverClose,
    pub handle_register: FnHandleRegister,
    pub handle_deregister: FnHandleDeregister,
    pub buf_register: FnBufRegister,
    pub buf_deregister: FnBufDeregister,
    pub read: FnRead,
    pub write: FnWrite,
    pub get_version: FnGetVersion,
}

const SEARCH_PATHS: &[&str] = &[
    "libcufile.so.0",
    "libcufile.so",
    "/usr/local/cuda/targets/sbsa-linux/lib/libcufile.so.0",
    "/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so.0",
    "/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcufile.so.0",
];

impl CuFile {
    pub fn load() -> Result<Self, String> {
        let mut last_err = String::new();
        for path in SEARCH_PATHS {
            match unsafe { Library::new(path) } {
                Ok(lib) => return Self::resolve(lib).map_err(|e| format!("{path}: {e}")),
                Err(e) => last_err = format!("{path}: {e}"),
            }
        }
        Err(format!("libcufile not found ({last_err})"))
    }

    fn resolve(lib: Library) -> Result<Self, String> {
        unsafe fn sym<'a, T: Copy + 'a>(lib: &'a Library, name: &[u8]) -> Result<T, String> {
            unsafe {
                let s: Symbol<'a, T> = lib
                    .get(name)
                    .map_err(|e| format!("symbol {}: {e}", String::from_utf8_lossy(name)))?;
                Ok(*s)
            }
        }
        unsafe {
            let driver_open = sym::<FnDriverOpen>(&lib, b"cuFileDriverOpen\0")?;
            let driver_close = sym::<FnDriverClose>(&lib, b"cuFileDriverClose\0")?;
            let handle_register = sym::<FnHandleRegister>(&lib, b"cuFileHandleRegister\0")?;
            let handle_deregister = sym::<FnHandleDeregister>(&lib, b"cuFileHandleDeregister\0")?;
            let buf_register = sym::<FnBufRegister>(&lib, b"cuFileBufRegister\0")?;
            let buf_deregister = sym::<FnBufDeregister>(&lib, b"cuFileBufDeregister\0")?;
            let read = sym::<FnRead>(&lib, b"cuFileRead\0")?;
            let write = sym::<FnWrite>(&lib, b"cuFileWrite\0")?;
            let get_version = sym::<FnGetVersion>(&lib, b"cuFileGetVersion\0")?;
            Ok(Self {
                _lib: lib,
                driver_open,
                driver_close,
                handle_register,
                handle_deregister,
                buf_register,
                buf_deregister,
                read,
                write,
                get_version,
            })
        }
    }
}

pub fn err_to_str(err: CUfileOpError) -> &'static str {
    match err {
        0 => "success",
        x if x == CU_FILE_DRIVER_NOT_INITIALIZED => "CU_FILE_DRIVER_NOT_INITIALIZED",
        x if x == CU_FILE_PLATFORM_NOT_SUPPORTED => "CU_FILE_PLATFORM_NOT_SUPPORTED",
        x if x == CU_FILE_IO_NOT_SUPPORTED => "CU_FILE_IO_NOT_SUPPORTED",
        x if x == CU_FILE_DEVICE_NOT_SUPPORTED => "CU_FILE_DEVICE_NOT_SUPPORTED",
        _ => "CU_FILE_OTHER_ERROR",
    }
}

pub fn nvidia_fs_loaded() -> bool {
    let modules = match std::fs::read_to_string("/proc/modules") {
        Ok(s) => s,
        Err(_) => return false,
    };
    modules
        .lines()
        .any(|l| l.starts_with("nvidia_fs ") || l.starts_with("nvidia-fs "))
}
