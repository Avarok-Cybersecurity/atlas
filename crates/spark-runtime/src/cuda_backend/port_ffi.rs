// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: resolve, once per process, the driver entry points the port
//! sprint's backend extensions call (graph exec update and node queries, graph
//! upload, batched async copies, event polling and timing, host-memory
//! registration, stream destruction), through `cuGetProcAddress` rather than as
//! link-time imports.
//!
//! Why at run time: a link-time import that the loaded `libcuda` lacks breaks the
//! whole binary at load (full RELRO binds every import up front), and two of the
//! drivers Atlas links against are thinner than NVIDIA's: SCALE's `libcuda` for
//! gfx1151 and the CUDA-13 driver stub CI links against (see the
//! `cuStreamIsCapturing` note in `cuda_backend.rs`). A missing entry point here
//! turns one feature off instead: each consumer asks `supports_*` first and every
//! `_cu` body has a fallback. Under `avarok_scale` nothing is resolved at all and
//! every extension reports unsupported.
//!
//! ABI versions are pinned per symbol: `cuGetProcAddress` returns the variant of
//! the requested CUDA version, so `cuGraphExecUpdate` at 12000 is the `_v2`
//! (result-info) form, `cuGraphKernelNodeGetParams` at 12000 the `_v2` struct, and
//! `cuMemcpyBatchAsync` at 13000 the 13.x form without `failIdx` (at 12080, for a
//! 12.8 or 12.9 driver, the form with it). The batch symbol is requested only when
//! `cuDriverGetVersion` says the driver has that version, because a driver asked
//! for a newer version than it implements answers with its own newest variant,
//! whose signature would not match.

use std::ffi::c_void;
use std::sync::OnceLock;

/// `cuGraphExecUpdate` (12.0 ABI).
pub(super) type GraphExecUpdateFn = unsafe extern "C" fn(u64, u64, *mut ExecUpdateInfo) -> i32;
/// `cuGraphGetNodes(graph, nodes, num_nodes)`.
pub(super) type GraphGetNodesFn = unsafe extern "C" fn(u64, *mut u64, *mut usize) -> i32;
/// `cuGraphNodeGetType(node, type)`.
pub(super) type GraphNodeGetTypeFn = unsafe extern "C" fn(u64, *mut u32) -> i32;
/// `cuGraphKernelNodeGetParams` (12.0 ABI, `CUDA_KERNEL_NODE_PARAMS_v2`).
pub(super) type KernelNodeGetParamsFn = unsafe extern "C" fn(u64, *mut KernelNodeParams) -> i32;
/// `cuGraphUpload(exec, stream)`.
pub(super) type GraphUploadFn = unsafe extern "C" fn(u64, u64) -> i32;
/// `cuMemcpyBatchAsync`, CUDA 13.x ABI: `(dsts, srcs, sizes, count, attrs, attrsIdxs,
/// numAttrs, stream)`.
pub(super) type MemcpyBatch13Fn = unsafe extern "C" fn(
    *mut u64,
    *mut u64,
    *mut usize,
    usize,
    *mut MemcpyAttributes,
    *mut usize,
    usize,
    u64,
) -> i32;
/// `cuMemcpyBatchAsync`, CUDA 12.8/12.9 ABI: as above with `failIdx` before the stream.
pub(super) type MemcpyBatch128Fn = unsafe extern "C" fn(
    *mut u64,
    *mut u64,
    *mut usize,
    usize,
    *mut MemcpyAttributes,
    *mut usize,
    usize,
    *mut usize,
    u64,
) -> i32;
/// `cuEventQuery(event)`.
pub(super) type EventQueryFn = unsafe extern "C" fn(u64) -> i32;
/// `cuEventElapsedTime(ms, start, end)` (the 12.0 form; `_v2` has the same signature).
pub(super) type EventElapsedFn = unsafe extern "C" fn(*mut f32, u64, u64) -> i32;
/// `cuMemHostRegister` (`_v2`): `(ptr, bytes, flags)`.
pub(super) type HostRegisterFn = unsafe extern "C" fn(*mut c_void, usize, u32) -> i32;
/// `cuMemHostUnregister(ptr)`.
pub(super) type HostUnregisterFn = unsafe extern "C" fn(*mut c_void) -> i32;
/// `cuStreamDestroy` (`_v2`).
pub(super) type StreamDestroyFn = unsafe extern "C" fn(u64) -> i32;

/// `CUgraphExecUpdateResultInfo`. Filled by the driver; only `result` is read (for the
/// log line of a refused update).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct ExecUpdateInfo {
    pub result: u32,
    _error_node: u64,
    _error_from_node: u64,
}

impl ExecUpdateInfo {
    pub(super) const EMPTY: Self = Self {
        result: 0,
        _error_node: 0,
        _error_from_node: 0,
    };
}

/// `CUDA_KERNEL_NODE_PARAMS_v2`. Filled by the driver; the signature reads the function
/// (`func`, or `kern` for a node launched from a library kernel). The other fields exist
/// for the layout only.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct KernelNodeParams {
    pub func: u64,
    _grid: [u32; 3],
    _block: [u32; 3],
    _shared_mem_bytes: u32,
    _kernel_params: *mut *mut c_void,
    _extra: *mut *mut c_void,
    pub kern: u64,
    _ctx: u64,
}

impl KernelNodeParams {
    pub(super) const EMPTY: Self = Self {
        func: 0,
        _grid: [0; 3],
        _block: [0; 3],
        _shared_mem_bytes: 0,
        _kernel_params: std::ptr::null_mut(),
        _extra: std::ptr::null_mut(),
        kern: 0,
        _ctx: 0,
    };
}

/// `CUmemLocation`: `{ CUmemLocationType type; int id; }`. Zero (INVALID, 0) is what
/// LMCache and vLLM pass: the hints apply only to managed or system-pageable operands,
/// and the tier's operands are device memory and page-locked host memory.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct MemLocation {
    _kind: u32,
    _id: i32,
}

/// `CUmemcpyAttributes` (`_v1`). Written by Rust, read only by the driver.
#[repr(C)]
#[derive(Clone, Copy, Default)]
#[allow(dead_code)]
pub(super) struct MemcpyAttributes {
    /// `CUmemcpySrcAccessOrder`: [`SRC_ACCESS_ORDER_STREAM`] or [`SRC_ACCESS_ORDER_ANY`].
    pub src_access_order: u32,
    pub(super) _src_loc_hint: MemLocation,
    pub(super) _dst_loc_hint: MemLocation,
    pub(super) _flags: u32,
}

/// `CU_MEMCPY_SRC_ACCESS_ORDER_STREAM`: sources are read in stream order.
pub(super) const SRC_ACCESS_ORDER_STREAM: u32 = 1;
/// `CU_MEMCPY_SRC_ACCESS_ORDER_ANY`: sources may be read out of stream order, also after
/// the call returns (no stream writes them concurrently).
pub(super) const SRC_ACCESS_ORDER_ANY: u32 = 3;

// The layouts the driver writes into, checked at compile time on the 64-bit hosts Atlas
// builds for (CUDA's own sizes for these structs on LP64).
const _: () = assert!(std::mem::size_of::<KernelNodeParams>() == 72);
const _: () = assert!(std::mem::size_of::<ExecUpdateInfo>() == 24);
const _: () = assert!(std::mem::size_of::<MemcpyAttributes>() == 24);

/// The two ABIs of the batch entry point. (Never constructed under SCALE, which resolves
/// nothing.)
#[derive(Clone, Copy)]
#[cfg_attr(avarok_scale, allow(dead_code))]
pub(super) enum MemcpyBatch {
    V13(MemcpyBatch13Fn),
    V128(MemcpyBatch128Fn),
}

/// Entry points resolved for this process. `None` = not available from this driver.
pub(super) struct PortDriver {
    /// `cuDriverGetVersion` (for example 13030), or 0 when unknown.
    pub driver_version: i32,
    pub graph_exec_update: Option<GraphExecUpdateFn>,
    pub graph_get_nodes: Option<GraphGetNodesFn>,
    pub graph_node_get_type: Option<GraphNodeGetTypeFn>,
    pub kernel_node_get_params: Option<KernelNodeGetParamsFn>,
    pub graph_upload: Option<GraphUploadFn>,
    pub memcpy_batch: Option<MemcpyBatch>,
    pub event_query: Option<EventQueryFn>,
    pub event_elapsed: Option<EventElapsedFn>,
    pub host_register: Option<HostRegisterFn>,
    pub host_unregister: Option<HostUnregisterFn>,
    pub stream_destroy: Option<StreamDestroyFn>,
}

impl PortDriver {
    const NONE: Self = Self {
        driver_version: 0,
        graph_exec_update: None,
        graph_get_nodes: None,
        graph_node_get_type: None,
        kernel_node_get_params: None,
        graph_upload: None,
        memcpy_batch: None,
        event_query: None,
        event_elapsed: None,
        host_register: None,
        host_unregister: None,
        stream_destroy: None,
    };

    /// Every entry point exec update and the topology signature need.
    pub(super) fn graph_update_ready(&self) -> bool {
        self.graph_exec_update.is_some()
            && self.graph_get_nodes.is_some()
            && self.graph_node_get_type.is_some()
            && self.kernel_node_get_params.is_some()
    }
}

/// The process's resolved entry points. Function pointers from `libcuda` are process-wide
/// (not per context), so one table serves every backend.
pub(super) fn driver() -> &'static PortDriver {
    static DRIVER: OnceLock<PortDriver> = OnceLock::new();
    DRIVER.get_or_init(resolve)
}

#[cfg(avarok_scale)]
fn resolve() -> PortDriver {
    PortDriver::NONE
}

#[cfg(not(avarok_scale))]
unsafe extern "C" {
    fn cuDriverGetVersion(driver_version: *mut i32) -> i32;
    fn cuGetProcAddress_v2(
        symbol: *const std::ffi::c_char,
        pfn: *mut *mut c_void,
        cuda_version: i32,
        flags: u64,
        symbol_status: *mut i32,
    ) -> i32;
}

#[cfg(not(avarok_scale))]
fn resolve() -> PortDriver {
    let mut version: i32 = 0;
    // SAFETY: plain out-parameter call; needs only an initialized driver, which every
    // backend method that reaches here already has.
    if unsafe { cuDriverGetVersion(&mut version) } != 0 || version < 12000 {
        return PortDriver::NONE;
    }
    let memcpy_batch = if version >= 13000 {
        sym::<MemcpyBatch13Fn>(c"cuMemcpyBatchAsync", 13000).map(MemcpyBatch::V13)
    } else if version >= 12080 {
        sym::<MemcpyBatch128Fn>(c"cuMemcpyBatchAsync", 12080).map(MemcpyBatch::V128)
    } else {
        None
    };
    PortDriver {
        driver_version: version,
        graph_exec_update: sym(c"cuGraphExecUpdate", 12000),
        graph_get_nodes: sym(c"cuGraphGetNodes", 12000),
        graph_node_get_type: sym(c"cuGraphNodeGetType", 12000),
        kernel_node_get_params: sym(c"cuGraphKernelNodeGetParams", 12000),
        graph_upload: sym(c"cuGraphUpload", 12000),
        memcpy_batch,
        event_query: sym(c"cuEventQuery", 12000),
        event_elapsed: sym(c"cuEventElapsedTime", 12000),
        host_register: sym(c"cuMemHostRegister", 12000),
        host_unregister: sym(c"cuMemHostUnregister", 12000),
        stream_destroy: sym(c"cuStreamDestroy", 12000),
    }
}

/// `cuGetProcAddress` for `name` at ABI `cuda_version`, as the function type `F`.
#[cfg(not(avarok_scale))]
fn sym<F: Copy>(name: &std::ffi::CStr, cuda_version: i32) -> Option<F> {
    /// `CU_GET_PROC_ADDRESS_LEGACY_STREAM`: the variants the link-time imports in
    /// `cuda_backend.rs` already use (no per-thread default stream).
    const LEGACY_STREAM: u64 = 1;
    /// `CU_GET_PROC_ADDRESS_SUCCESS`.
    const FOUND: i32 = 0;
    assert_eq!(
        std::mem::size_of::<F>(),
        std::mem::size_of::<*mut c_void>(),
        "sym::<F>: F must be a function pointer type"
    );
    let mut pfn: *mut c_void = std::ptr::null_mut();
    let mut status: i32 = -1;
    // SAFETY: `name` is NUL-terminated; both out-parameters are valid for writes.
    let rc = unsafe {
        cuGetProcAddress_v2(
            name.as_ptr(),
            &mut pfn,
            cuda_version,
            LEGACY_STREAM,
            &mut status,
        )
    };
    if rc != 0 || status != FOUND || pfn.is_null() {
        return None;
    }
    // SAFETY: `F` is the `unsafe extern "C" fn` type of `name` at ABI `cuda_version`, as
    // the type aliases above transcribe it from `cuda.h` 13.3; the size was checked above.
    Some(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&pfn) })
}
