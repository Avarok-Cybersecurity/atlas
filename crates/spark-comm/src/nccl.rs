// SPDX-License-Identifier: AGPL-3.0-only

//! Raw NCCL FFI bindings (minimal surface for EP all-reduce + broadcast).
//!
//! Only the functions Atlas actually calls are bound here — no attempt
//! at complete coverage. Type sizes match NCCL 2.28+ on aarch64
//! (symmetric memory `ncclMemAlloc`/`ncclMemFree` require NCCL ≥ 2.28).
//!
//! ## Safety
//!
//! Every `unsafe` block wraps a single NCCL FFI call. Invariants:
//! - The `NcclComm`/`NcclUniqueId` arguments come from prior `Self`
//!   constructors that called the matching NCCL init function.
//! - GPU buffers are valid `DevicePtr`s alive on the device that owns
//!   the comm.
//! - Counts × dtype-size match the buffer byte count.
//! - The `extern "C"` declarations match the NCCL header ABI for the
//!   linked library version.

use std::ffi::c_void;

/// Opaque NCCL communicator handle.
pub type NcclComm = *mut c_void;

/// NCCL unique ID for bootstrapping (128 bytes, passed by value in C ABI).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct NcclUniqueId {
    pub internal: [u8; 128],
}

/// NCCL config for ncclCommInitRankConfig.
/// Set blocking=0 for non-blocking mode (ncclGroupEnd returns immediately).
#[repr(C)]
pub struct NcclConfig {
    /// Size of this struct (for versioning).
    pub size: usize,
    /// Magic number (0x4d43434e = "NCCM" in LE for NCCL 2.27+).
    pub magic: u32,
    /// Version (NCCL_VERSION_CODE).
    pub version: u32,
    /// 1 = blocking (default), 0 = non-blocking.
    pub blocking: i32,
    /// CGA cluster size (0 = default).
    pub cga_cluster_size: i32,
    /// Min CTAs for launch (0 = default).
    pub min_ctas: i32,
    /// Max CTAs for launch (0 = default).
    pub max_ctas: i32,
    /// Network name (null-terminated, or all zeros for default).
    pub net_name: [u8; 8],
    /// Split share (0 = default).
    pub split_share: i32,
}

impl NcclConfig {
    /// Create a default config with non-blocking mode enabled.
    /// Must match NCCL_CONFIG_INITIALIZER from nccl.h:
    ///   { sizeof(ncclConfig_t), 0x4e43434c, NCCL_VERSION(MAJOR,MINOR,PATCH),
    ///     NCCL_CONFIG_UNDEF_INT, ... }
    pub fn non_blocking() -> Self {
        // NCCL_CONFIG_UNDEF_INT = -1 means "use default"
        Self {
            size: std::mem::size_of::<Self>(),
            magic: 0x4e43434c,    // "NCCL" in LE (not "NCCM")
            version: 22907,       // NCCL 2.29.7 (major*10000 + minor*100 + patch)
            blocking: 0,          // NON-BLOCKING
            cga_cluster_size: -1, // NCCL_CONFIG_UNDEF_INT
            min_ctas: -1,
            max_ctas: -1,
            net_name: [0; 8],
            split_share: -1,
        }
    }
}

/// NCCL result code.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NcclResult {
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    RemoteError = 6,
    InProgress = 7,
}

/// NCCL data types (matches nccl.h enum values).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
pub enum NcclDataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
}

/// NCCL reduction operations.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
#[allow(dead_code)]
pub enum NcclRedOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

#[link(name = "nccl")]
unsafe extern "C" {
    pub fn ncclGetUniqueId(id: *mut NcclUniqueId) -> NcclResult;

    pub fn ncclCommInitRank(
        comm: *mut NcclComm,
        nranks: i32,
        id: NcclUniqueId,
        rank: i32,
    ) -> NcclResult;

    /// Non-blocking variant: set config.blocking=0 so ncclGroupEnd returns immediately.
    /// Poll ncclCommGetAsyncError to check completion and detect hangs with timeout.
    pub fn ncclCommInitRankConfig(
        comm: *mut NcclComm,
        nranks: i32,
        id: NcclUniqueId,
        rank: i32,
        config: *const NcclConfig,
    ) -> NcclResult;

    pub fn ncclAllReduce(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: u64, // cudaStream_t
    ) -> NcclResult;

    pub fn ncclBroadcast(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        root: i32,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    pub fn ncclCommDestroy(comm: NcclComm) -> NcclResult;

    pub fn ncclGetErrorString(result: NcclResult) -> *const std::ffi::c_char;

    // Buffer registration (pre-registers with IB HCA, avoids per-call ibv_reg_mr).
    pub fn ncclCommRegister(
        comm: NcclComm,
        buff: *mut c_void,
        size: usize,
        handle: *mut *mut c_void,
    ) -> NcclResult;

    pub fn ncclCommDeregister(comm: NcclComm, handle: *mut c_void) -> NcclResult;

    // NCCL 2.28+ symmetric memory window APIs.
    //
    // Buffers allocated via `ncclMemAlloc` participate in symmetric memory
    // windows across the communicator, enabling:
    //   1. Copy-engine offload on NVLink-connected ranks (frees SMs for
    //      compute during AllReduce/AllGather).
    //   2. The device-side communication API (kernels can issue collectives
    //      directly without host round-trip), which is the substrate for
    //      fused AllReduce+RMSNorm+Residual kernels (TokenWeave-style).
    //
    // For Atlas's 2-rank Spark over RoCE, the copy-engine path doesn't
    // apply (RoCE is not NVLink), but the symmetric-memory windows are
    // still needed to compose with future device-API fusions and reduce
    // NCCL setup overhead via pre-registered handles.
    //
    // Returns `InvalidArgument` if the linked NCCL is < 2.28.
    pub fn ncclMemAlloc(ptr: *mut *mut c_void, size: usize) -> NcclResult;

    pub fn ncclMemFree(ptr: *mut c_void) -> NcclResult;

    // Point-to-point (for custom 2-rank all-reduce).
    pub fn ncclSend(
        sendbuf: *const c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    pub fn ncclRecv(
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        peer: i32,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    // Collective: all-gather (each rank sends `count`, recv gets `world_size * count`).
    pub fn ncclAllGather(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        sendcount: usize,
        datatype: NcclDataType,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    // Collective: reduce-scatter (send has `world_size * count`, each rank recvs `count`).
    pub fn ncclReduceScatter(
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        recvcount: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
        comm: NcclComm,
        stream: u64,
    ) -> NcclResult;

    // Group API (batch multiple send/recv into one launch).
    pub fn ncclGroupStart() -> NcclResult;
    pub fn ncclGroupEnd() -> NcclResult;

    // Health check: retrieve asynchronous errors from the communicator.
    pub fn ncclCommGetAsyncError(comm: NcclComm, async_error: *mut NcclResult) -> NcclResult;

    // Abort: destroy a communicator that is in a failed state.
    // Unlike ncclCommDestroy, this does not block and cleans up immediately.
    pub fn ncclCommAbort(comm: NcclComm) -> NcclResult;
}

// CUDA driver API for inter-stream synchronization (async all-reduce).
// spark-comm already links libcuda transitively via NCCL.
#[link(name = "cuda")]
unsafe extern "C" {
    fn cuStreamCreate(phStream: *mut u64, flags: u32) -> i32;
    fn cuEventCreate(phEvent: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(hEvent: u64, hStream: u64) -> i32;
    fn cuStreamWaitEvent(hStream: u64, hEvent: u64, flags: u32) -> i32;
    fn cuEventDestroy_v2(hEvent: u64) -> i32;
    fn cuStreamDestroy_v2(hStream: u64) -> i32;
    fn cuStreamSynchronize(hStream: u64) -> i32;
}

pub fn create_stream() -> anyhow::Result<u64> {
    let mut stream: u64 = 0;
    let status = unsafe { cuStreamCreate(&mut stream, 1) }; // CU_STREAM_NON_BLOCKING
    if status != 0 {
        anyhow::bail!("cuStreamCreate failed: status {status}");
    }
    Ok(stream)
}

pub fn create_event() -> anyhow::Result<u64> {
    let mut event: u64 = 0;
    let status = unsafe { cuEventCreate(&mut event, 0x02) }; // CU_EVENT_DISABLE_TIMING
    if status != 0 {
        anyhow::bail!("cuEventCreate failed: status {status}");
    }
    Ok(event)
}

pub fn record_event(event: u64, stream: u64) -> anyhow::Result<()> {
    let status = unsafe { cuEventRecord(event, stream) };
    if status != 0 {
        anyhow::bail!("cuEventRecord failed: status {status}");
    }
    Ok(())
}

pub fn stream_wait_event(stream: u64, event: u64) -> anyhow::Result<()> {
    let status = unsafe { cuStreamWaitEvent(stream, event, 0) };
    if status != 0 {
        anyhow::bail!("cuStreamWaitEvent failed: status {status}");
    }
    Ok(())
}

pub fn destroy_event(event: u64) {
    if event != 0 {
        unsafe { cuEventDestroy_v2(event) };
    }
}

pub fn destroy_stream(stream: u64) {
    if stream != 0 {
        unsafe { cuStreamDestroy_v2(stream) };
    }
}

pub fn sync_stream(stream: u64) -> anyhow::Result<()> {
    let status = unsafe { cuStreamSynchronize(stream) };
    if status != 0 {
        anyhow::bail!("cuStreamSynchronize failed: status {status}");
    }
    Ok(())
}

/// Allocate GPU memory backed by a symmetric memory window across the
/// communicator. NCCL 2.28+ only — older NCCL returns `InvalidArgument`.
///
/// Buffers from `ncclMemAlloc` enable copy-engine collectives over NVLink
/// and the device-side communication API. On Atlas's 2-rank Spark over
/// RoCE, the copy-engine offload is unavailable (RoCE != NVLink), but the
/// symmetric windows are required to compose with device-API fused kernels
/// (TokenWeave-style AR+RMSNorm).
///
/// # Safety
/// The returned pointer must be freed via [`nccl_mem_free`]. Passing the
/// pointer to non-NCCL allocators (e.g. `cudaFree`) is undefined behavior.
pub unsafe fn nccl_mem_alloc(size: usize) -> anyhow::Result<*mut c_void> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    let result = unsafe { ncclMemAlloc(&mut ptr, size) };
    check_nccl(result, "ncclMemAlloc")?;
    Ok(ptr)
}

/// Free a buffer previously returned by `nccl_mem_alloc`.
///
/// # Safety
/// `ptr` must have been returned by [`nccl_mem_alloc`] and not yet freed.
pub unsafe fn nccl_mem_free(ptr: *mut c_void) -> anyhow::Result<()> {
    let result = unsafe { ncclMemFree(ptr) };
    check_nccl(result, "ncclMemFree")
}

/// Convert NCCL result to anyhow::Result.
pub fn check_nccl(result: NcclResult, context: &str) -> anyhow::Result<()> {
    if result == NcclResult::Success {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = ncclGetErrorString(result);
            if ptr.is_null() {
                format!("{result:?}")
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        anyhow::bail!("NCCL error in {context}: {msg} ({result:?})")
    }
}
