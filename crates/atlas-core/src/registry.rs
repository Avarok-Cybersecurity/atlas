// SPDX-License-Identifier: AGPL-3.0-only

//! Global kernel registry — load PTX once, cache modules/functions/streams.
//!
//! Eliminates ~0.06-0.26ms overhead per kernel call from:
//! - CudaContext::new (driver init)
//! - CudaContext::load_module (PTX JIT compilation)
//! - CudaContext::new_stream (stream creation)
//! - cuModuleGetFunction (function lookup) — now cached after first call
//!
//! Usage:
//!   let reg = AtlasRegistry::get_or_init(ordinal, &[("gemm", PTX_SRC), ...])?;
//!   let func = reg.function("gemm", "dense_gemm_tc_bf16")?;
//!   unsafe { reg.stream.launch_builder(&func).arg(&ptr).launch(cfg)?; }
//!   reg.stream.synchronize()?;

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::Ptx;

pub use crate::cuda_host::{CudaHost, host, release};
use crate::error::{AtlasError, Result};

// Raw CUDA driver API. (`cuModuleLoadData`/`cuModuleUnload` left this list
// when the raw handles became views into the cudarc-loaded modules — the
// registry no longer loads or unloads anything through the raw API.)
unsafe extern "C" {
    fn cuModuleGetFunction(hfunc: *mut *mut c_void, hmod: *mut c_void, name: *const i8) -> i32;
    fn cuLaunchKernel(
        f: *mut c_void,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuFuncSetAttribute(hfunc: *mut c_void, attrib: i32, value: i32) -> i32;
    fn cuGetErrorName(error: i32, pStr: *mut *const i8) -> i32;
    fn cuGetErrorString(error: i32, pStr: *mut *const i8) -> i32;
    // Resolve a `__device__` symbol in a loaded CUmodule into a device pointer
    // + size in bytes. Used by drivers that need to read/write device globals
    // (e.g. InnerQ calibration state) without round-tripping through a kernel.
    fn cuModuleGetGlobal_v2(
        dptr: *mut u64,
        bytes: *mut usize,
        hmod: *mut c_void,
        name: *const i8,
    ) -> i32;
    fn cuMemcpyHtoDAsync_v2(dst: u64, src: *const c_void, bytes: usize, stream: u64) -> i32;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: u64, bytes: usize, stream: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
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

/// Wrapper for raw CUfunction handle (Send+Sync safe — handles are context-wide).
#[derive(Clone, Copy)]
pub struct RawCudaFunc(pub *mut c_void);
// SAFETY: CUfunction handles returned by `cuModuleGetFunction` remain valid
// for the lifetime of the owning CUcontext (the Atlas registry binds the
// process-wide context once at startup and never destroys it). The handle
// itself is opaque metadata — actual kernel launches go through cuLaunchKernel
// with caller-supplied stream synchronisation, so `Sync` does not imply
// concurrent execution, only concurrent reads of an immutable pointer.
unsafe impl Send for RawCudaFunc {}
unsafe impl Sync for RawCudaFunc {}

/// The PTX/CUBIN modules for **one** loaded model.
///
/// Model-scoped: the blob set comes from `atlas_kernels::ptx_for_model`, so it
/// changes with the checkpoint. Previously this was fused into a process
/// `OnceLock` singleton whose `get_or_init(ordinal, kernel_blobs)` silently
/// discarded the second caller's blobs — a swapped-in model would have run the
/// *previous* model's kernels with no error at all.
///
/// Obtain one with [`AtlasRegistry::load`] and propagate it (`Arc<AtlasRegistry>`);
/// there is deliberately no global accessor. Dropping the last handle unloads
/// the modules.
pub struct AtlasRegistry {
    host: Arc<CudaHost>,
    modules: HashMap<&'static str, Arc<CudaModule>>,
    /// Raw CUmodule handles for direct cuLaunchKernel access.
    raw_modules: HashMap<&'static str, *mut c_void>,
}

impl Drop for AtlasRegistry {
    /// Unloads this model's modules. Reached when the last `Arc` handle goes,
    /// which — because there is no global accessor — happens exactly when the
    /// owning run ends.
    fn drop(&mut self) {
        let failures = self.unload_raw();
        if !failures.is_empty() {
            // No `tracing` in atlas-core's dependency budget, and a `Drop` has
            // nowhere to return an error to. `release` below is the path that
            // reports properly; this is the backstop.
            eprintln!(
                "atlas: {} module(s) failed to unload: {}",
                failures.len(),
                failures.join("; ")
            );
        }
    }
}

// SAFETY: Same rationale as `RawCudaFunc`: the `raw_modules` map holds
// CUmodule handles obtained at startup from a single CUcontext. The map is
// populated once during registry init and is read-only from that point on,
// so concurrent reads are race-free at the Rust level. CUDA itself
// serializes kernel launches via the stream the caller supplies — this impl
// only asserts that the *handle metadata* is shareable across threads.
unsafe impl Send for AtlasRegistry {}
unsafe impl Sync for AtlasRegistry {}

impl AtlasRegistry {
    /// Load this model's kernel modules into the process CUDA context.
    ///
    /// Each call produces a fresh, independent module set; nothing is shared
    /// with a previously loaded model except the context and stream.
    pub fn load(
        ordinal: usize,
        kernel_blobs: &[(&'static str, &'static [u8])],
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::init(host(ordinal)?, kernel_blobs)?))
    }

    /// The process CUDA context this registry's modules live in.
    pub fn host(&self) -> &Arc<CudaHost> {
        &self.host
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.host.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.host.stream
    }

    /// Module names this registry loaded, for diagnostics.
    pub fn module_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.modules.keys().copied()
    }

    fn init(
        host: Arc<CudaHost>,
        kernel_blobs: &[(&'static str, &'static [u8])],
    ) -> Result<AtlasRegistry> {
        let ctx = &host.ctx;

        let mut modules = HashMap::new();
        let mut raw_modules = HashMap::new();
        for &(name, blob) in kernel_blobs {
            // NVIDIA emits PTX (ASCII text); SCALE/AMD (gfx1151) and HIP
            // emit a binary code object (ELF / clang offload bundle).
            // `cuModuleLoadData` accepts either, but PTX must arrive
            // NUL-terminated (the driver JIT parses it as a C string)
            // while a binary object is self-describing. Sniff per blob.
            let is_binary = blob.starts_with(b"\x7fELF")
                || blob.starts_with(b"__CLANG_OFFLOAD_BUNDLE__")
                || std::str::from_utf8(&blob[..blob.len().min(64)]).is_err();

            // Load via cudarc (safe API) — backs `function()` lookups.
            let ptx = if is_binary {
                Ptx::from_binary(blob.to_vec())
            } else {
                let src = std::str::from_utf8(blob).map_err(|e| {
                    AtlasError::ModuleLoad(format!("{name}: PTX not valid UTF-8: {e}"))
                })?;
                Ptx::from_src(src)
            };
            let module = ctx
                .load_module(ptx)
                .map_err(|e| AtlasError::ModuleLoad(format!("{name}: {e}")))?;

            // The raw handle for launch_on_stream (which avoids cudarc's
            // struct layouts) is the SAME module: derive it instead of
            // JIT-compiling the blob a second time through
            // `cuModuleLoadData`. The double load kept a second copy of
            // every module's SASS resident and doubled driver-JIT time at
            // boot for the entire kernel set. Lifetime: the handle is owned
            // by the `Arc<CudaModule>` stored right beside it — `modules`
            // and `raw_modules` live and die together in this struct, and
            // `unload_raw` no longer unloads (cudarc's `Drop` does).
            raw_modules.insert(name, module.cu_module_raw() as *mut c_void);
            modules.insert(name, module);
        }

        Ok(AtlasRegistry {
            host,
            modules,
            raw_modules,
        })
    }

    /// Look up a cached function handle (cudarc safe API).
    pub fn function(&self, module_name: &str, func_name: &str) -> Result<CudaFunction> {
        let module = self
            .modules
            .get(module_name)
            .ok_or_else(|| AtlasError::ModuleLoad(format!("Module '{module_name}' not loaded")))?;
        module
            .load_function(func_name)
            .map_err(|e| AtlasError::ModuleLoad(format!("{module_name}::{func_name}: {e}")))
    }

    /// Look up a function handle with OnceLock caching (cudarc safe API).
    pub fn function_cached(
        &self,
        cache: &OnceLock<CudaFunction>,
        module_name: &str,
        func_name: &str,
    ) -> Result<CudaFunction> {
        if let Some(f) = cache.get() {
            return Ok(f.clone());
        }
        let func = self.function(module_name, func_name)?;
        let _ = cache.set(func.clone());
        Ok(func)
    }

    /// Look up a raw CUfunction handle with OnceLock caching.
    /// Uses the raw CUDA driver API — no cudarc struct layout dependency.
    pub fn raw_function_cached(
        &self,
        cache: &OnceLock<RawCudaFunc>,
        module_name: &str,
        func_name: &str,
    ) -> Result<RawCudaFunc> {
        if let Some(f) = cache.get() {
            return Ok(*f);
        }
        let raw_mod = self
            .raw_modules
            .get(module_name)
            .ok_or_else(|| AtlasError::ModuleLoad(format!("Module '{module_name}' not loaded")))?;
        let c_name = CString::new(func_name).map_err(|e| {
            AtlasError::ModuleLoad(format!("{module_name}::{func_name}: CString: {e}"))
        })?;
        let mut func: *mut c_void = std::ptr::null_mut();
        let status =
            // SAFETY: pointer cast handles the platform difference between
            // `c_char = i8` (x86_64) and `c_char = u8` (aarch64); we use
            // `.cast()` rather than `as *const i8` so clippy's
            // `unnecessary_cast` is satisfied on x86_64 builds while the
            // call still type-checks on aarch64 (Atlas's actual GB10 target).
            unsafe { cuModuleGetFunction(&mut func, *raw_mod, c_name.as_ptr().cast()) };
        if status != 0 {
            return Err(AtlasError::ModuleLoad(format!(
                "{module_name}::{func_name}: cuModuleGetFunction failed: {}",
                cuda_error_text(status)
            )));
        }
        let raw = RawCudaFunc(func);
        let _ = cache.set(raw);
        Ok(raw)
    }

    /// Retire the raw handles. Idempotent; `Drop` calls it.
    ///
    /// Since the raw map stopped being a second `cuModuleLoadData` of every
    /// blob and became views into the cudarc-owned `modules`, there is
    /// nothing to `cuModuleUnload` here — the `Arc<CudaModule>`s unload the
    /// one real copy when they drop. Draining first keeps the invariant
    /// that no raw handle survives its module: the maps are torn down
    /// together, raw side first.
    pub(crate) fn unload_raw(&mut self) -> Vec<String> {
        self.raw_modules.drain().for_each(drop);
        self.modules.drain().for_each(drop);
        Vec::new()
    }

    /// Get the raw CUstream handle for Atlas's own stream.
    pub fn raw_stream(&self) -> u64 {
        self.host.stream.cu_stream() as u64
    }

    /// Resolve a `__device__` symbol in a loaded PTX module to its device
    /// pointer + byte length. Required for drivers that read/write device
    /// globals without launching a kernel (e.g. InnerQ calibration state).
    /// `symbol` must be the linker-visible name — C++ namespace symbols are
    /// Itanium-mangled (`_ZN7tq_plus14d_innerq_scaleE`).
    pub fn device_symbol(&self, module_name: &str, symbol: &str) -> Result<(u64, usize)> {
        let raw_mod = self
            .raw_modules
            .get(module_name)
            .ok_or_else(|| AtlasError::ModuleLoad(format!("Module '{module_name}' not loaded")))?;
        let c_sym = CString::new(symbol).map_err(|e| {
            AtlasError::ModuleLoad(format!("{module_name}::{symbol}: CString: {e}"))
        })?;
        let mut dptr: u64 = 0;
        let mut bytes: usize = 0;
        let status =
            unsafe { cuModuleGetGlobal_v2(&mut dptr, &mut bytes, *raw_mod, c_sym.as_ptr().cast()) };
        if status != 0 {
            return Err(AtlasError::ModuleLoad(format!(
                "{module_name}::{symbol}: cuModuleGetGlobal_v2 failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok((dptr, bytes))
    }

    /// Async H2D copy into a previously-resolved device pointer.
    ///
    /// # Safety
    /// Caller must ensure `dst` is a valid device pointer and the bytes
    /// pointed to by `src` outlive the copy (host buffers must persist
    /// until the next sync on `stream`).
    pub unsafe fn copy_h2d_async(
        &self,
        dst: u64,
        src: *const c_void,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let status = unsafe { cuMemcpyHtoDAsync_v2(dst, src, bytes, stream) };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuMemcpyHtoDAsync_v2 failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok(())
    }

    /// Async D2H copy from a device pointer. Same lifetime caveats as the
    /// H2D variant.
    ///
    /// # Safety
    /// Caller must keep `dst` alive until `stream` is synchronised.
    pub unsafe fn copy_d2h_async(
        &self,
        dst: *mut c_void,
        src: u64,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let status = unsafe { cuMemcpyDtoHAsync_v2(dst, src, bytes, stream) };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuMemcpyDtoHAsync_v2 failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok(())
    }

    /// Block the calling thread until all prior work on `stream` completes.
    pub fn stream_synchronize(&self, stream: u64) -> Result<()> {
        let status = unsafe { cuStreamSynchronize(stream) };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuStreamSynchronize failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok(())
    }

    /// Launch a kernel on a specified raw CUDA stream.
    ///
    /// When `stream_ptr` comes from the caller (e.g. `torch.cuda.current_stream().cuda_stream`),
    /// this ensures kernels are captured during CUDA graph recording.
    ///
    /// # Safety
    /// - `kernel_params` must contain valid pointers to arguments matching the kernel signature.
    /// - `stream_ptr` must be a valid CUstream handle (or 0 to use Atlas's own stream).
    /// - `raw_func` must be a valid CUfunction obtained from `raw_function_cached`.
    pub unsafe fn launch_on_stream(
        &self,
        raw_func: RawCudaFunc,
        cfg: LaunchConfig,
        stream_ptr: u64,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        // Always use the caller's stream directly. When stream_ptr=0, CUDA
        // treats it as the legacy default stream which has implicit
        // synchronization with all other streams in the same context.
        // Never fall back to Atlas's private stream — that breaks ordering
        // with PyTorch operations and prevents CUDA graph capture.
        let stream = stream_ptr;
        // Opt in to >48KB dynamic shared memory when requested.
        if cfg.shared_mem_bytes > 48 * 1024 {
            const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
            let attr_status = unsafe {
                cuFuncSetAttribute(
                    raw_func.0,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    cfg.shared_mem_bytes as i32,
                )
            };
            if attr_status != 0 {
                return Err(AtlasError::KernelLaunch(format!(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED={}) failed: {}",
                    cfg.shared_mem_bytes,
                    cuda_error_text(attr_status)
                )));
            }
        }
        let status = unsafe {
            cuLaunchKernel(
                raw_func.0,
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes,
                stream as *mut c_void,
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != 0 {
            return Err(AtlasError::KernelLaunch(format!(
                "cuLaunchKernel failed: {} (grid=[{},{},{}], block=[{},{},{}], shared_mem={})",
                cuda_error_text(status),
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes
            )));
        }
        Ok(())
    }
}
