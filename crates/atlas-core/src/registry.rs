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

use crate::error::{AtlasError, Result};

// Raw CUDA driver API
unsafe extern "C" {
    fn cuModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> i32;
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

/// Global singleton registry.
static REGISTRY: OnceLock<std::result::Result<AtlasRegistry, String>> = OnceLock::new();

/// Cached CUDA modules and a persistent stream.
pub struct AtlasRegistry {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    modules: HashMap<&'static str, Arc<CudaModule>>,
    /// Raw CUmodule handles for direct cuLaunchKernel access.
    raw_modules: HashMap<&'static str, *mut c_void>,
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
    /// Get or initialize the global registry.
    ///
    /// First call loads all PTX modules and creates the persistent stream.
    /// Subsequent calls return the cached registry instantly.
    pub fn get_or_init(
        ordinal: usize,
        ptx_sources: &[(&'static str, &str)],
    ) -> Result<&'static Self> {
        let result = REGISTRY.get_or_init(|| match Self::init(ordinal, ptx_sources) {
            Ok(reg) => Ok(reg),
            Err(e) => Err(format!("{e}")),
        });
        match result {
            Ok(reg) => Ok(reg),
            Err(msg) => Err(AtlasError::ModuleLoad(msg.clone())),
        }
    }

    fn init(ordinal: usize, ptx_sources: &[(&'static str, &str)]) -> Result<AtlasRegistry> {
        let ctx = CudaContext::new(ordinal).map_err(AtlasError::CudaDriver)?;
        let stream = ctx.new_stream().map_err(AtlasError::CudaDriver)?;

        let mut modules = HashMap::new();
        let mut raw_modules = HashMap::new();
        for &(name, src) in ptx_sources {
            // Load via cudarc (safe API, for backward compat)
            let ptx = Ptx::from_src(src);
            let module = ctx
                .load_module(ptx)
                .map_err(|e| AtlasError::ModuleLoad(format!("{name}: {e}")))?;
            modules.insert(name, module);

            // Load via raw CUDA API (for launch_on_stream — avoids cudarc layout issues)
            let src_nul = CString::new(src)
                .map_err(|e| AtlasError::ModuleLoad(format!("{name}: CString: {e}")))?;
            let mut raw_mod: *mut c_void = std::ptr::null_mut();
            let status =
                unsafe { cuModuleLoadData(&mut raw_mod, src_nul.as_ptr() as *const c_void) };
            if status != 0 {
                return Err(AtlasError::ModuleLoad(format!(
                    "{name}: cuModuleLoadData failed: {}",
                    cuda_error_text(status)
                )));
            }
            raw_modules.insert(name, raw_mod);
        }

        Ok(AtlasRegistry {
            ctx,
            stream,
            modules,
            raw_modules,
        })
    }

    /// Get the cached registry (panics if not initialized).
    pub fn get() -> &'static Self {
        REGISTRY
            .get()
            .expect("AtlasRegistry not initialized — call get_or_init first")
            .as_ref()
            .expect("AtlasRegistry initialization failed")
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

    /// Get the raw CUstream handle for Atlas's own stream.
    pub fn raw_stream(&self) -> u64 {
        self.stream.cu_stream() as u64
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
