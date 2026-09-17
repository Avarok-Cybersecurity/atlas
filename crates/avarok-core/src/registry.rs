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
//!   let reg = AvarokRegistry::get_or_init(ordinal, &[("gemm", PTX_SRC), ...])?;
//!   let func = reg.function("gemm", "dense_gemm_tc_bf16")?;
//!   unsafe { reg.stream.launch_builder(&func).arg(&ptr).launch(cfg)?; }
//!   reg.stream.synchronize()?;

use std::collections::{HashMap, HashSet};
use std::ffi::{CString, c_void};
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::Ptx;

pub use crate::cuda_host::{CudaHost, host, release};
use crate::elf_symbols::defined_function_symbols;
use crate::error::{AvarokError, Result};

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
/// Model-scoped: the blob set comes from `avarok_kernels::ptx_for_model`, so it
/// changes with the checkpoint. Previously this was fused into a process
/// `OnceLock` singleton whose `get_or_init(ordinal, kernel_blobs)` silently
/// discarded the second caller's blobs — a swapped-in model would have run the
/// *previous* model's kernels with no error at all.
///
/// Obtain one with [`AvarokRegistry::load`] and propagate it (`Arc<AvarokRegistry>`);
/// there is deliberately no global accessor. Dropping the last handle unloads
/// the modules.
pub struct AvarokRegistry {
    host: Arc<CudaHost>,
    modules: HashMap<&'static str, Arc<CudaModule>>,
    /// Raw CUmodule handles for direct cuLaunchKernel access.
    raw_modules: HashMap<&'static str, *mut c_void>,
    /// For every binary (code-object) module whose symbol table could be read,
    /// the kernel names that object actually DEFINES.
    ///
    /// The driver is not a reliable oracle for this: SCALE answers
    /// `cuModuleGetFunction` with SUCCESS for a name the object does not
    /// define, and the launch through that handle is the first thing that
    /// notices. See the module docs on [`crate::elf_symbols`]. So the registry
    /// asks the object, not the driver, and refuses the lookup itself.
    ///
    /// A module is ABSENT from this map when it is text (PTX, the unchanged
    /// path) or when its bytes could not be parsed, and an absent module is not
    /// guarded: the driver's answer stands, exactly as before.
    binary_kernels: HashMap<&'static str, HashSet<String>>,
    /// Raw CUfunction handle -> `"module::kernel"`, for every function this
    /// registry has resolved.
    ///
    /// A `RawCudaFunc` is a bare driver pointer: once it reaches
    /// [`AvarokRegistry::launch_on_stream`] there is nothing left in it that
    /// says what the kernel was, which is why a failed launch used to report
    /// only a grid and a CUresult. This is the reverse map, and it is written
    /// exactly once per kernel in [`AvarokRegistry::raw_function_cached`] (model
    /// init) and read ONLY from a failure path, so a launch pays nothing for
    /// it: no lookup, no lock, no branch on the hot path.
    func_names: Mutex<HashMap<u64, String>>,
}

impl Drop for AvarokRegistry {
    /// Unloads this model's modules. Reached when the last `Arc` handle goes,
    /// which — because there is no global accessor — happens exactly when the
    /// owning run ends.
    fn drop(&mut self) {
        let failures = self.unload_raw();
        if !failures.is_empty() {
            // No `tracing` in avarok-core's dependency budget, and a `Drop` has
            // nowhere to return an error to. `release` below is the path that
            // reports properly; this is the backstop.
            eprintln!(
                "avarok: {} module(s) failed to unload: {}",
                failures.len(),
                failures.join("; ")
            );
        }
    }
}

// SAFETY: Same rationale as `RawCudaFunc`: the `raw_modules` map holds
// CUmodule handles obtained at startup from a single CUcontext. The map is
// populated once during registry init and is read-only from that point on,
// so concurrent reads are race-free at the Rust level. `func_names` IS
// written after init (once per kernel lookup) and is a `Mutex` for exactly
// that reason. CUDA itself serializes kernel launches via the stream the
// caller supplies, so this impl only asserts that the *handle metadata* is
// shareable across threads.
unsafe impl Send for AvarokRegistry {}
unsafe impl Sync for AvarokRegistry {}

impl AvarokRegistry {
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
    ) -> Result<AvarokRegistry> {
        let ctx = &host.ctx;

        let mut modules = HashMap::new();
        let mut raw_modules = HashMap::new();
        let mut binary_kernels: HashMap<&'static str, HashSet<String>> = HashMap::new();
        let mut binary_count = 0usize;
        let mut empty_modules: Vec<&'static str> = Vec::new();
        for &(name, blob) in kernel_blobs {
            // NVIDIA emits PTX (ASCII text); SCALE/AMD (gfx1151) and HIP
            // emit a binary code object (ELF / clang offload bundle).
            // `cuModuleLoadData` accepts either, but PTX must arrive
            // NUL-terminated (the driver JIT parses it as a C string)
            // while a binary object is self-describing. Sniff per blob.
            let is_binary = blob.starts_with(b"\x7fELF")
                || blob.starts_with(b"__CLANG_OFFLOAD_BUNDLE__")
                || std::str::from_utf8(&blob[..blob.len().min(64)]).is_err();

            // Which kernels does this code object actually define? Read once,
            // here, while the blob is in hand. Text (PTX) modules are not
            // sniffed: the NVIDIA path never had this problem.
            if is_binary {
                binary_count += 1;
                match defined_function_symbols(blob) {
                    Some(defined) => {
                        if defined.is_empty() {
                            empty_modules.push(name);
                        }
                        binary_kernels.insert(name, defined);
                    }
                    None => {
                        // Not an ELF64 LE image (a clang offload bundle, say).
                        // Nothing to guard with, so nothing is guarded: the
                        // driver answers as it always did.
                        eprintln!(
                            "avarok: WARN: module '{name}': code object carries no readable ELF \
                             symbol table; kernel lookups fall through to the driver"
                        );
                    }
                }
            }

            // Load via cudarc (safe API) — backs `function()` lookups.
            let ptx = if is_binary {
                Ptx::from_binary(blob.to_vec())
            } else {
                let src = std::str::from_utf8(blob).map_err(|e| {
                    AvarokError::ModuleLoad(format!("{name}: PTX not valid UTF-8: {e}"))
                })?;
                Ptx::from_src(src)
            };
            let module = ctx
                .load_module(ptx)
                .map_err(|e| AvarokError::ModuleLoad(format!("{name}: {e}")))?;

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

        if binary_count > 0 {
            // avarok-core carries no `tracing` (see the note in `Drop`), so the
            // one summary line goes to stderr like every other registry report.
            empty_modules.sort_unstable();
            let empty = empty_modules
                .iter()
                .map(|m| {
                    format!(
                        "; module '{m}' defines no kernels on this target \
                         (optional module compiled out)"
                    )
                })
                .collect::<String>();
            eprintln!(
                "avarok: {binary_count} binary kernel module(s), {} with a readable symbol table{}",
                binary_kernels.len(),
                empty
            );
        }

        Ok(AvarokRegistry {
            host,
            modules,
            raw_modules,
            binary_kernels,
            func_names: Mutex::new(HashMap::new()),
        })
    }

    /// Look up a cached function handle (cudarc safe API).
    pub fn function(&self, module_name: &str, func_name: &str) -> Result<CudaFunction> {
        let module = self
            .modules
            .get(module_name)
            .ok_or_else(|| AvarokError::ModuleLoad(format!("Module '{module_name}' not loaded")))?;
        self.reject_undefined(module_name, func_name)?;
        module
            .load_function(func_name)
            .map_err(|e| AvarokError::ModuleLoad(format!("{module_name}::{func_name}: {e}")))
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
    /// Whether a module of this name was loaded for this run.
    pub fn has_module(&self, module_name: &str) -> bool {
        self.raw_modules.contains_key(module_name)
    }

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
            .ok_or_else(|| AvarokError::ModuleLoad(format!("Module '{module_name}' not loaded")))?;
        self.reject_undefined(module_name, func_name)?;
        let c_name = CString::new(func_name).map_err(|e| {
            AvarokError::ModuleLoad(format!("{module_name}::{func_name}: CString: {e}"))
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
            return Err(AvarokError::ModuleLoad(format!(
                "{module_name}::{func_name}: cuModuleGetFunction failed: {}",
                cuda_error_text(status)
            )));
        }
        let raw = RawCudaFunc(func);
        // Remember what this handle IS, while the names are still in scope.
        // This is the only place the registry mints one, so the map covers
        // every handle any launch can later fail on.
        self.func_names
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(func as u64, format!("{module_name}::{func_name}"));
        let _ = cache.set(raw);
        Ok(raw)
    }

    /// Refuse a lookup the module's own code object says cannot succeed.
    ///
    /// Only binary modules with a parsed symbol table are checked. Everything
    /// else (PTX, an unparsable object, a module that is not loaded at all)
    /// returns `Ok(())` and the driver decides, as it always did.
    ///
    /// The caller in `spark-model` is
    /// `layers::kernel_probe::try_kernel`, which matches on ANY `Err` and
    /// returns `KernelHandle(0)`, so an optional kernel guarded here degrades
    /// to the same "not present" it degrades to on NVIDIA.
    fn reject_undefined(&self, module_name: &str, func_name: &str) -> Result<()> {
        if is_undefined(self.binary_kernels.get(module_name), func_name) {
            return Err(AvarokError::ModuleLoad(undefined_symbol_message(
                module_name,
                func_name,
            )));
        }
        Ok(())
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
        // The names describe handles that are about to become stale; they go
        // with them.
        self.func_names
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
        self.raw_modules.drain().for_each(drop);
        self.binary_kernels.drain().for_each(drop);
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
            .ok_or_else(|| AvarokError::ModuleLoad(format!("Module '{module_name}' not loaded")))?;
        let c_sym = CString::new(symbol).map_err(|e| {
            AvarokError::ModuleLoad(format!("{module_name}::{symbol}: CString: {e}"))
        })?;
        let mut dptr: u64 = 0;
        let mut bytes: usize = 0;
        let status =
            unsafe { cuModuleGetGlobal_v2(&mut dptr, &mut bytes, *raw_mod, c_sym.as_ptr().cast()) };
        if status != 0 {
            return Err(AvarokError::ModuleLoad(format!(
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
            return Err(AvarokError::KernelLaunch(format!(
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
            return Err(AvarokError::KernelLaunch(format!(
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
            return Err(AvarokError::KernelLaunch(format!(
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
                return Err(AvarokError::KernelLaunch(func_attribute_failure_message(
                    &self.func_label(raw_func),
                    cfg.shared_mem_bytes,
                    &cuda_error_text(attr_status),
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
            return Err(AvarokError::KernelLaunch(launch_failure_message(
                &self.func_label(raw_func),
                &cuda_error_text(status),
                [cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2],
                [cfg.block_dim.0, cfg.block_dim.1, cfg.block_dim.2],
                cfg.shared_mem_bytes,
            )));
        }
        Ok(())
    }

    /// How a raw handle should be named in a diagnostic: the
    /// `module::kernel` it was resolved as, plus the pointer itself.
    ///
    /// Error paths only. It takes the `func_names` lock and allocates, which
    /// is free where it is used (a launch has already failed) and would not
    /// be on the launch itself.
    pub fn func_label(&self, raw_func: RawCudaFunc) -> String {
        let handle = raw_func.0 as u64;
        let guard = self
            .func_names
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        kernel_label(guard.get(&handle).map(String::as_str), handle)
    }
}

/// Name a kernel handle for a diagnostic.
///
/// `name` is `None` for a handle this registry never minted (nothing in Avarok
/// produces one today, but `RawCudaFunc` is a public tuple struct anyone can
/// construct), in which case the pointer is all there is to report.
///
/// Pure: no driver call, so the wording is unit-testable on a host with no GPU.
pub fn kernel_label(name: Option<&str>, handle: u64) -> String {
    match name {
        Some(name) => format!("{name} (fn@{handle:#x})"),
        None => format!("<unregistered kernel> (fn@{handle:#x})"),
    }
}

/// The text of a `cuLaunchKernel` failure. `label` comes from
/// [`AvarokRegistry::func_label`] and `err_text` from [`cuda_error_text`], both
/// of which are resolved by the caller so this stays pure and testable.
pub fn launch_failure_message(
    label: &str,
    err_text: &str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
) -> String {
    format!(
        "cuLaunchKernel failed for {label}: {err_text} \
         (grid=[{},{},{}], block=[{},{},{}], shared_mem={shared_mem})",
        grid[0], grid[1], grid[2], block[0], block[1], block[2]
    )
}

/// Whether a lookup is provably doomed: the module is binary, its symbol
/// table parsed, and `func_name` is not in it.
///
/// `None` (a PTX module, an unparsable object, a module that is not loaded)
/// is never a refusal. Split out from [`AvarokRegistry::reject_undefined`] so
/// the decision itself is testable on a host with no CUDA context.
pub fn is_undefined(defined: Option<&HashSet<String>>, func_name: &str) -> bool {
    defined.is_some_and(|set| !set.contains(func_name))
}

/// The text of a lookup refused because the module's code object does not
/// define the name. Pure, so the wording is unit-testable with no GPU.
pub fn undefined_symbol_message(module_name: &str, func_name: &str) -> String {
    format!(
        "{module_name}::{func_name}: not defined in this target's code object \
         (optional module compiled out?)"
    )
}

/// The text of a `cuFuncSetAttribute(MAX_DYNAMIC_SHARED=..)` failure. Same
/// division of labour as [`launch_failure_message`].
pub fn func_attribute_failure_message(label: &str, shared_mem: u32, err_text: &str) -> String {
    format!("cuFuncSetAttribute(MAX_DYNAMIC_SHARED={shared_mem}) failed for {label}: {err_text}")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        func_attribute_failure_message, is_undefined, kernel_label, launch_failure_message,
        undefined_symbol_message,
    };

    #[test]
    fn launch_failure_names_the_kernel() {
        let msg = launch_failure_message(
            &kernel_label(
                Some("qwen3.6-27b_nvfp4::dense_gemv_bf16_batch2"),
                0x7f0c_1234_5678,
            ),
            "CUDA_ERROR_INVALID_IMAGE (200): invalid image",
            [5440, 1, 1],
            [256, 1, 1],
            0,
        );
        assert_eq!(
            msg,
            "cuLaunchKernel failed for qwen3.6-27b_nvfp4::dense_gemv_bf16_batch2 \
             (fn@0x7f0c12345678): CUDA_ERROR_INVALID_IMAGE (200): invalid image \
             (grid=[5440,1,1], block=[256,1,1], shared_mem=0)"
        );
    }

    #[test]
    fn unregistered_handle_still_reports_the_pointer() {
        let msg = launch_failure_message(
            &kernel_label(None, 0x42),
            "CUDA_ERROR_INVALID_VALUE (1): invalid argument",
            [1, 2, 3],
            [64, 1, 1],
            8192,
        );
        assert_eq!(
            msg,
            "cuLaunchKernel failed for <unregistered kernel> (fn@0x42): \
             CUDA_ERROR_INVALID_VALUE (1): invalid argument \
             (grid=[1,2,3], block=[64,1,1], shared_mem=8192)"
        );
    }

    /// The refusal names both halves and says why, because on SCALE the
    /// alternative was a launch-time `CUDA_ERROR_INVALID_IMAGE` with nothing
    /// in it about an optional module.
    #[test]
    fn an_undefined_symbol_is_refused_by_name() {
        assert_eq!(
            undefined_symbol_message("nvfp4_mmq", "avarok_nvfp4_repack"),
            "nvfp4_mmq::avarok_nvfp4_repack: not defined in this target's code object \
             (optional module compiled out?)"
        );
    }

    /// The three answers a code object can give, and what each one licenses.
    #[test]
    fn only_a_parsed_set_that_lacks_the_name_refuses_a_lookup() {
        let defines: HashSet<String> = ["avarok_nvfp4_repack".to_string()].into_iter().collect();
        // Present: the driver is asked, as always.
        assert!(!is_undefined(Some(&defines), "avarok_nvfp4_repack"));
        // Parsed and absent: refused here, before any driver call. This is
        // nvfp4_mmq on gfx1201: SCALE would answer SUCCESS for this name.
        assert!(is_undefined(Some(&defines), "avarok_nvfp4_quantize"));
        // An optional module compiled out defines nothing at all.
        assert!(is_undefined(Some(&HashSet::new()), "avarok_nvfp4_repack"));
        // No parsed set (PTX, or an object we could not read): never refused.
        assert!(!is_undefined(None, "avarok_nvfp4_quantize"));
    }

    #[test]
    fn attribute_failure_names_the_kernel() {
        assert_eq!(
            func_attribute_failure_message(
                &kernel_label(Some("common::prefill_paged"), 0xabc),
                65536,
                "CUDA_ERROR_INVALID_VALUE (1): invalid argument"
            ),
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED=65536) failed for common::prefill_paged \
             (fn@0xabc): CUDA_ERROR_INVALID_VALUE (1): invalid argument"
        );
    }
}
