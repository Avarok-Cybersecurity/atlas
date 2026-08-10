// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in FlashInfer GDN prefill via `dlopen(libatlasgdn.so)` — behind `ATLAS_GDN_FLASHINFER=1`.
//!
//! Bridges Atlas's native packed-QKV + interleaved gate/beta buffers to the AOT-exported
//! FlashInfer chunked gated-delta-rule scan (tensor-core, ~11× the scalar FLA `chunk_delta_h`
//! at the Holo shape — see `3rdparty_patches/gdn_aot/STATUS.md`). The C-ABI shim
//! (`atlas_gdn_prefill_packed`) takes Atlas's exact native pointers: it deinterleaves
//! gate/beta in-shim and reads q/k/v straight out of the packed buffer via `conv_dim`
//! strides (no copy). Atlas's `gate` is already linear α (the kernel does the `logf`),
//! so there is NO gate-space conversion.
//!
//! Two resolution paths, preferred in this order:
//!   1. **Linked (cfg `atlas_gdn_aot`)** — gb10/aarch64 builds statically link the AOT
//!      kernel object + shim (`spark-model/build.rs::build_gdn_aot`), so the symbols
//!      below are real `extern "C"` prototypes checked at link time. No dlopen, no
//!      transmute, no `libcute_dsl_runtime.so`.
//!   2. **dlopen fallback** — other builds (or an explicit `ATLAS_GDN_LIB` override,
//!      which wins over the linked path so the two can be A/B'd; the parity test in
//!      `tests/gdn_aot_parity.rs` leans on this) reach `libatlasgdn.so` via
//!      dlopen/dlsym as before. The binary builds and runs without the library; it is
//!      only loaded when the flag is set.
//!
//! Either way the feature stays fully opt-in behind `ATLAS_GDN_FLASHINFER=1`.
use anyhow::{Result, anyhow, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::sync::OnceLock;

// SAFETY: these two declarations must match libdl/glibc exactly, or every call
// through them is UB. They do:
//   void *dlopen(const char *filename, int flags);
//   void *dlsym(void *handle, const char *symbol);
// `c_char`/`c_int`/`*mut c_void` are the platform-correct spellings of
// `char`/`int`/`void *`, and both are plain C functions with no variadics and no
// callback arguments, so the C ABI mapping is total. They are declared here
// rather than pulled from `libc` to keep this opt-in path dependency-free.
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

// SAFETY (both fn types): these are the Rust spellings of the two `extern "C"`
// entry points in `3rdparty_patches/gdn_aot/gdn_shim.cpp`, and the `transmute`s
// in `lib()` are sound only while they match argument-for-argument:
//
//   void atlas_gdn_load();
//   int  atlas_gdn_prefill_packed_managed(
//            void* qkv, void* gate_beta, void* output, void* h_state,
//            float scale, int total_seqlen, int nk, int nv, int kd, int vd,
//            int conv_dim, int gb_stride, int num_seqs, void* stream);
//
// Verified against gdn_shim.cpp: 4 pointers, one float, eight ints in that
// order, a trailing stream pointer, `int` return. If the shim's signature ever
// changes, THESE TWO TYPES MUST CHANGE WITH IT — `dlsym` returns an untyped
// `void*`, so nothing else in the toolchain will catch a mismatch.
type LoadFn = unsafe extern "C" fn();
// Managed entry: shim owns tensormaps/init/cu scratch (cached) — no per-call alloc/free/sync.
type PackedFn = unsafe extern "C" fn(
    *mut c_void, // qkv
    *mut c_void, // gate_beta
    *mut c_void, // output
    *mut c_void, // h_state (output state)
    c_float,     // scale
    c_int,       // total_seqlen
    c_int,       // nk
    c_int,       // nv
    c_int,       // kd
    c_int,       // vd
    c_int,       // conv_dim
    c_int,       // gb_stride
    c_int,       // num_seqs
    *mut c_void, // stream
) -> c_int;

// Link-time prototypes for the vendored AOT kernel (`libatlas_gdn_aot.a`,
// archived by spark-model/build.rs from gdn_shim.cpp + gdn_cute_rt_stub.cpp +
// gdn_transpose.cu + the committed gdn_holo_0.o). Same two entry points the
// dlopen path resolves by name — but HERE a drift between shim and Rust is a
// link error, not runtime UB. The declarations must still match gdn_shim.cpp
// argument-for-argument (see the PackedFn note above); the linker checks the
// NAMES exist, the parity test checks the behavior.
#[cfg(atlas_gdn_aot)]
mod linked {
    use std::os::raw::{c_float, c_int, c_void};
    unsafe extern "C" {
        pub fn atlas_gdn_load();
        pub fn atlas_gdn_prefill_packed_managed(
            qkv: *mut c_void,
            gate_beta: *mut c_void,
            output: *mut c_void,
            h_state: *mut c_void,
            scale: c_float,
            total_seqlen: c_int,
            nk: c_int,
            nv: c_int,
            kd: c_int,
            vd: c_int,
            conv_dim: c_int,
            gb_stride: c_int,
            num_seqs: c_int,
            stream: *mut c_void,
        ) -> c_int;
    }
}

struct Lib {
    prefill: PackedFn,
}
// SAFETY: the resolved fn pointers are process-global and immutable after load.
// The `dlopen` handle they came from is DELIBERATELY LEAKED (see `lib()`): it is
// never stored in a droppable value and `dlclose` is never called, so the
// library's mapping — and therefore the code these pointers address — lives for
// the whole process. There is no `Library` handle whose drop could unmap it out
// from under a call.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

/// STATIC, DELIBERATELY — process lifecycle. This is a `dlopen` handle and
/// the fn pointers resolved from it. The dynamic loader keys on the SONAME,
/// so a second `dlopen` of the same library returns the same handle and the
/// same code: caching it per model would add bookkeeping without changing
/// what is mapped. Nothing here is model-derived — the pointers are into a
/// shared object, not into a registry that a swap unloads — and the `None`
/// case (library absent) is a property of the machine, not of the model.
static LIB: OnceLock<Option<Lib>> = OnceLock::new();

fn lib() -> Option<&'static Lib> {
    // SAFETY: the whole initialiser is one unsafe block; the obligations are
    //   * `dlopen(cpath.as_ptr(), RTLD_NOW)` — `cpath` is a `CString` alive for
    //     the entire call, so the pointer is a valid NUL-terminated C string.
    //     RTLD_NOW (2) resolves every relocation up front, so a half-resolvable
    //     library fails HERE rather than at first call.
    //   * the returned handle is checked for null before either `dlsym`, and both
    //     `dlsym` results are checked for null before either `transmute`.
    //     `transmute`ing a null `*mut c_void` into a fn pointer and calling it
    //     would be immediate UB, so those two checks are load-bearing.
    //   * `transmute::<*mut c_void, _>` to `LoadFn`/`PackedFn` is sound only
    //     because the shim's C signatures match those types — see the note on
    //     the type aliases above, checked against gdn_shim.cpp.
    //   * LIFETIME: the handle `h` is intentionally never `dlclose`d and never
    //     escapes as a droppable value, so the mapping is leaked for the process
    //     lifetime and the two fn pointers can never dangle. `OnceLock` runs this
    //     at most once, so `atlas_gdn_load()` (which loads the cubin module onto
    //     the device) is called exactly once, as the shim's `g_loaded` expects.
    LIB.get_or_init(|| unsafe {
        // Linked path first: the kernel is inside this binary (cfg atlas_gdn_aot),
        // so no dlopen and no transmute. An explicit ATLAS_GDN_LIB override still
        // routes through dlopen — that is how the parity test A/Bs the linked
        // kernel against an external .so build of the same artifact.
        #[cfg(atlas_gdn_aot)]
        if std::env::var("ATLAS_GDN_LIB").is_err() {
            linked::atlas_gdn_load(); // load the cubin onto the device(s) once
            tracing::info!("ATLAS_GDN_FLASHINFER: FlashInfer GDN kernel linked-in (AOT, opt-in)");
            return Some(Lib {
                prefill: linked::atlas_gdn_prefill_packed_managed as PackedFn,
            });
        }
        let path = std::env::var("ATLAS_GDN_LIB").unwrap_or_else(|_| "libatlasgdn.so".to_string());
        let cpath = std::ffi::CString::new(path.clone()).ok()?;
        let h = dlopen(cpath.as_ptr(), RTLD_NOW);
        if h.is_null() {
            tracing::warn!("ATLAS_GDN_FLASHINFER: dlopen('{path}') failed — falling back to FLA");
            return None;
        }
        let load = dlsym(h, c"atlas_gdn_load".as_ptr());
        let prefill = dlsym(h, c"atlas_gdn_prefill_packed_managed".as_ptr());
        if load.is_null() || prefill.is_null() {
            tracing::warn!("ATLAS_GDN_FLASHINFER: symbols not found in lib — falling back to FLA");
            return None;
        }
        let load: LoadFn = std::mem::transmute(load);
        load(); // load the cubin module onto the device(s) once
        tracing::info!("ATLAS_GDN_FLASHINFER: FlashInfer GDN kernel loaded (opt-in)");
        Some(Lib {
            prefill: std::mem::transmute::<*mut c_void, PackedFn>(prefill),
        })
    })
    .as_ref()
}

/// True when `ATLAS_GDN_FLASHINFER=1` AND the library + symbols loaded successfully.
/// The env read is cached (checked per GDN layer per chunk otherwise); `lib()` is
/// already `OnceLock`-cached, so this is a cheap load after first call.
pub fn available() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("ATLAS_GDN_FLASHINFER").as_deref() == Ok("1"))
        && lib().is_some()
}

/// Surface the GDN kernel in the boot kernel audit (`--check-kernels`).
///
/// Called eagerly from `Qwen3SsmLayer::new`, like every other kernel lookup on
/// the `build_model` path, so the boot table shows `gdn_aot::…` whenever the
/// opt-in is active — the kernel was invisible to the audit for as long as it
/// lived behind a lazy dlopen at first prefill.
///
/// Only a SUCCESSFUL resolution is recorded. Flag-on-but-unresolved keeps the
/// pre-existing contract (a `warn!` from `lib()` + silent FLA fallback):
/// recording `loaded=false` would turn that fallback into a boot-gate failure,
/// and this change is plumbing, not a behavior change. `#[track_caller]` so
/// the audit row's site points at the layer constructor.
#[track_caller]
pub fn record_boot_audit() {
    static ONCE: OnceLock<()> = OnceLock::new();
    let site = std::panic::Location::caller();
    ONCE.get_or_init(|| {
        if available() {
            spark_runtime::kernel_audit::record(
                "gdn_aot",
                "atlas_gdn_prefill_packed_managed",
                true,
                site,
            );
        }
    });
}

/// Run one prefill GDN scan through the FlashInfer kernel on Atlas's native buffers.
///
/// `qkv`: packed `[Q(key_dim)|K(key_dim)|V(value_dim)]` bf16, row stride `conv_dim`.
/// `gate_beta`: interleaved `[gate(nv)|beta(nv)]` fp32, row stride `gb_stride`.
/// `output`: contiguous `[total, value_dim]` bf16. `h_state`: `[nv,kd,vd]` fp32 (final state out).
/// Single-stream only (`num_seqs == 1`); fresh prefill (zero init state).
#[allow(clippy::too_many_arguments)]
pub fn flashinfer_gdn_prefill(
    gpu: &dyn GpuBackend,
    qkv: DevicePtr,
    gate_beta: DevicePtr,
    output: DevicePtr,
    h_state: DevicePtr,
    scale: f32,
    total: u32,
    nk: u32,
    nv: u32,
    kd: u32,
    vd: u32,
    conv_dim: u32,
    gb_stride: u32,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    let l = lib().ok_or_else(|| anyhow!("FlashInfer GDN lib unavailable"))?;
    let _ = gpu; // scratch (tensormaps/init/cu) is now owned+cached inside the shim

    // `num_seqs == 1` is not a style preference — the managed shim writes a
    // fixed-size `long long h[2]` (16 bytes) into its cached `m_cu` cu_seqlens
    // buffer, which is only ever allocated once at `(num_seqs + 1) * 8`. Any
    // num_seqs > 1 both under-fills cu_seqlens and, if the first call had
    // num_seqs == 1, writes past the device allocation. Both call sites pass a
    // literal 1 today; this turns "documented in the doc comment" into an error
    // rather than silent device memory corruption.
    ensure!(
        num_seqs == 1,
        "flashinfer_gdn_prefill: shim is single-sequence only (num_seqs={num_seqs})"
    );

    // Managed shim entry: caches scratch internally (no per-call alloc/free → no async
    // use-after-free, no per-call sync). Async on `stream`, ordered with the rest of
    // the layer like the FLA path it replaces.
    //
    // SAFETY: `l.prefill` is a live, process-lifetime fn pointer whose type matches
    // the shim's C signature (see `PackedFn` / `lib()` above). The four device
    // pointers are passed through as opaque `void*` — Rust never dereferences them,
    // and the shim's reads are bounded by the shape arguments that accompany them:
    // `qkv` is read with row stride `conv_dim` for `total` rows, `gate_beta` with
    // row stride `gb_stride` for `total` rows, `output` written `[total, vd]`, and
    // `h_state` read AND written as `nv*kd*vd` f32. Those extents are the CALLER's
    // obligation (both call sites derive them from the same layer config that sized
    // the buffers) and are NOT checkable from here — a wrong `conv_dim`/`gb_stride`
    // is a device-side OOB, not something this wrapper can detect. `stream` is a
    // valid CUstream handle owned by the backend and outlives the async launch.
    let ret = unsafe {
        (l.prefill)(
            qkv.0 as *mut c_void,
            gate_beta.0 as *mut c_void,
            output.0 as *mut c_void,
            h_state.0 as *mut c_void,
            scale as c_float,
            total as c_int,
            nk as c_int,
            nv as c_int,
            kd as c_int,
            vd as c_int,
            conv_dim as c_int,
            gb_stride as c_int,
            num_seqs as c_int,
            stream as *mut c_void,
        )
    };

    if ret != 0 {
        bail!("atlas_gdn_prefill_packed_managed returned {ret}");
    }
    Ok(())
}
