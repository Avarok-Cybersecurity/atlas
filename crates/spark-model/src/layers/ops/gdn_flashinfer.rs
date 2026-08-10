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
//! dlopen (not link-time) keeps this fully opt-in: the binary builds and runs without the
//! library; it is only loaded when the flag is set. `ATLAS_GDN_LIB` overrides the path.
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
pub fn available() -> bool {
    std::env::var("ATLAS_GDN_FLASHINFER").as_deref() == Ok("1") && lib().is_some()
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
