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
use anyhow::{Result, anyhow, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::sync::OnceLock;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

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
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

static LIB: OnceLock<Option<Lib>> = OnceLock::new();

fn lib() -> Option<&'static Lib> {
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

    // Managed shim entry: caches scratch internally (no per-call alloc/free → no async
    // use-after-free, no per-call sync). Async on `stream`, ordered with the rest of
    // the layer like the FLA path it replaces.
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
