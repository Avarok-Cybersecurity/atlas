// SPDX-License-Identifier: AGPL-3.0-only

//! Opt-in FlashQLA GDN prefill via `dlopen(libatlasqla.so)` — behind `ATLAS_GDN_FLASHQLA=1`.
//!
//! Bridges Atlas's native packed-QKV + interleaved gate/beta buffers to the
//! FlashQLA TileLang kernel pipeline (cumsum → kkt_solve → fused_gdr_fwd),
//! with optional single-sequence auto-CP controlled inside the shim by
//! `ATLAS_QLA_AUTO_CP=1`.
//! The C ABI shim (`atlas_qla_prefill_log_gate`) takes Atlas's exact native
//! pointers with the gate already in log space. The legacy physical-gate ABI
//! remains exported by the shim for compatibility.
//!
//! dlopen (not link-time) keeps this fully opt-in: the binary builds and runs
//! without the library; it is only loaded when the flag is set. `ATLAS_QLA_LIB`
//! overrides the path.
use anyhow::{Result, anyhow, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::sync::OnceLock;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

type LoadFn = unsafe extern "C" fn() -> c_int;
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
    prefill_log_gate: PackedFn,
}
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

static LIB: OnceLock<Option<Lib>> = OnceLock::new();

fn native_mode() -> bool {
    std::env::var("ATLAS_QLA_IMPL").as_deref() == Ok("native")
}

/// Initialize the selected implementation once during model construction.
/// Native mode resolves Atlas kernel handles from the embedded PTX registry;
/// shim mode retains the legacy dlopen path for A/B measurements.
pub fn initialize(gpu: &dyn GpuBackend) -> bool {
    if std::env::var("ATLAS_GDN_FLASHQLA").as_deref() != Ok("1") {
        return false;
    }
    if native_mode() {
        return super::gdn_flashqla_native::initialize(gpu);
    }
    lib().is_some()
}

fn lib() -> Option<&'static Lib> {
    LIB.get_or_init(|| unsafe {
        let path = std::env::var("ATLAS_QLA_LIB").unwrap_or_else(|_| "libatlasqla.so".to_string());
        let cpath = std::ffi::CString::new(path.clone()).ok()?;
        let h = dlopen(cpath.as_ptr(), RTLD_NOW | RTLD_GLOBAL);
        if h.is_null() {
            tracing::warn!("ATLAS_GDN_FLASHQLA: dlopen('{path}') failed — falling back to FLA");
            return None;
        }
        let load = dlsym(h, c"atlas_qla_load".as_ptr());
        let prefill_log_gate = dlsym(h, c"atlas_qla_prefill_log_gate".as_ptr());
        if load.is_null() || prefill_log_gate.is_null() {
            tracing::warn!(
                "ATLAS_GDN_FLASHQLA: log-gate ABI symbols not found in lib — falling back to FLA"
            );
            return None;
        }
        let load: LoadFn = std::mem::transmute(load);
        let load_ret = load();
        if load_ret != 0 {
            tracing::warn!(
                "ATLAS_GDN_FLASHQLA: atlas_qla_load returned {load_ret} — falling back to FLA"
            );
            return None;
        }
        let auto_cp = std::env::var("ATLAS_QLA_AUTO_CP").as_deref() == Ok("1");
        tracing::info!(
            "ATLAS_GDN_FLASHQLA: FlashQLA GDN kernel loaded (opt-in, auto_cp={auto_cp})"
        );
        Some(Lib {
            prefill_log_gate: std::mem::transmute::<*mut c_void, PackedFn>(prefill_log_gate),
        })
    })
    .as_ref()
}

/// True when `ATLAS_GDN_FLASHQLA=1` AND the library + symbols loaded successfully.
pub fn available() -> bool {
    if std::env::var("ATLAS_GDN_FLASHQLA").as_deref() != Ok("1") {
        return false;
    }
    if native_mode() {
        super::gdn_flashqla_native::available()
    } else {
        lib().is_some()
    }
}

/// Whether Phase 1 must write log-space gates for the FlashQLA consumer.
/// Keeping this predicate here prevents Phase 1 and Phase 2 from selecting
/// different gate representations.
pub fn use_log_gate(exact_replay: bool, kd: usize, vd: usize) -> bool {
    !exact_replay && kd == 128 && vd == 128 && available()
}

/// Run one prefill GDN scan through the FlashQLA kernel on Atlas's native buffers.
///
/// `qkv`: packed `[Q(key_dim)|K(key_dim)|V(value_dim)]` bf16, row stride `conv_dim`.
/// `gate_beta`: interleaved `[gate(nv)|beta(nv)]` fp32, row stride `gb_stride`.
/// `output`: contiguous `[total, value_dim]` bf16. `h_state`: `[nv,kd,vd]` fp32 (final state out).
/// Single-stream only (`num_seqs == 1`); auto-CP is selected in the shim when
/// `ATLAS_QLA_AUTO_CP=1`, otherwise the no-CP pipeline is used.
#[allow(clippy::too_many_arguments)]
pub fn flashqla_gdn_prefill(
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
    if native_mode() {
        return super::gdn_flashqla_native::prefill(
            gpu, qkv, gate_beta, output, h_state, scale, total, nk, nv, kd, vd, conv_dim,
            gb_stride, num_seqs, stream,
        );
    }
    let l = lib().ok_or_else(|| anyhow!("FlashQLA GDN lib unavailable"))?;
    let _ = gpu;

    let ret = unsafe {
        (l.prefill_log_gate)(
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
        bail!("atlas_qla_prefill_log_gate returned {ret}");
    }
    Ok(())
}
