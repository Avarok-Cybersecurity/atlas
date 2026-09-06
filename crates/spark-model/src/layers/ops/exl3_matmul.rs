// SPDX-License-Identifier: AGPL-3.0-only

//! Launch wrappers for the native EXL3 (QTIP trellis) matmul kernels —
//! module `exl3_matmul` (`kernels/gb10/common/exl3_matmul.cu`, device code
//! vendored from turboderp's ExLlamaV3, MIT).
//!
//! Data path (fully fused in-kernel, caller passes RAW fp16 activations):
//!
//! ```text
//!     C = ( ((A .* suh) H128/sqrt128) @ W_hat ) H128/sqrt128 .* svh
//! ```
//!
//! Contracts every caller must honor:
//!  * ALL gemm/mgemm/gemv launches are COOPERATIVE (`grid.sync()` inside);
//!    the wrappers set `.cooperative()` and size the grid to stay fully
//!    co-resident — never launch these handles through a plain path.
//!  * `locks`: one per-device int32 buffer of [`EXL3_LOCKS_BYTES`], zeroed
//!    ONCE at allocation ([`exl3_locks_alloc`]) — the in-kernel protocols
//!    self-reset, never re-zero it between launches.
//!  * `suh` (f16 `[k]`), `svh` (f16 `[n]`) and the `A_had` fp16 scratch are
//!    MANDATORY (unconditionally dereferenced). `A_had` needs `m*k` halves
//!    for gemm/gemv (it may alias `A`), and `bszm*m*k` for mgemm — an
//!    undersized mgemm scratch is silent OOB corruption, so [`exl3_mgemm`]
//!    takes the capacity and asserts it.
//!  * gemm/mgemm need 90KB dynamic smem; the wrappers raise
//!    `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` once per handle
//!    (the attribute is sticky on the CUfunction).
//!  * `sm_count`: resolve `GpuBackend::sm_count()` ONCE at construction and
//!    pass it in — the trait forbids per-launch queries.
//!  * mgemm with `min_index >= 0` (expert-range filtering) uses module-scope
//!    `__device__` globals — serialize filtered mgemm launches per device.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

// Explicit `#[path]`: `ops.rs` loads THIS file with one too, so the child
// resolves against `ops/exl3_matmul/` (moe_grouped_a.rs precedent).
#[path = "exl3_matmul/envelope.rs"]
mod envelope;
pub use envelope::{
    EXL3_GEMM_K_BITS, exl3_dense_kernel_names, exl3_gemm_serves_k, exl3_gemv_serves_k,
};
#[path = "exl3_matmul/gemm.rs"]
mod gemm;
pub use gemm::{exl3_gemm, exl3_gemm_abf16, exl3_gemm_abf16_obf16, exl3_gemm_kernel_name};
#[path = "exl3_matmul/mgemm.rs"]
mod mgemm;
pub use mgemm::{
    exl3_bf16_to_f16, exl3_f16_to_bf16, exl3_f16_to_bf16_2d, exl3_f32_to_bf16, exl3_f32_to_bf16_2d,
    exl3_mgemm,
};
#[path = "exl3_matmul/moe_ingress.rs"]
mod moe_ingress;
pub use moe_ingress::{exl3_moe_stage_ingress, exl3_moe_stage_sorted};
#[path = "exl3_matmul/moe_decode.rs"]
mod moe_decode;
pub use moe_decode::{
    Exl3MoeProj, Exl3MoeScratch, exl3_moe_decode_routed, exl3_moe_replicate_a_bf16,
    exl3_moe_stage_routing, exl3_silu_mul_f16,
};
#[path = "exl3_matmul/moe_prefill_det.rs"]
mod moe_prefill_det;
pub use moe_prefill_det::{
    exl3_det_moe_prefill_enabled, set_exl3_det_moe_prefill_from_cli, slot_of_flat,
};
#[path = "exl3_matmul/moe_prefill.rs"]
mod moe_prefill;
pub use moe_prefill::{
    EXL3_MOE_FUSED_K_BITS, EXL3_MOE_MIXED_K_BITS, EXL3_MOE_ROWS_PER_EXPERT_DEFAULT,
    EXL3_MOE_ROWS_PER_EXPERT_ENV, EXL3_MOE_ROWS_PER_EXPERT_LEGACY, EXL3_MOE_ROWS_PER_EXPERT_MIN,
    EXL3_MOE_WIDE_ROWS_KILL_ENV, Exl3MoeExpertTier, Exl3MoeOverflowCtx, Exl3MoePrefillScratch,
    Exl3MoePrefillStats, Exl3MoeRowCap, Exl3MoeRowCapGeometry, Exl3MoeRowCapSource,
    exl3_moe_expert_tier, exl3_moe_fused, exl3_moe_fused_serves, exl3_moe_needs_host_sync,
    exl3_moe_prefill_routed, exl3_moe_row_cap_from_env, exl3_moe_row_cap_kernel_max,
    exl3_moe_temp_slab_bytes, resolve_exl3_moe_row_cap,
};

/// Dynamic shared memory every gemm/mgemm instance is launched with (the
/// upstream `SMEM_MAX`); also the value of the one-time attribute raise.
pub const EXL3_SMEM_MAX: u32 = 90 * 1024;

const MAX_TILES_C: usize = 1024 * 1024;
const MAX_BARRIERS: usize = 1024;
const MOE_SCHED_INTS: usize = 66;
/// Size of the per-device locks buffer (split-k spinlocks + group-barrier
/// counter/sense pairs + MoE scheduler region): 4,202,760 bytes.
pub const EXL3_LOCKS_BYTES: usize = 4 * (MAX_TILES_C + 2 * MAX_BARRIERS + MOE_SCHED_INTS);

/// Hard GEMV row cap (`EXL3_GEMV_MAX_M` upstream); `m > 8` must use the GEMM.
pub const EXL3_GEMV_MAX_M: usize = 8;

// Per-shape tile tables, index = shape_idx 1..4 (0 unused).
const TILESIZE_K: [usize; 5] = [0, 16, 32, 32, 16];
const TILESIZE_N: [usize; 5] = [0, 128, 128, 256, 512];
const BLOCKDIM: [u32; 5] = [0, 256, 512, 512, 256];

/// Allocate + zero the per-device locks buffer. Do this ONCE per device and
/// share the pointer across every exl3 launch on it.
pub fn exl3_locks_alloc(gpu: &dyn GpuBackend) -> Result<DevicePtr> {
    let p = gpu.alloc(EXL3_LOCKS_BYTES)?;
    gpu.memset(p, 0, EXL3_LOCKS_BYTES)?;
    Ok(p)
}

/// One-time 90KB max-dynamic-smem raise per kernel handle (process-lifetime
/// memo — the attribute is sticky on the CUfunction, mirroring upstream's
/// `kernel_attr_set`).
fn raise_smem_once(gpu: &dyn GpuBackend, h: KernelHandle) -> Result<()> {
    static RAISED: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    let set = RAISED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut set = set.lock().expect("exl3 smem-raise set poisoned");
    if set.insert(h.0) {
        gpu.set_kernel_max_dynamic_smem(h, EXL3_SMEM_MAX as usize)?;
    }
    Ok(())
}

fn c_suffix(c_fp32: bool) -> &'static str {
    if c_fp32 { "f32" } else { "f16" }
}

fn ensure_k_cb(k_bits: u32, cb: u32) -> Result<()> {
    ensure!(
        exl3_gemm_serves_k(k_bits),
        "exl3: no kernels instantiated for K={k_bits} (have {EXL3_GEMM_K_BITS:?})"
    );
    ensure!(
        cb == 1 || cb == 2,
        "exl3: no kernels instantiated for cb={cb} (have 1=MCG, 2=MUL1)"
    );
    Ok(())
}

/// `true` when shape_idx has an instantiated kernel for this K (shape 1 is
/// gemm-only, K in {2,4} — the only combination the Blackwell heuristic can
/// pick it for).
fn shape_available(shape: usize, k_bits: u32, multi: bool) -> bool {
    match shape {
        1 => !multi && (k_bits == 2 || k_bits == 4),
        2..=4 => true,
        _ => false,
    }
}

/// Divisibility gate per shape: `k % TILESIZE_K == 0 && n % TILESIZE_N == 0`.
pub fn exl3_gemm_shape_compat(shape: usize, k: usize, n: usize) -> bool {
    (1..=4).contains(&shape)
        && k.is_multiple_of(TILESIZE_K[shape])
        && n.is_multiple_of(TILESIZE_N[shape])
}

/// The Blackwell (== Hopper) branch of upstream `select_gemm_shape`, ported
/// verbatim. `mod_256`/`mod_512` test the UNSCALED n; the size comparisons
/// use `k * bszm_in` / `n * bszm_out` (upstream scales before the switch).
/// `size_m` is ignored upstream. Returns shape_idx 1..4.
pub fn select_exl3_gemm_shape(
    k: usize,
    n: usize,
    k_bits: u32,
    multi: bool,
    bszm_in: usize,
    bszm_out: usize,
) -> usize {
    let mod_256 = n.is_multiple_of(256);
    let mod_512 = n.is_multiple_of(512);
    let k = k * bszm_in;
    let n = n * bszm_out;
    if (k_bits == 4 || k_bits == 2) && !multi && k <= 2048 {
        return 1;
    }
    if k_bits >= 7 {
        if mod_256 && n <= 8192 {
            return if k > 32768 { 3 } else { 2 };
        }
        if mod_512 && n > 32768 {
            return 4;
        }
        return 2;
    }
    if mod_256 && n <= 4096 {
        return if k > 8192 && k_bits >= 3 { 3 } else { 2 };
    }
    if mod_512 && n > 16384 {
        return 4;
    }
    if mod_256 { 3 } else { 2 }
}

/// Heuristic pick + compat/availability fallback (shape 2 is the universal
/// fallback — only needs `n % 128 == 0`).
fn resolve_gemm_shape(
    k: usize,
    n: usize,
    k_bits: u32,
    multi: bool,
    bszm_in: usize,
    bszm_out: usize,
    force_shape: Option<usize>,
) -> Result<usize> {
    if let Some(s) = force_shape {
        ensure!(
            shape_available(s, k_bits, multi) && exl3_gemm_shape_compat(s, k, n),
            "exl3: forced shape {s} unavailable/incompatible for K={k_bits} k={k} n={n} multi={multi}"
        );
        return Ok(s);
    }
    let h = select_exl3_gemm_shape(k, n, k_bits, multi, bszm_in, bszm_out);
    for s in [h, 2, 3, 4, 1] {
        if shape_available(s, k_bits, multi) && exl3_gemm_shape_compat(s, k, n) {
            return Ok(s);
        }
    }
    bail!("exl3: no compatible gemm shape for k={k} n={n} (n must be a multiple of 128)")
}

/// GEMV occupancy assumption (blocks per SM), env-overridable while the
/// backend grows a real occupancy query. 1 is always co-residency-safe at
/// these block sizes; larger values widen the grid cap (a cooperative launch
/// that exceeds true co-residency FAILS cleanly, it does not deadlock).
fn gemv_occ() -> usize {
    static OCC: OnceLock<usize> = OnceLock::new();
    *OCC.get_or_init(|| {
        std::env::var("ATLAS_EXL3_GEMV_OCC")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(1)
    })
}

/// Upstream `exl3_gemv_cfg` heuristic, Blackwell fall-through (arch gate is
/// commented out upstream; Ada-only branches dropped). -1 ineligible /
/// 0 narrow / 1 wide. Ampere-tuned — treat as a starting point and force
/// cfg from the measured table when serving.
fn gemv_cfg_blackwell(k: usize, n: usize, k_bits: u32, narrow_coresident: usize) -> i32 {
    if k_bits == 2 {
        return if n <= 8192 { 0 } else { 1 };
    }
    if n / 32 <= narrow_coresident {
        return 0;
    }
    if k <= 2048 && n <= 8192 {
        return 0;
    }
    if k_bits == 3 {
        return -1;
    }
    if n >= 8192 && k <= 4096 {
        return 1;
    }
    -1
}

/// Small-m fused GEMV (`m <= 8`). Same 10-slot signature/data path as
/// [`exl3_gemm`]; `locks` is unused by the kernel but passed for signature
/// parity. Returns `Ok(false)` when the path refuses (shape/K/cb outside the
/// envelope, or the heuristic declines) — the caller then falls through to
/// [`exl3_gemm`]. `force_cfg`: `Some(0)` narrow / `Some(1)` wide bypasses
/// the heuristic (still shape-gated).
#[allow(clippy::too_many_arguments)]
pub fn exl3_gemv(
    gpu: &dyn GpuBackend,
    a: DevicePtr,
    b_trellis: DevicePtr,
    c: DevicePtr,
    m: usize,
    k: usize,
    n: usize,
    k_bits: u32,
    cb: u32,
    c_fp32: bool,
    locks: DevicePtr,
    suh: DevicePtr,
    a_had: DevicePtr,
    svh: DevicePtr,
    force_cfg: Option<u32>,
    sm_count: u32,
    stream: u64,
) -> Result<bool> {
    if m == 0
        || m > EXL3_GEMV_MAX_M
        || !exl3_gemv_serves_k(k_bits)
        || (cb != 1 && cb != 2)
        || !k.is_multiple_of(128)
        || !n.is_multiple_of(128)
    {
        return Ok(false);
    }
    let coresident = gemv_occ() * sm_count as usize;
    let cfg = match force_cfg {
        Some(cfg) => {
            ensure!(cfg <= 1, "exl3_gemv: cfg must be 0 (narrow) or 1 (wide)");
            cfg as i32
        }
        None => gemv_cfg_blackwell(k, n, k_bits, coresident),
    };
    if cfg < 0 {
        return Ok(false);
    }
    let (block, cols) = if cfg == 0 { (512u32, 32) } else { (256u32, 64) };
    let grid = (n / cols).min(coresident) as u32;
    if grid < 1 {
        return Ok(false);
    }
    let mmode = if m == 1 { 0 } else { 1 };
    let name = format!(
        "exl3_gemv_k{k_bits}_cb{cb}_m{mmode}_cfg{cfg}_{}",
        c_suffix(c_fp32)
    );
    let h = gpu.kernel("exl3_matmul", &name)?;
    // Dynamic smem is 0 (static only) — no attribute raise needed.
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([block, 1, 1])
        .cooperative()
        .arg_ptr(a)
        .arg_ptr(b_trellis)
        .arg_ptr(c)
        .arg_i32(m as i32)
        .arg_i32(k as i32)
        .arg_i32(n as i32)
        .arg_ptr(locks)
        .arg_ptr(suh)
        .arg_ptr(a_had)
        .arg_ptr(svh)
        .launch(stream)?;
    Ok(true)
}

// Shape-heuristic / locks-sizing tests — split on the 500-LoC cap.
#[cfg(test)]
#[path = "exl3_matmul/tests.rs"]
mod tests;
