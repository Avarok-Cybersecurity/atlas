// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side driver for TurboQuant+ InnerQ per-channel K equalization.
//!
//! Triggers via `TURBO_INNERQ=N` env var (N = calibration token count). The
//! kernel-side state lives in `kernels/gb10/common/tq_plus_innerq_apply.cu`
//! as `__device__` globals inside `namespace tq_plus` — deliberately in the
//! SAME translation unit (= same PTX module; Atlas has no `-rdc` device
//! linking) as the apply/accumulate kernels that read it. This driver
//! manipulates that state directly via the CUDA Driver API:
//!
//!   `cuModuleGetGlobal_v2` → device pointer for each symbol
//!   `cuMemcpyHtoDAsync_v2` / `cuMemcpyDtoHAsync_v2` → push/pull state
//!
//! Two-phase operation:
//!   1. `start()`     — zero counters, set `d_innerq_calibrating = 1`.
//!   2. `maybe_finalize()` — read `d_innerq_count`; once it crosses
//!      `target_tokens`, read `d_innerq_sq_accum`, compute per-channel
//!      scale + scale_inv, upload, set `d_innerq_active = 1`.
//!
//! Math identity: `<Q/s, s·K> = <Q, K>` — the kernel-side apply pass
//! multiplies Q by `scale_inv` pre-WHT and K by `scale` post-WHT, leaving
//! attention dot products unchanged while smoothing K variance across
//! channels.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use std::sync::Arc;

use atlas_core::registry::AtlasRegistry;

// Itanium-mangled names for `tq_plus::*` device globals. The kernel TU is
// `kernels/gb10/common/tq_plus_innerq_apply.cu` — the module that also holds
// `tq_plus_innerq_apply_q/_k`, the only kernels reading this state (module =
// file stem; no [modules] override in common/KERNEL.toml). It MUST be that
// module: each PTX module gets its own copy of `__device__` globals (no
// -rdc), so uploading anywhere else feeds a copy the kernels never see.
const MODULE: &str = "tq_plus_innerq_apply";
const SYM_SCALE: &str = "_ZN7tq_plus14d_innerq_scaleE";
const SYM_SCALE_INV: &str = "_ZN7tq_plus18d_innerq_scale_invE";
const SYM_SQ_ACCUM: &str = "_ZN7tq_plus17d_innerq_sq_accumE";
const SYM_COUNT: &str = "_ZN7tq_plus14d_innerq_countE";
const SYM_ACTIVE: &str = "_ZN7tq_plus15d_innerq_activeE";
const SYM_CALIBRATING: &str = "_ZN7tq_plus20d_innerq_calibratingE";

// Matches INNERQ_MAX_CHANNELS in tq_plus_innerq.cuh. Head dim = 128 today.
const MAX_CHANNELS: usize = 128;

pub struct InnerQDriver {
    /// This model's kernel modules. Held rather than fetched from a global:
    /// the device symbols below live in THESE modules, and a swapped-in model's
    /// driver must never resolve them against the previous model's.
    registry: Arc<AtlasRegistry>,
    pub target_tokens: i32,
    pub strength: f32,
    pub calibrating: AtomicBool,
    pub finalized: AtomicBool,
}

impl InnerQDriver {
    /// Reads `TURBO_INNERQ` and `TURBO_INNERQ_STRENGTH` env vars. Returns
    /// `None` if `TURBO_INNERQ` is unset, unparsable, or `<= 0`.
    pub fn from_env(registry: Arc<AtlasRegistry>) -> Option<Self> {
        let n = std::env::var("TURBO_INNERQ")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .filter(|&n| n > 0)?;
        let strength: f32 = std::env::var("TURBO_INNERQ_STRENGTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&s: &f32| s > 0.0 && s <= 1.0)
            .unwrap_or(0.5);
        Some(Self {
            registry,
            target_tokens: n,
            strength,
            calibrating: AtomicBool::new(false),
            finalized: AtomicBool::new(false),
        })
    }

    /// Enter calibration phase: zero `d_innerq_sq_accum` / `d_innerq_count`
    /// / `d_innerq_active`, set `d_innerq_calibrating = 1`. Idempotent.
    pub fn start(&self) -> Result<()> {
        let reg = &self.registry;
        let stream = reg.raw_stream();

        let zeros_f32 = [0.0f32; MAX_CHANNELS];
        let zero_i32: i32 = 0;
        let one_i32: i32 = 1;

        let (sq_ptr, sq_bytes) = reg
            .device_symbol(MODULE, SYM_SQ_ACCUM)
            .with_context(|| format!("resolve {MODULE}::{SYM_SQ_ACCUM}"))?;
        let (count_ptr, _) = reg.device_symbol(MODULE, SYM_COUNT)?;
        let (active_ptr, _) = reg.device_symbol(MODULE, SYM_ACTIVE)?;
        let (calib_ptr, _) = reg.device_symbol(MODULE, SYM_CALIBRATING)?;

        let copy_bytes = sq_bytes.min(std::mem::size_of_val(&zeros_f32));
        // SAFETY (all four copies): `copy_h2d_async` requires (a) a valid device
        // destination, (b) `bytes` readable from `src`, and (c) `src` alive until
        // the next sync on `stream`.
        //   (a) every `*_ptr` came from `device_symbol()` above, i.e. straight from
        //       `cuModuleGetGlobal_v2` on THIS model's loaded module (the
        //       `self.registry` field exists precisely so a hot-swapped model
        //       cannot resolve against the previous model's copy).
        //   (b) `copy_bytes = min(sq_bytes, size_of_val(&zeros_f32))` is bounded by
        //       BOTH the device symbol's reported length and the 512-byte host
        //       array, so neither side can be over-run. The three scalar copies
        //       move `size_of::<i32>()` out of `&i32` locals — exact by
        //       construction — into `d_innerq_count/_active/_calibrating`, which
        //       are `__device__ int` in tq_plus_innerq.cuh.
        //   (c) `zeros_f32`, `zero_i32` and `one_i32` are stack locals of this fn
        //       and the `stream_synchronize` immediately after the block retires
        //       every copy before they go out of scope.
        unsafe {
            reg.copy_h2d_async(
                sq_ptr,
                zeros_f32.as_ptr() as *const c_void,
                copy_bytes,
                stream,
            )?;
            reg.copy_h2d_async(
                count_ptr,
                &zero_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
            reg.copy_h2d_async(
                active_ptr,
                &zero_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
            reg.copy_h2d_async(
                calib_ptr,
                &one_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }
        // Stack locals must live until the copies retire.
        reg.stream_synchronize(stream)?;

        self.calibrating.store(true, Ordering::Release);
        self.finalized.store(false, Ordering::Release);
        tracing::info!(
            "InnerQ calibration started: target={} tokens, strength={:.2}",
            self.target_tokens,
            self.strength,
        );
        Ok(())
    }

    /// Poll `d_innerq_count`. When it crosses `target_tokens`, pull
    /// `d_innerq_sq_accum`, compute per-channel scale/scale_inv, upload,
    /// and flip `d_innerq_active = 1`. Returns `Ok(true)` on the call
    /// that activates, `Ok(false)` on every other call (including
    /// auto-disable when channels are already balanced).
    pub fn maybe_finalize(&self, group_size: i32) -> Result<bool> {
        if self.finalized.load(Ordering::Acquire) {
            return Ok(false);
        }
        let gs = group_size as usize;
        if gs == 0 || gs > MAX_CHANNELS {
            bail!("group_size {group_size} out of range (1..={MAX_CHANNELS})");
        }

        let reg = &self.registry;
        let stream = reg.raw_stream();

        let (count_ptr, _) = reg.device_symbol(MODULE, SYM_COUNT)?;
        let mut count: i32 = 0;
        // SAFETY: `count_ptr` is `d_innerq_count`, a `__device__ int`, resolved from
        // this model's own module; the transfer is exactly `size_of::<i32>()` bytes
        // into `&mut count`, a live stack local that outlives the copy — the
        // `stream_synchronize` on the next line retires the DMA before `count` is
        // read. `&mut count` is the only reference to it here.
        unsafe {
            reg.copy_d2h_async(
                &mut count as *mut i32 as *mut c_void,
                count_ptr,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }
        reg.stream_synchronize(stream)?;

        if count < self.target_tokens {
            return Ok(false);
        }

        let (sq_ptr, sq_bytes) = reg.device_symbol(MODULE, SYM_SQ_ACCUM)?;
        let mut sq_accum = [0.0f32; MAX_CHANNELS];
        let accum_bytes = gs * std::mem::size_of::<f32>();
        // The host side is bounded by the `gs <= MAX_CHANNELS` check above, but the
        // DEVICE side is only bounded by `MAX_CHANNELS == INNERQ_MAX_CHANNELS` in
        // tq_plus_innerq.cuh — a constant in a different language in a different
        // tree. `device_symbol` hands back the symbol's real length; use it rather
        // than assume the two constants are still in step.
        if accum_bytes > sq_bytes {
            bail!(
                "{MODULE}::{SYM_SQ_ACCUM} is {sq_bytes} bytes but group_size {group_size} \
                 needs {accum_bytes} — INNERQ_MAX_CHANNELS and MAX_CHANNELS disagree"
            );
        }
        // SAFETY: `copy_d2h_async` needs a valid device source, `bytes` writable at
        // `dst`, and `dst` alive until the sync. `sq_ptr` is `d_innerq_sq_accum`
        // from this model's module; `accum_bytes = gs * 4` is `<= sq_bytes` (checked
        // immediately above) on the device side and `<= MAX_CHANNELS * 4 =
        // size_of_val(&sq_accum)` on the host side, since `gs <= MAX_CHANNELS` was
        // enforced at the top of this fn. `sq_accum` is a fully-initialised stack
        // array that lives until the end of the fn, and the `stream_synchronize`
        // below retires the copy before it is read.
        unsafe {
            reg.copy_d2h_async(
                sq_accum.as_mut_ptr() as *mut c_void,
                sq_ptr,
                accum_bytes,
                stream,
            )?;
        }
        reg.stream_synchronize(stream)?;

        // Identity-preserving equalization: scale[i] = (mean_rms / rms[i])
        // ^strength, clamped to [0.5, 2.0]; auto-disable if max/min ratio
        // < 1.2 either way.
        let count_f = count as f32;
        let mut rms = [0.0f32; MAX_CHANNELS];
        let mut mean_rms = 0.0f32;
        for i in 0..gs {
            rms[i] = (sq_accum[i] / count_f).sqrt();
            mean_rms += rms[i];
        }
        mean_rms /= gs as f32;

        let mut scale = [1.0f32; MAX_CHANNELS];
        let mut scale_inv = [1.0f32; MAX_CHANNELS];
        let mut max_ratio = 0.0f32;
        let mut min_ratio = 1e30f32;
        for i in 0..gs {
            let ratio = if rms[i] > 1e-10 {
                mean_rms / rms[i]
            } else {
                1.0
            };
            let s = ratio.powf(self.strength).clamp(0.5, 2.0);
            scale[i] = s;
            scale_inv[i] = 1.0 / s;
            if ratio > max_ratio {
                max_ratio = ratio;
            }
            if ratio < min_ratio {
                min_ratio = ratio;
            }
        }

        let (calib_ptr, _) = reg.device_symbol(MODULE, SYM_CALIBRATING)?;
        let zero_i32: i32 = 0;
        // SAFETY: `calib_ptr` is `d_innerq_calibrating`, a `__device__ int` in this
        // model's module; the copy is exactly `size_of::<i32>()` bytes out of
        // `&zero_i32`. `zero_i32` is declared on the line above and stays in scope
        // to the end of the fn, so it outlives the copy on BOTH exits: the
        // auto-disable branch syncs before returning, and the normal path syncs
        // after the scale uploads below.
        unsafe {
            reg.copy_h2d_async(
                calib_ptr,
                &zero_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }

        if max_ratio < 1.2 && min_ratio > (1.0 / 1.2) {
            reg.stream_synchronize(stream)?;
            self.calibrating.store(false, Ordering::Release);
            self.finalized.store(true, Ordering::Release);
            tracing::info!(
                "InnerQ auto-disabled (channels already balanced: max_ratio={max_ratio:.3}, \
                 min_ratio={min_ratio:.3})"
            );
            return Ok(false);
        }

        let (scale_ptr, scale_bytes) = reg.device_symbol(MODULE, SYM_SCALE)?;
        let (scale_inv_ptr, scale_inv_bytes) = reg.device_symbol(MODULE, SYM_SCALE_INV)?;
        let (active_ptr, _) = reg.device_symbol(MODULE, SYM_ACTIVE)?;
        let one_i32: i32 = 1;
        let copy_bytes = gs * std::mem::size_of::<f32>();
        // These two are WRITES into device globals — an over-run corrupts whatever
        // the linker placed after them in the module's global segment, silently.
        // Bound against the symbols' real lengths rather than against
        // MAX_CHANNELS's agreement with INNERQ_MAX_CHANNELS.
        if copy_bytes > scale_bytes || copy_bytes > scale_inv_bytes {
            bail!(
                "{MODULE} scale symbols are {scale_bytes}/{scale_inv_bytes} bytes but \
                 group_size {group_size} needs {copy_bytes} — INNERQ_MAX_CHANNELS and \
                 MAX_CHANNELS disagree"
            );
        }
        // SAFETY: `scale_ptr`/`scale_inv_ptr` are `d_innerq_scale` /
        // `d_innerq_scale_inv` from this model's module. `copy_bytes = gs * 4` is
        // `<= scale_bytes`/`scale_inv_bytes` on the device side (checked directly
        // above) and `<= MAX_CHANNELS * 4 = size_of_val(&scale)` on the host side
        // (`gs <= MAX_CHANNELS`, enforced at the top of this fn). `scale` and
        // `scale_inv` are `[f32; MAX_CHANNELS]` stack arrays initialised to 1.0 at
        // declaration and fully written for indices `0..gs` by the loop above; they
        // stay in scope past the `stream_synchronize` that retires these copies.
        unsafe {
            reg.copy_h2d_async(
                scale_ptr,
                scale.as_ptr() as *const c_void,
                copy_bytes,
                stream,
            )?;
            reg.copy_h2d_async(
                scale_inv_ptr,
                scale_inv.as_ptr() as *const c_void,
                copy_bytes,
                stream,
            )?;
        }
        // Order: scale uploads must retire before active flips so any kernel
        // observing active=1 sees the finalized scales (cuMemcpyAsync within
        // the same stream is already strict-order, but the active flag is
        // visible to kernels on OTHER streams once a host-side sync passes).
        reg.stream_synchronize(stream)?;
        // SAFETY: `active_ptr` is `d_innerq_active`, a `__device__ int` in this
        // model's module; the copy is exactly `size_of::<i32>()` bytes out of
        // `&one_i32`, a stack local declared above that outlives the
        // `stream_synchronize` on the line after.
        unsafe {
            reg.copy_h2d_async(
                active_ptr,
                &one_i32 as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
                stream,
            )?;
        }
        reg.stream_synchronize(stream)?;

        self.calibrating.store(false, Ordering::Release);
        self.finalized.store(true, Ordering::Release);
        tracing::info!(
            "InnerQ scales activated (group_size={group_size}, max_ratio={max_ratio:.3}, \
             strength={:.2})",
            self.strength,
        );
        Ok(true)
    }
}
