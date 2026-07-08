// SPDX-License-Identifier: AGPL-3.0-only
//! CUTLASS host-wrapper test suite (split for ≤500 LoC).
//!
//! Shared CUDA/test helpers live here; the actual `#[test]` functions live in
//! the `bf16` (dense BF16 + bench/algo sweeps) and `nvfp4` (NVFP4 numeric
//! comparator + transpose) sibling modules.

use super::*;
use std::ffi::c_void;

mod bf16;
mod nvfp4;

pub(super) const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub(super) const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

unsafe extern "C" {
    pub(super) fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    pub(super) fn cudaFree(ptr: *mut c_void) -> i32;
    pub(super) fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: i32) -> i32;
    pub(super) fn cudaDeviceSynchronize() -> i32;
}

pub(super) fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    ((bits + 0x7fff + lsb) >> 16) as u16
}

pub(super) fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

pub(super) fn cuda_check(status: i32, what: &str) {
    assert_eq!(status, 0, "CUDA {what} failed: {status}");
}

pub(super) unsafe fn device_alloc(bytes: usize) -> *mut c_void {
    let mut ptr = std::ptr::null_mut();
    cuda_check(unsafe { cudaMalloc(&mut ptr, bytes) }, "malloc");
    ptr
}

pub(super) unsafe fn copy_h2d<T>(dst: *mut c_void, src: &[T]) {
    cuda_check(
        unsafe {
            cudaMemcpy(
                dst,
                src.as_ptr() as *const c_void,
                std::mem::size_of_val(src),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        },
        "copy h2d",
    );
}

pub(super) unsafe fn copy_d2h<T>(dst: &mut [T], src: *const c_void) {
    cuda_check(
        unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut c_void,
                src,
                std::mem::size_of_val(dst),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        },
        "copy d2h",
    );
}

pub(super) type CutlassVariant = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *mut c_void,
    i32,
    i32,
    i32,
    *mut c_void,
    usize,
    *mut c_void,
) -> i32;

#[allow(clippy::too_many_arguments)]
pub(super) fn run_cutlass_variant(
    name: &str,
    f: CutlassVariant,
    act: *mut c_void,
    weight: *mut c_void,
    out: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let ctx = ctx()?;
    let status = unsafe {
        f(
            act,
            weight,
            out,
            m as i32,
            n as i32,
            k as i32,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            std::ptr::null_mut(),
        )
    };
    if status != 0 {
        bail!("CUTLASS variant {name} failed: status {status} for {m}x{n}x{k}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_cublaslt_algo(
    algo_index: i32,
    act: *mut c_void,
    weight: *mut c_void,
    out: *mut c_void,
    m: usize,
    n: usize,
    k: usize,
) -> Result<i32> {
    let ctx = ctx()?;
    let mut returned = 0i32;
    let status = unsafe {
        atlas_cublaslt_bf16_gemm_act_weight_t_algo(
            act,
            weight,
            out,
            m as i32,
            n as i32,
            k as i32,
            ctx.workspace as *mut c_void,
            ctx.ws_size,
            std::ptr::null_mut(),
            algo_index,
            &mut returned,
        )
    };
    if status != 0 {
        bail!(
            "cuBLASLt algo {algo_index} failed: status {status} returned={returned} for {m}x{n}x{k}"
        );
    }
    Ok(returned)
}

// ---- NVFP4 op-level numeric helpers ------------------------------------
//
// Shared by the NVFP4 comparator/transpose tests. See `nvfp4.rs` for the
// full op-level comparator rationale.

/// E2M1 (FP4) level magnitudes, indexed by the low 3 bits; bit 3 is sign.
pub(super) const E2M1_LEVELS: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

pub(super) fn decode_e2m1(nib: u8) -> f32 {
    let mag = E2M1_LEVELS[(nib & 0x7) as usize];
    if nib & 0x8 != 0 { -mag } else { mag }
}

/// Matches the device `float_to_e2m1` round-to-nearest-bin in the wrapper.
pub(super) fn f32_to_e2m1(x: f32) -> u8 {
    let sign = if x < 0.0 { 0x8u8 } else { 0 };
    let ax = x.abs();
    let mag = if ax <= 0.25 {
        0
    } else if ax <= 0.75 {
        1
    } else if ax <= 1.25 {
        2
    } else if ax <= 1.75 {
        3
    } else if ax <= 2.5 {
        4
    } else if ax <= 3.5 {
        5
    } else if ax <= 5.0 {
        6
    } else {
        7
    };
    sign | mag
}

/// Decode an OCP E4M3 byte (the format `__nv_fp8_e4m3` stores the weight
/// per-group scale in) to f32. Scales are always positive here.
pub(super) fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((byte >> 3) & 0x0f) as i32;
    let m = (byte & 0x07) as i32;
    let val = if e == 0 {
        // subnormal: m/8 * 2^(1-7)
        (m as f32 / 8.0) * 2f32.powi(1 - 7)
    } else {
        // normal: (1 + m/8) * 2^(e-7)
        (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
    };
    sign * val
}

/// Deterministic, full-rank-ish pseudo-random value in roughly [-0.5, 0.5].
pub(super) fn gen_val(seed: u64) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF0);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    ((x >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
}

pub(super) fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}
