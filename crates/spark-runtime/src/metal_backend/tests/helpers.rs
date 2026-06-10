// SPDX-License-Identifier: AGPL-3.0-only
//! Shared helpers used by every `metal_backend::tests` submodule —
//! backend construction with graceful skip when no Metal device is
//! available (CI macOS runners are sometimes headless),
//! byte-conversion utilities, and a synthetic MLX-int8 fixture builder.

use crate::metal_backend::MetalGpuBackend;

/// Construct a `MetalGpuBackend` for tests, or return `None` if the
/// host can't open a Metal device. CI runners (especially virtualised
/// macOS hosts on GitHub Actions) report
/// `MTLCreateSystemDefaultDevice returned null` — letting every
/// kernel parity test panic is noise; skipping silently is the
/// right shape for environment-gated tests.
///
/// Callers should `let Some(backend) = maybe_backend() else { return };`
/// at the top of each test fn.
pub(super) fn maybe_backend() -> Option<MetalGpuBackend> {
    let modules = atlas_kernels::metallib_modules();
    match MetalGpuBackend::new(0, &modules) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping metal_backend test: {e}");
            None
        }
    }
}

// ── Byte-conversion helpers (bytemuck-free) ──────────────────

pub(super) fn u32_slice_to_bytes(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

pub(super) fn bf16_slice_to_bytes(values: &[half::bf16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

pub(super) fn bytes_to_bf16_vec(bytes: &[u8]) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

/// Build an MLX-int8 fixture (synthetic packed weights, scales, biases,
/// + the dequantised reference). Returned as raw little-endian byte
/// blobs so callers can `copy_h2d` them straight into `MTLBuffer`s,
/// plus an FP32-friendly `Vec<bf16>` for CPU-side reference math.
///
/// Returned tuple: `(packed_bytes_le, scales_bytes_le, biases_bytes_le, w_bf16_dequant)`.
pub(super) fn build_mlx_fixture(
    n_rows: usize,
    n_cols: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<half::bf16>) {
    assert!(n_cols % 4 == 0 && n_cols % group_size == 0);
    let groups_per_row = n_cols / group_size;

    let mut bytes_flat: Vec<u8> = Vec::with_capacity(n_rows * n_cols);
    for r in 0..n_rows {
        for c in 0..n_cols {
            bytes_flat.push(((r * 13 + c * 5 + 17) % 256) as u8);
        }
    }
    let mut packed: Vec<u32> = Vec::with_capacity(n_rows * n_cols / 4);
    for r in 0..n_rows {
        for c in (0..n_cols).step_by(4) {
            let base = r * n_cols + c;
            let word = (bytes_flat[base] as u32)
                | ((bytes_flat[base + 1] as u32) << 8)
                | ((bytes_flat[base + 2] as u32) << 16)
                | ((bytes_flat[base + 3] as u32) << 24);
            packed.push(word);
        }
    }
    let mut scales: Vec<half::bf16> = Vec::with_capacity(n_rows * groups_per_row);
    let mut biases: Vec<half::bf16> = Vec::with_capacity(n_rows * groups_per_row);
    for r in 0..n_rows {
        for g in 0..groups_per_row {
            scales.push(half::bf16::from_f32(
                0.001 + 0.0005 * r as f32 + 0.0007 * g as f32,
            ));
            biases.push(half::bf16::from_f32(
                -0.05 + 0.01 * r as f32 + 0.005 * g as f32,
            ));
        }
    }

    let mut w_dequant: Vec<half::bf16> = vec![half::bf16::ZERO; n_rows * n_cols];
    for r in 0..n_rows {
        for c in 0..n_cols {
            let byte = bytes_flat[r * n_cols + c] as f32;
            let g = c / group_size;
            let s = scales[r * groups_per_row + g].to_f32();
            let b = biases[r * groups_per_row + g].to_f32();
            w_dequant[r * n_cols + c] = half::bf16::from_f32(byte * s + b);
        }
    }
    (
        u32_slice_to_bytes(&packed),
        bf16_slice_to_bytes(&scales),
        bf16_slice_to_bytes(&biases),
        w_dequant,
    )
}

// ── TurboQuant CPU references (shared by parity_turbo*.rs) ──

/// float → FP8 E4M3 byte. Mirrors `f32_to_e4m3` in
/// `kv_cache_append_turbo8.metal` exactly (saturating, round-half-away
/// on the mantissa).
pub(super) fn cpu_f32_to_e4m3(f: f32) -> u8 {
    let sign: u8 = if f < 0.0 { 0x80 } else { 0x00 };
    let a = f.abs();
    if a >= 448.0 {
        return sign | 0x7E;
    }
    if a < 0.001953125 {
        let m = (a * 512.0).round() as u32;
        return sign | m as u8;
    }
    let mut e = a.log2().floor() as i32;
    if e < -6 {
        e = -6;
    }
    let man = a / (e as f32).exp2();
    let mut m3 = ((man - 1.0) * 8.0).round() as u32;
    if m3 == 8 {
        e += 1;
        m3 = 0;
    }
    sign | (((e + 7) as u8) << 3) | m3 as u8
}

/// FP8 E4M3 byte → float. Mirrors `e4m3_to_f32` in
/// `attention_decode_turbo8.metal`.
pub(super) fn cpu_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xF;
    let m = b & 7;
    if e == 0 {
        return sign * m as f32 * 0.001953125;
    }
    sign * (1.0 + m as f32 * 0.125) * ((e as i32 - 7) as f32).exp2()
}

/// Deterministic pseudo-random bf16-representable test value in ~[-2, 2].
pub(super) fn test_val(i: usize) -> f32 {
    let raw = ((i * 2654435761) >> 7) % 4001;
    f32::from(half::bf16::from_f32(raw as f32 / 1000.0 - 2.0))
}

// Rademacher sign tables (seed=42) mirroring tq_plus_signs —
// the metal build compiles wht_bf16 with -DTQ_PLUS_SIGNS.
#[rustfmt::skip]
pub(super) const TQP_SIGNS1_128: [f32; 128] = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
pub(super) const TQP_SIGNS2_128: [f32; 128] = [
    1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
    1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0,
    -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
];
pub(super) const TQP_SIGNS1_256: [f32; 256] = [
    -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
    1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
    1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0,
    -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
    -1.0,
];
pub(super) const TQP_SIGNS2_256: [f32; 256] = [
    -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
    -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
    1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
    1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
    -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
    1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
    1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
    -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
    -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
];
