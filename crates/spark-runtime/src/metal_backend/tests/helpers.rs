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
/// Is this run ALLOWED to have no Metal device?
///
/// Pure, so the rule can be tested without a device and without touching the
/// process environment from a parallel test.
pub(super) fn no_device_is_declared(v: Option<&str>) -> bool {
    v == Some("1")
}

pub(super) fn maybe_backend() -> Option<MetalGpuBackend> {
    let modules = avarok_kernels::metallib_modules();
    match MetalGpuBackend::new(0, &modules) {
        Ok(b) => Some(b),
        Err(e) => {
            // ★ A SKIPPED LEG MUST NOT READ AS A PASS.
            //
            // 43 call sites spell the skip `let Some(b) = maybe_backend() else
            // { return };`, and libtest counts an early return as `ok`. So on a
            // host with no Metal device this suite reports GREEN having executed
            // nothing -- indistinguishable from a suite that ran and passed, and
            // the required context `cargo test --features metal (macOS aarch64)`
            // is exactly that suite.
            //
            // The fix is not to stop skipping; a device-less runner is a real
            // thing. It is to make the run DECLARE it, once, in its environment,
            // so that "we could not look" stops being spelled the same way as
            // "we looked and it was fine".
            if !no_device_is_declared(std::env::var("AVAROK_METAL_NO_DEVICE").ok().as_deref()) {
                panic!(
                    "no Metal device ({e}), and AVAROK_METAL_NO_DEVICE is not set.\n\
                     A leg that genuinely has no device must say so: set \
                     AVAROK_METAL_NO_DEVICE=1 on that step. Without it, skipping \
                     would report this suite as passing having executed nothing."
                );
            }
            eprintln!("skipping metal_backend test (AVAROK_METAL_NO_DEVICE=1 declared): {e}");
            None
        }
    }
}

#[cfg(test)]
mod no_device_rule_tests {
    use super::no_device_is_declared;

    /// The declaration is EXACTLY "1". Anything else -- unset, empty, "0",
    /// "true", "yes" -- leaves the panic armed, because a typo in a CI step
    /// must not silently re-open the hole this closes.
    #[test]
    fn only_an_explicit_one_declares_a_device_less_run() {
        assert!(no_device_is_declared(Some("1")));
        for v in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some("yes"),
            Some(" 1"),
        ] {
            assert!(
                !no_device_is_declared(v),
                "{v:?} must NOT count as a declaration"
            );
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

/// The cosine gate every kernel A/B in this repo is judged by.
///
/// `crates/spark-model/examples/gdn_regresident_microtest.rs:26` uses exactly
/// this value for the GDN output and state; the W8A16 microtests use 0.9995 and
/// the MoE ones 0.999. A parity test that asserts an ABSOLUTE bound instead has
/// to pick a number in the units of whatever it happens to be comparing, and
/// gets it wrong silently: `parity_gdn` bounded `max|expected - actual|` at
/// 0.02 while building inputs whose outputs land near 1e-3, so a kernel that
/// wrote nothing at all passed it.
pub(super) const COSINE_GATE: f64 = 0.9999;

/// Cosine similarity accumulated in f64, the `cos_bf16` of the microtests.
///
/// Returns NaN when either side has zero norm, and every comparison against
/// NaN is false — so `cosine_bf16(..) >= COSINE_GATE` REJECTS an all-zero
/// output rather than congratulating it. That is the property an absolute
/// difference bound cannot have.
pub(super) fn cosine_bf16(a: &[half::bf16], b: &[half::bf16]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i].to_f32() as f64, b[i].to_f32() as f64);
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    d / (na.sqrt() * nb.sqrt())
}

/// Ratio of the two sides' L2 norms, smaller over larger, so it lands in
/// [0, 1] like a cosine and is judged by the SAME constant — no new threshold
/// is invented here.
///
/// Cosine alone is scale-free: a kernel with a uniform 1.005x gain scores
/// 1.0000000 against the reference and sails through. Direction and magnitude
/// are different failures, and the GDN kernel can produce either, so both are
/// checked. Returns NaN when either norm is zero, which fails the gate.
pub(super) fn norm_ratio_f32(a: &[f32], b: &[f32]) -> f64 {
    let na: f64 = a
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    let nb: f64 = b
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    na.min(nb) / na.max(nb)
}

/// [`norm_ratio_f32`] over bf16 inputs.
pub(super) fn norm_ratio_bf16(a: &[half::bf16], b: &[half::bf16]) -> f64 {
    let fa: Vec<f32> = a.iter().map(|v| v.to_f32()).collect();
    let fb: Vec<f32> = b.iter().map(|v| v.to_f32()).collect();
    norm_ratio_f32(&fa, &fb)
}

/// `cos_f32` of the microtests; same zero-norm rule as [`cosine_bf16`].
pub(super) fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        d += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return f64::NAN;
    }
    d / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod cosine_tests {
    use super::{COSINE_GATE, cosine_bf16, cosine_f32};

    /// The control for the defect this replaced: an all-zero "output" must not
    /// pass. Under the old `max|diff| < 0.02` bound against ~1e-3 values it did.
    #[test]
    fn an_all_zero_side_never_passes_the_gate() {
        let truth: Vec<half::bf16> = (0..256)
            .map(|i| half::bf16::from_f32(0.001 * ((i as f32) * 0.0123).sin()))
            .collect();
        let zeros = vec![half::bf16::ZERO; truth.len()];
        let max_abs = truth
            .iter()
            .map(|v| v.to_f32().abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs < 0.02,
            "precondition: these values are small enough that the OLD 0.02 bound \
             accepted an all-zero output ({max_abs} < 0.02)"
        );
        assert!(
            !(cosine_bf16(&truth, &zeros) >= COSINE_GATE),
            "an all-zero output passed the cosine gate"
        );
        assert!(!(cosine_f32(&[0.0, 0.0], &[1.0, 2.0]) >= COSINE_GATE));
    }

    /// ... while an identical side scores exactly 1.0, so the gate is not
    /// simply rejecting everything.
    #[test]
    fn an_identical_side_scores_one() {
        let v: Vec<half::bf16> = (0..64)
            .map(|i| half::bf16::from_f32((i as f32) * 0.01 - 0.3))
            .collect();
        assert!(cosine_bf16(&v, &v) >= COSINE_GATE);
        assert!(cosine_f32(&[1.0, -2.0, 3.5], &[1.0, -2.0, 3.5]) >= COSINE_GATE);
    }

    /// A sign flip on one element of a short vector is a large angular change;
    /// the gate must see it. Guards against a gate that passes anything.
    #[test]
    fn a_single_flipped_element_is_caught() {
        let a: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let b: Vec<f32> = vec![1.0, 1.0, 1.0, -1.0];
        assert!(!(cosine_f32(&a, &b) >= COSINE_GATE));
    }

    /// The reason `norm_ratio_*` exists: cosine CANNOT see a uniform gain, so
    /// a kernel scaling every output by 1.005 is invisible to it. The norm
    /// ratio catches exactly that, judged by the same constant.
    #[test]
    fn a_uniform_gain_is_invisible_to_cosine_and_caught_by_the_norm_ratio() {
        let a: Vec<f32> = (0..128)
            .map(|i| 0.001 * ((i as f32) * 0.0123).sin())
            .collect();
        let b: Vec<f32> = a.iter().map(|v| v * 1.005).collect();
        assert!(
            cosine_f32(&a, &b) >= COSINE_GATE,
            "precondition: cosine is scale-free and must NOT flag a pure gain"
        );
        assert!(
            !(super::norm_ratio_f32(&a, &b) >= COSINE_GATE),
            "the norm ratio must catch the gain cosine cannot see"
        );
    }
}
