// SPDX-License-Identifier: AGPL-3.0-only

//! CPU reference for Atlas's block-scaled FP8 (E4M3) arithmetic.
//!
//! Three GPU kernels define the format the dense FP8 route runs on, and this
//! module is their host mirror, written so a CPU test can pin the numerics
//! and a GPU oracle (`examples/fp8_ada_microtest.rs`) can compare a device
//! result against something that is not another kernel:
//!
//! * `quantize_bf16_to_fp8_blockscaled.cu`: weights, one FP32 scale per
//!   128x128 tile, `scale = max|x| / 448` floored at `1e-12`, each element
//!   `round_e4m3(x / scale)` saturating to +-448;
//! * `per_token_group_quant_fp8.cu`: activations, the same rule over 1x128
//!   groups of one token;
//! * `w8a16_gemv.cu` (and the Ada drop-in `fp8_gemv_ada.cu`): the decode
//!   dot product, whose reduction ORDER [`gemv_w8a16_exact`] reproduces
//!   operation for operation, so a device result can be required to match
//!   bit for bit.
//!
//! The W8A8 prefill GEMM ([`gemm_w8a8_ref`]) cannot be modelled bit-exactly:
//! the order of the additions inside one `mma.sync` is the hardware's. The
//! reference keeps the kernel's two-level structure (exact inner sum per
//! 128-wide K group, then an FP32 fold with `a_scale * b_scale`) and is used
//! with a tolerance.
//!
//! Decoding goes through [`avarok_core::numeric::fp8_e4m3_to_f32`], the
//! table every Atlas host path already shares (NaN bytes decode to +-0).

use avarok_core::numeric::fp8_e4m3_to_f32;

/// Largest finite E4M3 magnitude; a group's absolute maximum maps here.
pub const E4M3_MAX: f32 = 448.0;
/// Edge of a weight scale block, both axes, and of an activation group.
pub const FP8_BLOCK: usize = 128;
/// The kernels' floor on a scale, so an all-zero group stays finite.
pub const SCALE_FLOOR: f32 = 1e-12;

/// `f32 -> E4M3` with round-to-nearest-even and SATFINITE saturation, as
/// `__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)` (one
/// `cvt.rn.satfinite.e4m3x2.f32` on sm_89+).
///
/// Never returns `0x7F` / `0xFF` for a finite input: those are E4M3's NaNs.
/// NaN input encodes to `0x7F`, as the hardware does.
pub fn e4m3_encode_satfinite(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let sign: u8 = if v.is_sign_negative() { 0x80 } else { 0x00 };
    let a = v.abs();
    if a >= E4M3_MAX {
        return sign | 0x7E;
    }
    // Codes are contiguous in magnitude: code = (biased_exp << 3) + mantissa,
    // and a mantissa that rounds up to 8 carries into the next exponent,
    // which is the next code. So one rounded integer per binade suffices.
    const MIN_NORMAL: f32 = 0.015_625; // 2^-6
    let code = if a < MIN_NORMAL {
        // Subnormal: value = m * 2^-9, m in 0..=7 (8 is the first normal).
        (a * 512.0).round_ties_even() as u32
    } else {
        // `a` is a normal f32 here, so its exponent field is the binade.
        let e = ((a.to_bits() >> 23) & 0xFF) as i32 - 127; // -6..=8
        let frac = a / f32::from_bits(((e + 127) as u32) << 23) - 1.0; // [0, 1), exact
        let m = (frac * 8.0).round_ties_even() as u32; // 0..=8
        (((e + 7) as u32) << 3) + m
    };
    // `a < 448` keeps the top binade's mantissa at <= 6, i.e. code <= 0x7E;
    // the min() is a guard on that reasoning, not a reachable path.
    sign | code.min(0x7E) as u8
}

/// One group's scale, exactly as both quantizer kernels compute it.
pub fn group_scale(values: impl Iterator<Item = f32>) -> f32 {
    let amax = values.fold(0.0f32, |m, x| m.max(x.abs()));
    let s = amax / E4M3_MAX;
    if s < SCALE_FLOOR { SCALE_FLOOR } else { s }
}

/// One element under a group scale: `round_e4m3(clamp(x / scale, +-448))`.
///
/// The clamp is spelled as the kernels spell it, `fmaxf(fminf(v, 448), -448)`:
/// Rust's `min`/`max` return the non-NaN operand as `fminf`/`fmaxf` do, so a
/// NaN quotient (a NaN or infinite element) saturates to +448 (`0x7E`) as on
/// the device, where `clamp` would pass the NaN through to `0x7F`. That NaN
/// behavior is why `clippy::manual_clamp` is allowed here.
#[allow(clippy::manual_clamp)]
pub fn quantize_with_scale(x: f32, scale: f32) -> u8 {
    let v = (x / scale).min(E4M3_MAX).max(-E4M3_MAX);
    e4m3_encode_satfinite(v)
}

/// A `[n, k]` weight in Atlas's block-scaled FP8 layout: `bytes[n*k]` E4M3
/// row-major and `scales[ceil(n/128) * ceil(k/128)]` FP32 row-major, the
/// layout `Fp8Weight` with `WeightQuantFormat::Fp8BlockScaled` points at.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockScaledFp8 {
    pub n: usize,
    pub k: usize,
    pub bytes: Vec<u8>,
    pub scales: Vec<f32>,
}

impl BlockScaledFp8 {
    /// Scale-grid columns, `ceil(k / 128)`. Rounding this down shears the
    /// grid by one column per row, which reads as "FP8 is inaccurate" rather
    /// than as an indexing bug (see the quantizer kernel's NOTE).
    pub fn k_blocks(&self) -> usize {
        self.k.div_ceil(FP8_BLOCK)
    }

    /// The scale that applies to element `[row, col]`.
    pub fn scale_at(&self, row: usize, col: usize) -> f32 {
        self.scales[(row / FP8_BLOCK) * self.k_blocks() + col / FP8_BLOCK]
    }

    /// `E4M3(byte) * scale` for one element, the value every kernel computes.
    pub fn value_at(&self, row: usize, col: usize) -> f32 {
        fp8_e4m3_to_f32(self.bytes[row * self.k + col]) * self.scale_at(row, col)
    }

    /// The whole weight dequantized to FP32, row-major `[n, k]`.
    pub fn dequant(&self) -> Vec<f32> {
        (0..self.n * self.k)
            .map(|i| self.value_at(i / self.k, i % self.k))
            .collect()
    }
}

/// Host mirror of `quantize_bf16_to_fp8_blockscaled`: `x` is the `[n, k]`
/// source (BF16 values widened to f32), partial edge tiles included.
pub fn quantize_blockscaled(x: &[f32], n: usize, k: usize) -> BlockScaledFp8 {
    assert_eq!(
        x.len(),
        n * k,
        "quantize_blockscaled: {} values for [{n},{k}]",
        x.len()
    );
    let (nb, kb) = (n.div_ceil(FP8_BLOCK), k.div_ceil(FP8_BLOCK));
    let mut bytes = vec![0u8; n * k];
    let mut scales = vec![0f32; nb * kb];
    for bn in 0..nb {
        let rows = bn * FP8_BLOCK..((bn + 1) * FP8_BLOCK).min(n);
        for bk in 0..kb {
            let cols = bk * FP8_BLOCK..((bk + 1) * FP8_BLOCK).min(k);
            let tile = || {
                rows.clone()
                    .flat_map(|r| cols.clone().map(move |c| r * k + c))
            };
            let s = group_scale(tile().map(|i| x[i]));
            scales[bn * kb + bk] = s;
            for i in tile() {
                bytes[i] = quantize_with_scale(x[i], s);
            }
        }
    }
    BlockScaledFp8 {
        n,
        k,
        bytes,
        scales,
    }
}

/// Host mirror of `per_token_group_quant_fp8`: `x` is `[m, k]` with
/// `k % 128 == 0`. Returns `(bytes[m*k], scales[m * k/128])`.
pub fn per_token_group_quant(x: &[f32], m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(x.len(), m * k);
    assert!(
        k.is_multiple_of(FP8_BLOCK),
        "per_token_group_quant: k={k} not a multiple of 128"
    );
    let groups = k / FP8_BLOCK;
    let mut bytes = vec![0u8; m * k];
    let mut scales = vec![0f32; m * groups];
    for row in 0..m {
        for g in 0..groups {
            let span = row * k + g * FP8_BLOCK..row * k + (g + 1) * FP8_BLOCK;
            let s = group_scale(x[span.clone()].iter().copied());
            scales[row * groups + g] = s;
            for i in span {
                bytes[i] = quantize_with_scale(x[i], s);
            }
        }
    }
    (bytes, scales)
}

/// Lanes per output row in the decode GEMVs (`256 threads / 4 rows`).
const GEMV_LANES: usize = 64;
/// K values one lane consumes per chunk (one 16-byte load of E4M3 bytes).
const GEMV_CHUNK: usize = 16;

/// EXACT host model of the W8A16 decode GEMV's arithmetic, before the final
/// BF16 round: `out[n]` is the FP32 value the kernel passes to
/// `__float2bfloat16`.
///
/// Same lane walk (lane `l` of 64 visits chunks `l, l+64, ...`), same
/// per-element `w = decode * scale; acc = acc + a * w` with separate rounded
/// multiply and add (Rust never contracts to FMA), same 5-step `shfl.down`
/// butterfly within each 32-lane warp, same final two-warp add. `a` holds the
/// BF16 activations widened to f32 and must have `w.k` entries; `w.k` must be
/// a multiple of 16.
pub fn gemv_w8a16_exact(a: &[f32], w: &BlockScaledFp8) -> Vec<f32> {
    assert_eq!(a.len(), w.k);
    assert!(
        w.k.is_multiple_of(GEMV_CHUNK),
        "gemv_w8a16_exact: k={} not a multiple of 16",
        w.k
    );
    let chunks = w.k / GEMV_CHUNK;
    (0..w.n)
        .map(|row| {
            let mut lane_acc = [0f32; GEMV_LANES];
            for (lane, acc) in lane_acc.iter_mut().enumerate() {
                let mut c = lane;
                while c < chunks {
                    let scale = w.scale_at(row, c * GEMV_CHUNK);
                    for i in 0..GEMV_CHUNK {
                        let col = c * GEMV_CHUNK + i;
                        let wv = fp8_e4m3_to_f32(w.bytes[row * w.k + col]) * scale;
                        *acc += a[col] * wv;
                    }
                    c += GEMV_LANES;
                }
            }
            let warp = |half: &[f32]| -> f32 {
                let mut v = [0f32; 32];
                v.copy_from_slice(half);
                // shfl_down butterfly: lane i adds lane i+offset (lanes past
                // 31 return their own value, which only lanes >= 32-offset
                // read, and lane 0's result never depends on them).
                for offset in [16usize, 8, 4, 2, 1] {
                    let prev = v;
                    for i in 0..32 {
                        let src = if i + offset < 32 {
                            prev[i + offset]
                        } else {
                            prev[i]
                        };
                        v[i] = prev[i] + src;
                    }
                }
                v[0]
            };
            warp(&lane_acc[..32]) + warp(&lane_acc[32..])
        })
        .collect()
}

/// Reference for the W8A8 block-scaled prefill GEMM:
/// `c[m, n] = sum_g (sum_{k in g} A[m,k] * B[n,k]) * a_scale[m,g] * b_scale[n/128,g]`
/// with the inner sums over raw E4M3 values taken exactly (f64), and the fold
/// done in f32 as the kernels do it. Returns FP32 before the BF16 round.
pub fn gemm_w8a8_ref(a_bytes: &[u8], a_scales: &[f32], m: usize, w: &BlockScaledFp8) -> Vec<f32> {
    let k = w.k;
    assert!(
        k.is_multiple_of(FP8_BLOCK),
        "gemm_w8a8_ref: k={k} not a multiple of 128"
    );
    let groups = k / FP8_BLOCK;
    assert_eq!(a_bytes.len(), m * k);
    assert_eq!(a_scales.len(), m * groups);
    let mut out = vec![0f32; m * w.n];
    for row in 0..m {
        for col in 0..w.n {
            let mut outer = 0f32;
            for g in 0..groups {
                let mut inner = 0f64;
                for kk in g * FP8_BLOCK..(g + 1) * FP8_BLOCK {
                    let av = fp8_e4m3_to_f32(a_bytes[row * k + kk]) as f64;
                    let bv = fp8_e4m3_to_f32(w.bytes[col * k + kk]) as f64;
                    inner += av * bv;
                }
                let s = a_scales[row * groups + g] * w.scales[(col / FP8_BLOCK) * groups + g];
                outer += inner as f32 * s;
            }
            out[row * w.n + col] = outer;
        }
    }
    out
}

/// The scale of each [`gemm_w8a8_ref`] output's terms:
/// `sum_g |a_scale[m,g] * b_scale[n/128,g]| * sum_{k in g} |A[m,k] * B[n,k]|`.
///
/// A device result's accumulation error is proportional to this, not to the
/// result: where the terms cancel, the output is near zero but its error is
/// not. A tolerance for the GEMM is therefore a few BF16 ulps of the result
/// plus a small multiple of this bound.
pub fn gemm_w8a8_term_mag(
    a_bytes: &[u8],
    a_scales: &[f32],
    m: usize,
    w: &BlockScaledFp8,
) -> Vec<f32> {
    let k = w.k;
    let groups = k / FP8_BLOCK;
    assert_eq!(a_bytes.len(), m * k);
    assert_eq!(a_scales.len(), m * groups);
    let mut out = vec![0f32; m * w.n];
    for row in 0..m {
        for col in 0..w.n {
            let mut mag = 0f64;
            for g in 0..groups {
                let inner: f64 = (g * FP8_BLOCK..(g + 1) * FP8_BLOCK)
                    .map(|kk| {
                        let av = fp8_e4m3_to_f32(a_bytes[row * k + kk]) as f64;
                        let bv = fp8_e4m3_to_f32(w.bytes[col * k + kk]) as f64;
                        (av * bv).abs()
                    })
                    .sum();
                let s = a_scales[row * groups + g] * w.scales[(col / FP8_BLOCK) * groups + g];
                mag += inner * f64::from(s.abs());
            }
            out[row * w.n + col] = mag as f32;
        }
    }
    out
}

/// Round to BF16 and back, ties to even: what the kernels' final
/// `__float2bfloat16` stores.
pub fn round_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

#[cfg(test)]
#[path = "fp8_blockscale_ref_tests.rs"]
mod tests;
