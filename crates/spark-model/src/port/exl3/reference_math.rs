// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/src/core/exl3_format.zig:136-187 (rotate, rotateBlock, dequantize: W = diag(suh) H_K Wq H_N
// diag(svh)) and zig/kernels/cuda/exl3_decode.cuh:138-155 (fwht128, HAD_SCALE), after ExLlamaV3
// (turboderp-org/exllamav3, MIT, Copyright (c) 2025 Turboderp). Apache-2.0; see EXL3-ATTRIBUTION.md.
// Changes for Atlas: translated from Zig and CUDA C++ to Rust; the FP64 rotation uses a fast transform
// instead of the explicit 128 x 128 sign sum; the bit-exact fp32 mirrors of the Atlas rotation and BF16
// dequant kernels and the GEMV error bound are new for Atlas.

//! Responsibility: the Hadamard side of the EXL3 CPU oracle.
//!
//! * Bit-exact MIRRORS of two kernels (`exl3_hadamard_ada.cu`): [`rot_in_mirror`] (the input rotation,
//!   fp16 out) and [`dequant_bf16_mirror`] (the BF16 dequant). Both follow the kernels' fp32 operation
//!   order exactly (Ada common builds with `--fmad=false`, so nothing is contracted), so the GPU result
//!   must equal them byte for byte.
//! * FP64 references: [`dequant_f64`] (the full weight from its definition), [`dense_f64`], and
//!   [`gemv_row_ref`], the Hadamard-domain product of one row with its error [`GemvBound`] for the linear
//!   kernel, whose tensor-core sums have no fixed order.

/// 1 / sqrt(128) as the kernels' f32 literal `0.08838834764831845f` rounds (bits 0x3DB504F3).
pub const HAD_SCALE: f32 = f32::from_bits(0x3DB5_04F3);
/// 1 / sqrt(128) in f64.
pub fn had_scale_f64() -> f64 {
    1.0 / 128f64.sqrt()
}

pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    half::bf16::from_bits(bits).to_f32()
}

/// Round to nearest even fp16 (`__floats2half2_rn`).
pub fn f32_to_f16(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

/// Round to nearest even bf16 (`__floats2bfloat162_rn`).
pub fn f32_to_bf16(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

/// The kernels' `fwht128` in fp32: in-place butterflies `(a + b, a - b)` at strides 1, 2, ..., 64. The
/// warp form (4 values a lane, then `__shfl_xor` at lane strides 1..16) performs exactly these
/// operations on exactly these operands, so the results are bit-identical.
pub fn fwht128_f32(v: &mut [f32; 128]) {
    let mut d = 1;
    while d < 128 {
        let mut base = 0;
        while base < 128 {
            for j in base..base + d {
                let (a, b) = (v[j], v[j + d]);
                v[j] = a + b;
                v[j + d] = a - b;
            }
            base += 2 * d;
        }
        d *= 2;
    }
}

/// The same transform in f64 (unscaled).
pub fn fwht128_f64(v: &mut [f64]) {
    debug_assert_eq!(v.len(), 128);
    let mut d = 1;
    while d < 128 {
        let mut base = 0;
        while base < 128 {
            for j in base..base + d {
                let (a, b) = (v[j], v[j + d]);
                v[j] = a + b;
                v[j + d] = a - b;
            }
            base += 2 * d;
        }
        d *= 2;
    }
}

/// `exl3_ada_rot_in_bf16`, bit for bit: per 128-value block of each row, `f32(x) * f32(suh)`, `fwht128`,
/// `* HAD_SCALE`, round to fp16. `x` bf16 bits `[rows, k]`, `suh` fp16 bits `[k]`; returns fp16 bits.
pub fn rot_in_mirror(x: &[u16], rows: usize, k: usize, suh: &[u16]) -> Vec<u16> {
    let mut out = vec![0u16; rows * k];
    let mut blk = [0f32; 128];
    for r in 0..rows {
        for b in 0..k / 128 {
            let at = r * k + b * 128;
            for (j, v) in blk.iter_mut().enumerate() {
                *v = bf16_to_f32(x[at + j]) * f16_to_f32(suh[b * 128 + j]);
            }
            fwht128_f32(&mut blk);
            for (j, v) in blk.iter().enumerate() {
                out[at + j] = f32_to_f16(*v * HAD_SCALE);
            }
        }
    }
    out
}

/// The rotation in f64 with no rounding: `((x * suh) @ H) / sqrt(128)` per block (`x` as f64).
pub fn rot_in_f64(x: &[f64], rows: usize, k: usize, suh: &[u16]) -> Vec<f64> {
    let mut out = vec![0f64; rows * k];
    for r in 0..rows {
        for b in 0..k / 128 {
            let at = r * k + b * 128;
            let blk = &mut out[at..at + 128];
            for j in 0..128 {
                blk[j] = x[at + j] * f64::from(f16_to_f32(suh[b * 128 + j]));
            }
            fwht128_f64(blk);
            for v in blk.iter_mut() {
                *v *= had_scale_f64();
            }
        }
    }
    out
}

/// `exl3_ada_dequant_bf16_<K2>_<CB>`, bit for bit: `wq` fp16 bits `[k, n]` (from [`super::reference::unpack`])
/// to bf16 bits `[n, k]`. Per 128 x 128 block: rows along n through `fwht128` then `* HAD_SCALE`; columns
/// along k through `fwht128` then `((v * HAD_SCALE) * suh[k]) * svh[n]`, round to bf16.
pub fn dequant_bf16_mirror(wq: &[u16], k: usize, n: usize, suh: &[u16], svh: &[u16]) -> Vec<u16> {
    let mut out = vec![0u16; n * k];
    let mut s = vec![0f32; 128 * 128];
    let mut v = [0f32; 128];
    for kb in 0..k / 128 {
        for nb in 0..n / 128 {
            for i in 0..128 {
                for j in 0..128 {
                    s[i * 128 + j] = f16_to_f32(wq[(kb * 128 + i) * n + nb * 128 + j]);
                }
            }
            for i in 0..128 {
                v.copy_from_slice(&s[i * 128..i * 128 + 128]);
                fwht128_f32(&mut v);
                for j in 0..128 {
                    s[i * 128 + j] = v[j] * HAD_SCALE;
                }
            }
            for c in 0..128 {
                for i in 0..128 {
                    v[i] = s[i * 128 + c];
                }
                fwht128_f32(&mut v);
                let nn = nb * 128 + c;
                let sv = f16_to_f32(svh[nn]);
                for (i, vi) in v.iter().enumerate() {
                    let kk = kb * 128 + i;
                    out[nn * k + kk] = f32_to_bf16(vi * HAD_SCALE * f16_to_f32(suh[kk]) * sv);
                }
            }
        }
    }
    out
}

/// The full weight `W [k, n]` in f64: `diag(suh) H_K Wq H_N diag(svh)`, H scaled by 1/sqrt(128).
pub fn dequant_f64(wq: &[u16], k: usize, n: usize, suh: &[u16], svh: &[u16]) -> Vec<f64> {
    let mut w: Vec<f64> = wq.iter().map(|&b| f64::from(f16_to_f32(b))).collect();
    let mut col = vec![0f64; 128];
    // H_K: every column along k
    for c in 0..n {
        for kb in 0..k / 128 {
            for i in 0..128 {
                col[i] = w[(kb * 128 + i) * n + c];
            }
            fwht128_f64(&mut col);
            for i in 0..128 {
                w[(kb * 128 + i) * n + c] = col[i] * had_scale_f64();
            }
        }
    }
    // H_N: every row along n, then both scales
    for r in 0..k {
        let su = f64::from(f16_to_f32(suh[r]));
        for nb in 0..n / 128 {
            let row = &mut w[r * n + nb * 128..r * n + nb * 128 + 128];
            fwht128_f64(row);
            for (j, v) in row.iter_mut().enumerate() {
                *v *= had_scale_f64() * su * f64::from(f16_to_f32(svh[nb * 128 + j]));
            }
        }
    }
    w
}

/// `x [rows, k] @ W [k, n]` in f64.
pub fn dense_f64(x: &[f64], rows: usize, w: &[f64], k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0f64; rows * n];
    for r in 0..rows {
        for kk in 0..k {
            let xv = x[r * k + kk];
            if xv == 0.0 {
                continue;
            }
            let wr = &w[kk * n..kk * n + n];
            for (o, wv) in out[r * n..r * n + n].iter_mut().zip(wr) {
                *o += xv * wv;
            }
        }
    }
    out
}

/// The FP64 reference of one output row of the linear kernel and its error bound.
#[derive(Clone, Debug, Default)]
pub struct GemvBound {
    /// `[n]` reference outputs.
    pub want: Vec<f64>,
    /// `[n]` largest admissible `|got - want|`.
    pub bound: Vec<f64>,
}

impl GemvBound {
    /// Outputs (bf16 bits) outside the bound, or non-finite.
    pub fn violations(&self, got_bf16: &[u16]) -> usize {
        got_bf16
            .iter()
            .zip(self.want.iter().zip(&self.bound))
            .filter(|(g, (w, b))| {
                let gv = f64::from(bf16_to_f32(**g));
                !gv.is_finite() || (gv - *w).abs() > **b
            })
            .count()
    }
}

/// One row through the Hadamard-domain linear: `xh` fp16 bits `[k]` (the rotated input, e.g. from
/// [`rot_in_mirror`]), `wq` fp16 bits `[k, n]`, `svh` fp16 bits `[n]`. `want[c] = svh[c] / sqrt(128) *
/// (H yrot)[c]`, `yrot = xh @ Wq` exact in f64.
///
/// The bound covers, per output: the fp32 tensor-core sums of `k / 16` mma steps plus the warp and split
/// sums (charged at `(k/16 + 16) * 2^-22` of the absolute term sum, four units a step, generous for an
/// accumulator that may truncate), carried through the transform with unit coefficients; the 7 fp32
/// butterfly stages (`8 * 2^-24` of the block's absolute sum); the f32 `HAD_SCALE` and the two fp32
/// products; and the final bf16 rounding (`2^-8` relative, twice the half unit).
pub fn gemv_row_ref(xh: &[u16], wq: &[u16], k: usize, n: usize, svh: &[u16]) -> GemvBound {
    let x: Vec<f64> = xh.iter().map(|&b| f64::from(f16_to_f32(b))).collect();
    let mut yrot = vec![0f64; n];
    let mut abs = vec![0f64; n];
    for (kk, &xv) in x.iter().enumerate() {
        if xv == 0.0 {
            continue;
        }
        let row = &wq[kk * n..kk * n + n];
        for c in 0..n {
            let t = xv * f64::from(f16_to_f32(row[c]));
            yrot[c] += t;
            abs[c] += t.abs();
        }
    }
    let acc_unit = (k / 16 + 16) as f64 * 2f64.powi(-22);
    let mut want = vec![0f64; n];
    let mut bound = vec![0f64; n];
    for nb in 0..n / 128 {
        let blk = nb * 128..nb * 128 + 128;
        let mut h = yrot[blk.clone()].to_vec();
        fwht128_f64(&mut h);
        let acc_err: f64 = abs[blk.clone()].iter().sum::<f64>() * acc_unit;
        let y_abs: f64 = yrot[blk.clone()].iter().map(|v| v.abs()).sum();
        let stage_err = 8.0 * 2f64.powi(-24) * y_abs;
        for (j, hv) in h.iter().enumerate() {
            let c = nb * 128 + j;
            let sv = f64::from(f16_to_f32(svh[c]));
            let w = hv * had_scale_f64() * sv;
            want[c] = w;
            bound[c] = had_scale_f64() * sv.abs() * (acc_err + stage_err) * (1.0 + 2f64.powi(-20))
                + w.abs() * (2f64.powi(-8) + 4.0 * 2f64.powi(-23))
                + f64::from(f32::MIN_POSITIVE);
        }
    }
    GemvBound { want, bound }
}
