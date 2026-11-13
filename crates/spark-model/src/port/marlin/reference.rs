// SPDX-License-Identifier: AGPL-3.0-only
// The dequantization rule mirrors vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// csrc/quantization/marlin/dequant.h (kU4B8: `(128 + q) - 136`; kU4 with zero points:
// `(128 + q) - (128 + z)`, both exact in BF16) and marlin_template.h:102-112 (`scale`: one `__hmul2`,
// a single round-to-nearest-even of the exact product). Copyright contributors to the vLLM project,
// Apache-2.0; Copyright (C) Marlin.2024 Elias Frantar, Apache-2.0. Changes: restated as a CPU model in
// Rust; the GEMM reference and its error bound are Atlas's.

//! Responsibility: the logical int4 weight ([`LogicalInt4`]) every Marlin source is parsed into, the
//! BF16 helpers, the CPU dequantization that is bit-exact to the kernel's, and the CPU GEMM oracle
//! with its error bound ("2 BF16 ulps plus a term-magnitude bound").
//!
//! Bit-exactness of the dequant: `q - z` is an integer in -15..=15, exact in BF16, and the kernel
//! multiplies it by the BF16 scale with one rounding; a 4-bit integer times an 8-bit significand is
//! exact in `f32`, so `bf16_rne(f32(q - z) * f32(s))` is the kernel's value, signed zeros included.

use anyhow::{Result, ensure};

use super::MarlinType;

/// The BF16 bits of `x`, rounded to nearest even (NaN stays NaN).
pub fn f32_to_bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

/// The `f32` value of BF16 bits.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// FP16 bits to BF16 bits (round to nearest even). GPTQ and AWQ checkpoints store FP16 scales; the
/// BF16-activation kernel reads BF16 scales, which is what vLLM does for a BF16 model too.
pub fn f16_to_bf16(h: u16) -> u16 {
    f32_to_bf16(half::f16::from_bits(h).to_f32())
}

/// One dequantized weight: `bf16_rne((q - z) * s)`.
pub fn dequant(q: u8, z: u8, s: u16) -> u16 {
    let d = i32::from(q) - i32::from(z);
    f32_to_bf16(d as f32 * bf16_to_f32(s))
}

/// The spacing of BF16 values at `|x|` (the subnormal spacing below the normal range).
pub fn ulp_bf16(x: f64) -> f64 {
    let a = x.abs();
    let min_normal = f64::from(f32::MIN_POSITIVE);
    if a < min_normal {
        return min_normal * 2f64.powi(-7);
    }
    2f64.powi(a.log2().floor() as i32 - 7)
}

/// A logical int4 weight in Marlin's B orientation.
///
/// `q[kk * n + nn]` is the code of input `kk`, output `nn` (`B = W^T`, shape `[k, n]`); codes are
/// 0..=15. `scales[g * n + nn]` is the BF16 scale of group `g = kk / group`. `zeros` (same indexing
/// as `scales`, values 0..=15) is `None` for the `kU4B8` type (implicit zero 8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalInt4 {
    pub n: usize,
    pub k: usize,
    pub group: usize,
    pub q: Vec<u8>,
    pub scales: Vec<u16>,
    pub zeros: Option<Vec<u8>>,
}

impl LogicalInt4 {
    /// Groups along K.
    pub fn groups(&self) -> usize {
        self.k / self.group
    }

    /// The kernel weight type this weight needs.
    pub fn btype(&self) -> MarlinType {
        if self.zeros.is_some() {
            MarlinType::U4
        } else {
            MarlinType::U4B8
        }
    }

    /// The zero point of group `g`, output `nn`.
    pub fn zero(&self, g: usize, nn: usize) -> u8 {
        match &self.zeros {
            Some(z) => z[g * self.n + nn],
            None => 8,
        }
    }

    /// Structural checks: sizes, code range, finite scales.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.n > 0 && self.k > 0 && self.group > 0,
            "marlin: empty weight [{}x{}] group {}",
            self.n,
            self.k,
            self.group
        );
        ensure!(
            self.k.is_multiple_of(self.group),
            "marlin: K={} is not a multiple of the group {}",
            self.k,
            self.group
        );
        ensure!(
            self.q.len() == self.k * self.n,
            "marlin: {} codes for a [{}x{}] weight",
            self.q.len(),
            self.n,
            self.k
        );
        let gn = self.groups() * self.n;
        ensure!(
            self.scales.len() == gn,
            "marlin: {} scales, {gn} expected",
            self.scales.len()
        );
        if let Some(z) = &self.zeros {
            ensure!(
                z.len() == gn,
                "marlin: {} zero points, {gn} expected",
                z.len()
            );
            ensure!(
                z.iter().all(|&v| v < 16),
                "marlin: a zero point is not 4-bit"
            );
        }
        ensure!(
            self.q.iter().all(|&v| v < 16),
            "marlin: a code is not 4-bit"
        );
        if let Some(i) = self
            .scales
            .iter()
            .position(|&s| !bf16_to_f32(s).is_finite())
        {
            anyhow::bail!(
                "marlin: scale {i} (group {}, output {}) is not finite",
                i / self.n,
                i % self.n
            );
        }
        Ok(())
    }

    /// The dequantized weight as BF16 `[n, k]` row-major (the `DenseWeight` layout, the
    /// `dequant_bf16` contract), bit-exact to the kernel's B operand.
    pub fn dequant_nk(&self) -> Vec<u16> {
        let (n, k) = (self.n, self.k);
        let mut out = vec![0u16; n * k];
        for kk in 0..k {
            let g = kk / self.group;
            for nn in 0..n {
                let s = self.scales[g * n + nn];
                out[nn * k + kk] = dequant(self.q[kk * n + nn], self.zero(g, nn), s);
            }
        }
        out
    }
}

/// The reference product of `x` (`[m, k]` BF16) and a dequantized weight `w_nk` (`[n, k]` BF16):
/// `(value, magnitude)` per output, `value = sum x*w` and `magnitude = sum |x*w|`, both in `f64`.
pub fn reference_gemm(
    x: &[u16],
    m: usize,
    w_nk: &[u16],
    n: usize,
    k: usize,
) -> (Vec<f64>, Vec<f64>) {
    let xf: Vec<f64> = x.iter().map(|&v| f64::from(bf16_to_f32(v))).collect();
    let wf: Vec<f64> = w_nk.iter().map(|&v| f64::from(bf16_to_f32(v))).collect();
    let mut val = vec![0f64; m * n];
    let mut mag = vec![0f64; m * n];
    for r in 0..m {
        let xr = &xf[r * k..(r + 1) * k];
        for c in 0..n {
            let wc = &wf[c * k..(c + 1) * k];
            let (mut s, mut a) = (0f64, 0f64);
            for (xv, wv) in xr.iter().zip(wc) {
                let p = xv * wv;
                s += p;
                a += p.abs();
            }
            val[r * n + c] = s;
            mag[r * n + c] = a;
        }
    }
    (val, mag)
}

/// The tolerance of one output: two BF16 ulps of the reference (the final rounding plus slack) and
/// the FP32 accumulation bound `k * 2^-24 * sum |x*w|` (every BF16 product is exact in FP32).
pub fn tolerance(value: f64, magnitude: f64, k: usize) -> f64 {
    2.0 * ulp_bf16(value) + k as f64 * 2f64.powi(-24) * magnitude
}

/// The outcome of comparing a kernel output against [`reference_gemm`].
#[derive(Clone, Debug, PartialEq)]
pub struct OracleReport {
    /// Outputs outside the tolerance (non-finite outputs count).
    pub failures: usize,
    /// The largest `|got - ref| / tolerance` seen.
    pub worst_ratio: f64,
    /// `(row, col, got, ref, tol)` of the first failure.
    pub first: Option<(usize, usize, f32, f64, f64)>,
}

impl OracleReport {
    pub fn ok(&self) -> bool {
        self.failures == 0
    }
}

/// Compare a BF16 `[m, n]` output against the reference product of `x` and `w_nk`.
pub fn check_output(
    got: &[u16],
    x: &[u16],
    m: usize,
    w_nk: &[u16],
    n: usize,
    k: usize,
) -> OracleReport {
    let (val, mag) = reference_gemm(x, m, w_nk, n, k);
    let mut rep = OracleReport {
        failures: 0,
        worst_ratio: 0.0,
        first: None,
    };
    for (i, (&g, (&v, &a))) in got.iter().zip(val.iter().zip(&mag)).enumerate() {
        let gf = bf16_to_f32(g);
        let tol = tolerance(v, a, k);
        let err = (f64::from(gf) - v).abs();
        let ratio = if gf.is_finite() {
            err / tol
        } else {
            f64::INFINITY
        };
        if ratio > rep.worst_ratio {
            rep.worst_ratio = ratio;
        }
        if ratio > 1.0 {
            rep.failures += 1;
            if rep.first.is_none() {
                rep.first = Some((i / n, i % n, gf, v, tol));
            }
        }
    }
    rep
}
