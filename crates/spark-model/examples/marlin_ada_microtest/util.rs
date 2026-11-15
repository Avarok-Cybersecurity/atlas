// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the Marlin microtest's plumbing: a deterministic generator, uploads and
//! downloads, the sampled-column CPU oracle (the "2 BF16 ulps plus a term-magnitude bound" of
//! `port::marlin::reference`), PASS/FAIL bookkeeping in the form the GPU measurement scripts (not
//! part of this repository) parse, and a synchronized timing loop.

use anyhow::Result;
use spark_model::port::marlin::reference::{bf16_to_f32, f32_to_bf16, tolerance};
use spark_model::port::w4::W4Scratch;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::Instant;

/// A 64-bit LCG (Knuth's constants); deterministic across runs.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f32 {
        ((self.next() >> 29) as f32) / ((1u64 << 24) as f32)
    }

    /// BF16 bits of roughly normal values with standard deviation `sigma`.
    pub fn bf16s(&mut self, len: usize, sigma: f32) -> Vec<u16> {
        (0..len)
            .map(|_| {
                let x = (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma;
                f32_to_bf16(x)
            })
            .collect()
    }

    /// A 4-bit code.
    pub fn code(&mut self) -> u8 {
        (self.next() % 16) as u8
    }
}

pub fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub fn upload_u16(gpu: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    upload(gpu, &u16_bytes(v))
}

pub fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, len: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut host = vec![0u8; len];
    gpu.copy_d2h(ptr, &mut host)?;
    Ok(host)
}

pub fn download_u16(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u16>> {
    Ok(download(gpu, ptr, n * 2)?
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// A scratch whose `ws` holds `bytes` (the FP32 reduce buffer); the other regions are unused by
/// Marlin.
pub fn scratch(gpu: &dyn GpuBackend, bytes: usize) -> Result<W4Scratch> {
    Ok(W4Scratch {
        ws: gpu.alloc(bytes)?,
        ws_bytes: bytes,
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    })
}

/// The worst `|got - ref| / tol` over rows `0..m` and the columns `0, stride, 2 stride, ...` plus
/// the last column, and the number of outputs outside the bound (non-finite outputs count).
/// `w_nk` is the dequantized weight (`[n, k]` BF16), bit-exact to the kernel's B operand.
pub fn oracle(
    got: &[u16],
    x: &[u16],
    m: usize,
    w_nk: &[u16],
    n: usize,
    k: usize,
    stride: usize,
) -> (f64, usize) {
    let mut cols: Vec<usize> = (0..n).step_by(stride.max(1)).collect();
    if cols.last() != Some(&(n - 1)) {
        cols.push(n - 1);
    }
    let mut worst = 0f64;
    let mut bad = 0usize;
    for r in 0..m {
        let xr = &x[r * k..(r + 1) * k];
        for &c in &cols {
            let wc = &w_nk[c * k..(c + 1) * k];
            let (mut s, mut a) = (0f64, 0f64);
            for (&xv, &wv) in xr.iter().zip(wc) {
                let p = f64::from(bf16_to_f32(xv)) * f64::from(bf16_to_f32(wv));
                s += p;
                a += p.abs();
            }
            let g = bf16_to_f32(got[r * n + c]);
            let ratio = if g.is_finite() {
                (f64::from(g) - s).abs() / tolerance(s, a, k)
            } else {
                f64::INFINITY
            };
            worst = worst.max(ratio);
            if ratio > 1.0 {
                bad += 1;
            }
        }
    }
    (worst, bad)
}

/// Gate bookkeeping: every line starts with PASS, FAIL, CONTROL-OK or info.
#[derive(Default)]
pub struct Tally {
    pub gates: usize,
    pub fails: usize,
}

impl Tally {
    pub fn gate(&mut self, ok: bool, what: &str) {
        self.gates += 1;
        if ok {
            println!("PASS {what}");
        } else {
            self.fails += 1;
            println!("FAIL {what}");
        }
    }

    /// A planted control: it must FAIL its gate; a control that passes fails the run.
    pub fn control(&mut self, gate_failed: bool, what: &str) {
        self.gates += 1;
        if gate_failed {
            println!("CONTROL-OK {what}");
        } else {
            self.fails += 1;
            println!("FAIL control passed its gate: {what}");
        }
    }
}

/// Microseconds per call of `f` over `iters` calls after `warm` warm-up calls, the stream
/// synchronized before and after the timed loop.
pub fn time_us(
    gpu: &dyn GpuBackend,
    warm: usize,
    iters: usize,
    mut f: impl FnMut() -> Result<()>,
) -> Result<f64> {
    for _ in 0..warm {
        f()?;
    }
    gpu.synchronize(0)?;
    let t = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    gpu.synchronize(0)?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / iters.max(1) as f64)
}
