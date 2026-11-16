// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the keep-packed GGUF microtest's plumbing (port lane L19): a deterministic generator,
//! random and planted packed weights, uploads and downloads, a threaded float64 reference GEMV, the
//! grading with an extra per-output slack, verdict bookkeeping and the timing loop.

use std::time::Instant;

use anyhow::Result;
use spark_model::port::gguf_q4::Q4Type;
use spark_model::port::gguf_q4::reference::{bf16_to_f32, gemv_f64};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// xorshift64*: deterministic bytes and values.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
    /// `len` bf16 bit patterns of values uniform in [-1, 1).
    pub fn bf16_bits(&mut self, len: usize) -> Vec<u16> {
        (0..len).map(|_| bf16(self.unit())).collect()
    }
}

pub fn bf16(v: f32) -> u16 {
    half::bf16::from_f32(v).to_bits()
}

pub fn f16_bits(v: f32) -> [u8; 2] {
    half::f16::from_f32(v).to_bits().to_le_bytes()
}

/// Random packed bytes of `[n, k]` with small, finite, positive f16 scales (the code bytes are random).
pub fn random_weight(rng: &mut Rng, ty: Q4Type, n: usize, k: usize) -> Vec<u8> {
    let bb = ty.block_bytes();
    let mut v = Vec::with_capacity(n * (k / 256) * bb);
    for _ in 0..n * (k / 256) {
        let start = v.len();
        v.extend((0..bb).map(|_| rng.byte()));
        let b = &mut v[start..];
        let scale = f16_bits(0.001 + 0.01 * rng.unit().abs());
        match ty {
            Q4Type::Iq4Xs => b[0..2].copy_from_slice(&scale),
            Q4Type::Q4K => {
                b[0..2].copy_from_slice(&scale);
                let dmin = f16_bits(0.001 + 0.004 * rng.unit().abs());
                b[2..4].copy_from_slice(&dmin);
            }
            Q4Type::Q6K => b[208..210].copy_from_slice(&scale),
        }
    }
    v
}

/// Random raw blocks of `bb` bytes whose first `f16_fields` halves are finite scales (the dequant gate's
/// input for IQ4_XS, IQ4_NL and Q5_K).
pub fn random_blocks(rng: &mut Rng, n_blocks: usize, bb: usize, f16_fields: usize) -> Vec<u8> {
    let mut v: Vec<u8> = (0..n_blocks * bb).map(|_| rng.byte()).collect();
    for blk in v.chunks_exact_mut(bb) {
        for f in 0..f16_fields {
            let s = f16_bits(rng.unit() * 0.05);
            blk[2 * f..2 * f + 2].copy_from_slice(&s);
        }
    }
    v
}

/// Planted fault: the last super-block of every row zeroed (all three types then contribute nothing for
/// it: IQ4_XS and Q6_K have `d = 0`, Q4_K has `d = dmin = 0`).
pub fn drop_last_block(raw: &[u8], ty: Q4Type, n: usize, k: usize) -> Vec<u8> {
    let bb = ty.block_bytes();
    let per_row = k / 256;
    let mut v = raw.to_vec();
    for row in 0..n {
        let at = (row * per_row + per_row - 1) * bb;
        v[at..at + bb].fill(0);
    }
    v
}

pub fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, len: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut host = vec![0u8; len];
    gpu.copy_d2h(ptr, &mut host)?;
    Ok(host)
}

pub fn download_u16(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u16>> {
    Ok(download(gpu, ptr, 2 * n)?
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

pub fn widen(v: &[u16]) -> Vec<f32> {
    v.iter().map(|&b| bf16_to_f32(b)).collect()
}

/// [`gemv_f64`] over `threads` slices of the output rows: `(want, mag)` laid out `[m][n]`.
pub fn gemv_par(
    w: &[f32],
    n: usize,
    k: usize,
    x: &[f32],
    m: usize,
    threads: usize,
) -> (Vec<f64>, Vec<f64>) {
    let chunk = n.div_ceil(threads.max(1)).max(1);
    let parts: Vec<(usize, usize, Vec<f64>, Vec<f64>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|j0| {
                let j1 = (j0 + chunk).min(n);
                let ws = &w[j0 * k..j1 * k];
                s.spawn(move || {
                    let (want, mag) = gemv_f64(ws, j1 - j0, k, x, m);
                    (j0, j1, want, mag)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("oracle thread"))
            .collect()
    });
    let mut want = vec![0f64; m * n];
    let mut mag = vec![0f64; m * n];
    for (j0, j1, pw, pm) in parts {
        let width = j1 - j0;
        for r in 0..m {
            want[r * n + j0..r * n + j1].copy_from_slice(&pw[r * width..(r + 1) * width]);
            mag[r * n + j0..r * n + j1].copy_from_slice(&pm[r * width..(r + 1) * width]);
        }
    }
    (want, mag)
}

/// `(over, worst)`: outputs beyond `bound_fn(want, mag, k) + slack` and the worst error in bound units.
pub fn grade_slack(
    got: &[f32],
    want: &[f64],
    mag: &[f64],
    slack: Option<&[f64]>,
    k: usize,
    bound_fn: fn(f64, f64, usize) -> f64,
) -> (usize, f64) {
    let mut over = 0usize;
    let mut worst = 0f64;
    for (i, g) in got.iter().enumerate() {
        let b = bound_fn(want[i], mag[i], k) + slack.map_or(0.0, |s| s[i]);
        let e = (f64::from(*g) - want[i]).abs() / b;
        if !g.is_finite() || e > 1.0 {
            over += 1;
        }
        if e.is_finite() {
            worst = worst.max(e);
        }
    }
    (over, worst)
}

pub fn unequal<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// Gate and control bookkeeping: any FAIL, or a control that passes its gate, fails the run.
#[derive(Default)]
pub struct Tally {
    pub gates: usize,
    pub fails: usize,
}

impl Tally {
    pub fn gate(&mut self, name: &str, bad: usize, detail: &str) {
        self.gates += 1;
        self.fails += usize::from(bad > 0);
        let v = if bad == 0 { "PASS" } else { "FAIL" };
        println!("  [{v}] {name}: bad={bad} {detail}");
    }
    pub fn control(&mut self, name: &str, bad: usize, detail: &str) {
        self.gates += 1;
        self.fails += usize::from(bad == 0);
        let v = if bad > 0 {
            "CONTROL-OK"
        } else {
            "CONTROL-PASSED(FAIL)"
        };
        println!("  [{v}] {name}: bad={bad} {detail}");
    }
    pub fn info(&self, name: &str, detail: &str) {
        println!("  [INFO] {name}: {detail}");
    }
}

pub const REPS: usize = 20;
pub const WARMUP: usize = 3;

/// Mean microseconds of `run(i)` over `REPS` calls after `WARMUP`; `i` cycles over the caller's weight
/// copies so the L2 never holds the matrix being timed.
pub fn time_us(gpu: &dyn GpuBackend, mut run: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    for i in 0..WARMUP {
        run(i)?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for i in 0..REPS {
        run(WARMUP + i)?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64)
}
