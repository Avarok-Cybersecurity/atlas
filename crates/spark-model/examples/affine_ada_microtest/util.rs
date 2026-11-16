// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the affine microtest's plumbing: a deterministic generator, uploads and downloads,
//! guarded output buffers, verdict bookkeeping and the timing loop.

use anyhow::{Result, ensure};
use spark_model::port::affine::reference::{bf16_of, f32_of};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::Instant;

/// Sentinel bytes either side of every output.
pub const GUARD: usize = 64;

pub struct Lcg(pub u64);

impl Lcg {
    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// bf16-exact values, roughly normal with standard deviation `sigma`.
    pub fn bf16s(&mut self, len: usize, sigma: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                let x = (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma;
                f32_of(bf16_of(x))
            })
            .collect()
    }
    pub fn below(&mut self, n: usize) -> usize {
        ((self.unit() as f64) * n as f64) as usize % n.max(1)
    }
}

pub fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16_of(*x)).collect()
}

pub fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn u32_bytes(v: &[u32]) -> Vec<u8> {
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
    Ok(download(gpu, ptr, n * 2)?
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

pub fn download_u32(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<u32>> {
    Ok(download(gpu, ptr, n * 4)?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn download_f32(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize) -> Result<Vec<f32>> {
    Ok(download_u32(gpu, ptr, n)?
        .into_iter()
        .map(f32::from_bits)
        .collect())
}

/// A BF16 output buffer with sentinel guards either side.
pub struct Out {
    base: DevicePtr,
    pub elems: usize,
}

impl Out {
    pub fn new(gpu: &dyn GpuBackend, elems: usize) -> Result<Self> {
        let o = Self {
            base: gpu.alloc(elems * 2 + 2 * GUARD)?,
            elems,
        };
        o.reset(gpu)?;
        Ok(o)
    }
    pub fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }
    pub fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&vec![0x5a; self.elems * 2 + 2 * GUARD], self.base)
    }
    /// The payload as BF16 bits, after checking both guards.
    pub fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u16>> {
        let host = download(gpu, self.base, self.elems * 2 + 2 * GUARD)?;
        let guards = host[..GUARD].iter().chain(&host[GUARD + self.elems * 2..]);
        ensure!(
            guards.into_iter().all(|b| *b == 0x5a),
            "a kernel wrote outside its output"
        );
        Ok(host[GUARD..GUARD + self.elems * 2]
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect())
    }
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.base)
    }
}

pub fn unequal<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

/// Gate bookkeeping: every gate prints one line; a failed gate or a passing control fails the run.
#[derive(Default)]
pub struct Tally {
    pub fails: usize,
    pub gates: usize,
}

impl Tally {
    /// A gate that must pass: `bad == 0`.
    pub fn gate(&mut self, name: &str, bad: usize, detail: &str) {
        self.gates += 1;
        self.fails += usize::from(bad != 0);
        let v = if bad == 0 { "PASS" } else { "FAIL" };
        println!("  [{v}] {name}: bad={bad} {detail}");
    }
    /// A planted control that must FAIL its gate: `bad > 0`. A control that passes is a failed run.
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
}

pub const REPS: usize = 20;
pub const WARMUP: usize = 3;

/// Mean microseconds of `run(i)` over `REPS` calls after `WARMUP`, `i` cycling over the copies the
/// closure owns (each call should touch a distinct weight copy so the L2 never holds it).
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
