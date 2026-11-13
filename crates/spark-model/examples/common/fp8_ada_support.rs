// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: helpers of `fp8_ada_microtest`, split out so that example
//! stays under the 500-line cap once it times cold. The first half is a pure
//! move (the generator, upload and download, the guarded output buffer, the
//! BF16 comparison helpers); the second half is the cold-ring plumbing the
//! retrofit adds.
//!
//! An example that includes this file also declares the `cold_l2` and
//! `profiler` modules.
#![allow(dead_code)]

use crate::cold_l2::{
    Arm, BenchCfg, DEFAULT_RING_CAP, Ring, RingPlan, bench_arms, build_ring, plan_ring,
};
use anyhow::{Result, ensure};
use spark_model::weight_map::fp8_blockscale_ref as fref;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// BF16's machine epsilon, 2^-7.
pub const BF16_EPS: f32 = 1.0 / 128.0;
/// Accumulation error allowed per unit of term magnitude (see `acc_err`):
/// 2^-13. The FP8 `mma.sync` sums its products below FP32 precision; on an
/// RTX 4060 Ti the worst output at these shapes is 2^-15 of its term
/// magnitude, with the Ada and gb10 kernels bit-identical, so the margin is
/// 4x and the error is the instruction's, not the tiling's.
pub const ACC_TOL: f32 = 1.0 / 8192.0;
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

    /// BF16 values, roughly normal with standard deviation `sigma`.
    pub fn bf16s(&mut self, len: usize, sigma: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                let x = (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma;
                fref::round_bf16(x)
            })
            .collect()
    }
}

pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, len: usize) -> Result<Vec<u8>> {
    let mut host = vec![0u8; len];
    gpu.copy_d2h(ptr, &mut host)?;
    Ok(host)
}

/// A BF16 output buffer with sentinel guards either side.
pub struct Out {
    base: DevicePtr,
    bytes: usize,
}

impl Out {
    pub fn new(gpu: &dyn GpuBackend, elems: usize) -> Result<Self> {
        let o = Self {
            base: gpu.alloc(elems * 2 + 2 * GUARD)?,
            bytes: elems * 2,
        };
        o.reset(gpu)?;
        Ok(o)
    }

    pub fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }

    pub fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&vec![0x5a; self.bytes + 2 * GUARD], self.base)
    }

    /// The payload as BF16 bits, after checking both guards.
    pub fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u16>> {
        gpu.synchronize(0)?;
        let host = download(gpu, self.base, self.bytes + 2 * GUARD)?;
        let guards = host[..GUARD].iter().chain(&host[GUARD + self.bytes..]);
        ensure!(
            guards.into_iter().all(|b| *b == 0x5a),
            "a kernel wrote outside its output"
        );
        Ok(host[GUARD..GUARD + self.bytes]
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect())
    }
}

pub fn unequal(a: &[u16], b: &[u16]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

pub fn host_bits(v: &[f32]) -> Vec<u16> {
    v.iter()
        .map(|x| half::bf16::from_f32(*x).to_bits())
        .collect()
}

/// A GEMM output's accumulation error: its distance from the reference after
/// allowing 2 BF16 ulps for the final round, as a fraction of the output's
/// term magnitude (`gemm_w8a8_term_mag`). The error of a long dot product
/// scales with the size of its terms, not of its result, so a bound relative
/// to the result alone fails every output whose terms cancel.
pub fn acc_err(got: u16, want: f32, mag: f32) -> f32 {
    let g = half::bf16::from_bits(got).to_f32();
    let round = 2.0 * BF16_EPS * want.abs();
    ((g - want).abs() - round).max(0.0) / mag.max(f32::MIN_POSITIVE)
}

pub fn verdict(fails: usize) -> &'static str {
    if fails == 0 { "PASS" } else { "FAIL" }
}

// ---- cold-ring plumbing (the retrofit) ----

/// A byte-identical copy of a device buffer.
pub fn clone_buf(gpu: &dyn GpuBackend, src: DevicePtr, bytes: usize) -> Result<DevicePtr> {
    let dst = gpu.alloc(bytes)?;
    gpu.copy_d2d(src, dst, bytes)?;
    Ok(dst)
}

/// A cold ring (distinct copies, more than twice the L2) and the one-copy hot
/// ring over its first element, for the cold-against-hot contrast.
pub struct Rings<T: Copy> {
    pub cold: Ring<T>,
    pub hot: Ring<T>,
    pub plan: RingPlan,
}

/// `make(c)` builds copy `c`; copy 0 should be the original, so a result read
/// back afterwards describes the verified weights.
pub fn make_rings<T: Copy>(
    l2: usize,
    per_copy_bytes: usize,
    make: impl FnMut(usize) -> Result<T>,
) -> Result<Rings<T>> {
    let plan = plan_ring(per_copy_bytes, l2, DEFAULT_RING_CAP)?;
    let cold = build_ring(&plan, make)?;
    let hot = Ring::new(vec![cold.all()[0]]);
    Ok(Rings { cold, hot, plan })
}

/// Median microseconds of the two kernels (index 0 = gb10, 1 = ada), timed
/// interleaved in rotated order over `ring`. `launch(i, copy)` enqueues
/// kernel `i` on one copy of the weights.
pub fn pair_us<T: Copy>(
    gpu: &dyn GpuBackend,
    cfg: &BenchCfg,
    ring: &Ring<T>,
    launch: &dyn Fn(usize, T) -> Result<()>,
) -> Result<[f64; 2]> {
    let mut arms = vec![
        Arm::new("gb10", || launch(0, ring.pick())),
        Arm::new("ada", || launch(1, ring.pick())),
    ];
    let r = bench_arms(gpu, &mut arms, cfg)?;
    Ok([r[0].median_us, r[1].median_us])
}
