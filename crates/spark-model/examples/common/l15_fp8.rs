// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: helpers shared by the L15 Ada FP8 microtests
//! (`fp8_ada_batchm_microtest`, `fp8_ada_l2hint_microtest`,
//! `rmsnorm_quant_fp8_microtest`): seeded inputs, guarded device buffers, a
//! byte comparator, warm and cold-L2 timers, and the raw launches of the
//! kernels whose host wrappers are crate-private.
//!
//! Included with `#[path = "common/l15_fp8.rs"] mod l15;`, so it is compiled
//! into each example on its own; every example uses only part of it.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

pub const GUARD: usize = 64;
pub const SENTINEL: u8 = 0x5a;
pub const REPS: u32 = 20;
pub const WARMUP: u32 = 3;
/// Bytes rewritten between cold-L2 reps: twice the RTX 4060 Ti's 32 MB L2.
pub const L2_FLUSH_BYTES: usize = 64 << 20;

/// The seeded generator the existing FP8 microtests use.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// Roughly Gaussian BF16 values (sum of four uniforms), widened to f32.
    pub fn bf16s(&mut self, len: usize, sigma: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                let x = (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma;
                half::bf16::from_f32(x).to_f32()
            })
            .collect()
    }
    /// Finite E4M3 bytes: every code except the two NaNs (0x7F, 0xFF), which
    /// no Atlas quantizer emits.
    pub fn e4m3_bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len)
            .map(|_| {
                loop {
                    let b = (self.next_u32() & 0xFF) as u8;
                    if (b & 0x7F) != 0x7F {
                        break b;
                    }
                }
            })
            .collect()
    }
    /// Positive block scales, `ceil(n/128) * ceil(k/128)` of them.
    pub fn block_scales(&mut self, n: usize, k: usize) -> Vec<f32> {
        (0..n.div_ceil(128) * k.div_ceil(128))
            .map(|_| 1e-3 + 2e-2 * self.unit())
            .collect()
    }
}

pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect()
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn bf16_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub fn unequal_bytes(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
}

pub fn verdict(fails: usize) -> &'static str {
    if fails == 0 { "PASS" } else { "FAIL" }
}

/// A device buffer with sentinel guards either side.
pub struct Guarded {
    base: DevicePtr,
    pub bytes: usize,
}

impl Guarded {
    pub fn new(gpu: &dyn GpuBackend, bytes: usize) -> Result<Self> {
        let base = gpu.alloc(bytes + 2 * GUARD)?;
        gpu.copy_h2d(&vec![SENTINEL; bytes + 2 * GUARD], base)?;
        Ok(Self { base, bytes })
    }
    /// A guarded buffer holding `payload`.
    pub fn with(gpu: &dyn GpuBackend, payload: &[u8]) -> Result<Self> {
        let g = Self::new(gpu, payload.len())?;
        gpu.copy_h2d(payload, g.ptr())?;
        Ok(g)
    }
    pub fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }
    /// The payload, after checking both guards.
    pub fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        gpu.synchronize(0)?;
        let mut host = vec![0u8; self.bytes + 2 * GUARD];
        gpu.copy_d2h(self.base, &mut host)?;
        let guards = host[..GUARD].iter().chain(&host[GUARD + self.bytes..]);
        ensure!(
            guards.into_iter().all(|b| *b == SENTINEL),
            "a kernel wrote outside its output"
        );
        Ok(host[GUARD..GUARD + self.bytes].to_vec())
    }
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.base)
    }
}

/// Mean microseconds of `run` over [`REPS`] back-to-back launches after
/// [`WARMUP`] (hot L2: weights the previous rep left resident count).
pub fn time_warm_us(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS))
}

/// Median microseconds of `run` with [`L2_FLUSH_BYTES`] rewritten and the
/// stream drained before each rep (cold L2, DRAM-true). Includes one sync's
/// host latency, which is the same for both sides of any A/B here.
pub fn time_cold_us(
    gpu: &dyn GpuBackend,
    flush: DevicePtr,
    mut run: impl FnMut() -> Result<()>,
) -> Result<f64> {
    let mut samples = Vec::with_capacity(REPS as usize);
    for rep in 0..WARMUP + REPS {
        gpu.memset_async(flush, (rep & 0xFF) as u8, L2_FLUSH_BYTES, 0)?;
        gpu.synchronize(0)?;
        let t0 = Instant::now();
        run()?;
        gpu.synchronize(0)?;
        if rep >= WARMUP {
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

/// GB/s for `bytes` moved in `us`.
pub fn gbs(bytes: f64, us: f64) -> f64 {
    bytes / (us * 1e-6) / 1e9
}

// ── Raw launches (the host wrappers live in crate-private modules) ──────

/// `fp8_gemv_ada{,_l2ef}` / `fp8_gemv_ada_batchm{,_l2ef}_<M>`: A, B, scale, C, N, K.
#[allow(clippy::too_many_arguments)]
pub fn gemv(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    s: DevicePtr,
    c: DevicePtr,
    n: u32,
    kk: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(s)
        .arg_ptr(c)
        .arg_u32(n)
        .arg_u32(kk)
        .launch(0)
}

/// The `_dual` entries: A, B1, S1, C1, B2, S2, C2, N, K; grid z = 2.
#[allow(clippy::too_many_arguments)]
pub fn gemv_dual(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    w1: DevicePtr,
    s1: DevicePtr,
    c1: DevicePtr,
    w2: DevicePtr,
    s2: DevicePtr,
    c2: DevicePtr,
    n: u32,
    kk: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w1)
        .arg_ptr(s1)
        .arg_ptr(c1)
        .arg_ptr(w2)
        .arg_ptr(s2)
        .arg_ptr(c2)
        .arg_u32(n)
        .arg_u32(kk)
        .launch(0)
}

/// The `_silu_input` entries: gate, up, B, S, C, N, K.
#[allow(clippy::too_many_arguments)]
pub fn gemv_silu(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    g: DevicePtr,
    u: DevicePtr,
    w: DevicePtr,
    s: DevicePtr,
    c: DevicePtr,
    n: u32,
    kk: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(g)
        .arg_ptr(u)
        .arg_ptr(w)
        .arg_ptr(s)
        .arg_ptr(c)
        .arg_u32(n)
        .arg_u32(kk)
        .launch(0)
}

/// The batch-M entry names, `[plain, dual, silu_input]`, as
/// `layers/dense_ffn_ada_kernels/verify.rs` tables them.
pub fn batchm_names(m: usize, l2ef: bool) -> [String; 3] {
    let tag = if l2ef { "l2ef_" } else { "" };
    [
        format!("fp8_gemv_ada_batchm_{tag}{m}"),
        format!("fp8_gemv_ada_batchm_{tag}dual_{m}"),
        format!("fp8_gemv_ada_batchm_{tag}silu_input_{m}"),
    ]
}
