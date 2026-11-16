// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: helpers of `affine_prefill_microtest` (port lane L17b), split out so that example
//! stays under the 500-line cap: upload and download, the guarded output buffer, a device copy of
//! a host affine weight, the int8 scratch, the sampled rows and one prompt GEMM call.
#![allow(dead_code)]

use anyhow::{Result, ensure};
use spark_model::port::affine::prefill::oracle::{self, HostAffine, Lcg};
use spark_model::port::affine::prefill::{self, Packed};
use spark_model::port::w4::W4Scratch;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::Instant;

/// The largest prompt the scratch serves (the 3,921-token TTFT prompt).
pub const MAX_M: usize = 3921;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;

pub fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| oracle::bf16_bits(*x).to_le_bytes())
        .collect()
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(16))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

pub fn read(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    gpu.synchronize(0)?;
    let mut h = vec![0u8; bytes];
    gpu.copy_d2h(p, &mut h)?;
    Ok(h)
}

pub fn u16s(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

pub fn verdict(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

/// An output buffer with sentinel guards either side; `read` checks them.
pub struct Guarded {
    pub base: DevicePtr,
    bytes: usize,
}

impl Guarded {
    pub fn new(gpu: &dyn GpuBackend, bytes: usize) -> Result<Self> {
        let base = gpu.alloc(bytes + 2 * GUARD)?;
        gpu.copy_h2d(&vec![SENTINEL; bytes + 2 * GUARD], base)?;
        Ok(Self { base, bytes })
    }

    pub fn ptr(&self) -> DevicePtr {
        self.base.offset(GUARD)
    }

    pub fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let h = read(gpu, self.base, self.bytes + 2 * GUARD)?;
        let guards = h[..GUARD].iter().chain(&h[GUARD + self.bytes..]);
        ensure!(
            guards.into_iter().all(|b| *b == SENTINEL),
            "a kernel wrote outside its output"
        );
        Ok(h[GUARD..GUARD + self.bytes].to_vec())
    }
}

/// One weight on the device and on the host.
pub struct Weight {
    pub host: HostAffine,
    pub dev: Packed,
}

impl Weight {
    pub fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [self.dev.words, self.dev.scales, self.dev.biases] {
            gpu.free(p)?;
        }
        Ok(())
    }
}

/// A random weight in the tiled layout, uploaded.
pub fn weight(gpu: &dyn GpuBackend, rng: &mut Lcg, n: usize, k: usize) -> Result<Weight> {
    let host = HostAffine::random(rng, n, k, 0.02);
    let words: Vec<u8> = host
        .tiled_words()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let dev = Packed {
        words: upload(gpu, &words)?,
        scales: upload(gpu, &u16_bytes(&host.tiled_scales()))?,
        biases: upload(gpu, &u16_bytes(&host.tiled_biases()))?,
        n,
        k,
        npad: host.npad(),
    };
    Ok(Weight { host, dev })
}

/// An int8 scratch for [`MAX_M`] rows of `k`, zeroed, so the bytes an exceptional group leaves
/// unwritten compare equal between two scratches.
pub fn scratch(gpu: &dyn GpuBackend, k: usize) -> Result<W4Scratch> {
    let (xq_n, xs_n) = (prefill::xq_bytes(MAX_M, k), prefill::xs_bytes(MAX_M, k));
    let s = W4Scratch {
        ws: gpu.alloc(256)?,
        ws_bytes: 256,
        tickets: DevicePtr::NULL,
        xq: gpu.alloc(xq_n)?,
        xs: gpu.alloc(xs_n)?,
    };
    gpu.memset(s.xq, 0, xq_n)?;
    gpu.memset(s.xs, 0, xs_n)?;
    Ok(s)
}

pub fn free_scratch(gpu: &dyn GpuBackend, s: &W4Scratch) -> Result<()> {
    for p in [s.ws, s.xq, s.xs] {
        gpu.free(p)?;
    }
    Ok(())
}

/// Rows checked against the host references: tile edges and the last row.
pub fn sample_rows(m: usize) -> Vec<usize> {
    let mut r: Vec<usize> = [0, 1, 15, 16, 63, 64, 127, 511, 1023, 2047]
        .into_iter()
        .filter(|&r| r < m)
        .collect();
    r.push(m - 1);
    r.dedup();
    r
}

/// The prompt GEMM of `w` on the first `m` rows of `x`, on one route, read from a guarded output.
pub fn gemm(
    gpu: &dyn GpuBackend,
    w: &Weight,
    x: DevicePtr,
    m: usize,
    s: &W4Scratch,
    int8: bool,
) -> Result<Vec<u16>> {
    let out = Guarded::new(gpu, m * w.dev.n * 2)?;
    let ran = prefill::run_packed(&[(w.dev, out.ptr())], gpu, x, m, s, int8, 0)?;
    ensure!(ran, "the prompt GEMM declined {:?}", w.dev);
    let bytes = out.read(gpu)?;
    gpu.free(out.base)?;
    Ok(u16s(&bytes))
}

/// Rows `rows` of a row-major `[_, n]` matrix.
pub fn pick<T: Copy>(all: &[T], n: usize, rows: &[usize]) -> Vec<T> {
    rows.iter()
        .flat_map(|&r| all[r * n..(r + 1) * n].iter().copied())
        .collect()
}

/// Mean microseconds of `run` over 20 calls after 3 warm-ups, synchronized.
pub fn time_us(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..3 {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..20 {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / 20.0)
}
