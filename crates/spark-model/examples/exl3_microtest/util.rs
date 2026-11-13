// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the EXL3 microtest's plumbing: a deterministic generator, random EXL3 matrices,
//! uploads and downloads, verdict bookkeeping, the parallel host decode and references, and the
//! timing loop.

use anyhow::Result;
use spark_model::port::exl3::format::{Exl3Codebook, trellis_bytes, u16_bytes};
use spark_model::port::exl3::reference::{codebook_lut, unpack_tile_rows};
use spark_model::port::exl3::reference_math::{GemvBound, f32_to_bf16, gemv_row_ref};
use spark_model::port::w4::W4Scratch;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::Instant;

pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    pub fn normal(&mut self, sigma: f32) -> f32 {
        (self.unit() + self.unit() + self.unit() + self.unit() - 2.0) * 1.7 * sigma
    }
    pub fn u16s(&mut self, len: usize) -> Vec<u16> {
        (0..len).map(|_| self.next_u32() as u16).collect()
    }
    /// bf16 bits of `len` roughly normal values.
    pub fn bf16s(&mut self, len: usize) -> Vec<u16> {
        (0..len).map(|_| f32_to_bf16(self.normal(1.0))).collect()
    }
    /// fp16 bits of scales with magnitude 0.5 to 1.5 and random signs.
    pub fn scales(&mut self, len: usize) -> Vec<u16> {
        (0..len)
            .map(|_| {
                let m = 0.5 + self.unit();
                let v = if self.next_u32() & 1 == 1 { -m } else { m };
                half::f16::from_f32(v).to_bits()
            })
            .collect()
    }
}

/// A random EXL3 matrix: trellis int16 elements and fp16 scales, owned.
pub struct Matrix {
    pub k: usize,
    pub n: usize,
    pub halves: u16,
    pub cb: Exl3Codebook,
    pub trellis: Vec<u16>,
    pub suh: Vec<u16>,
    pub svh: Vec<u16>,
}

impl Matrix {
    pub fn random(r: &mut Lcg, k: usize, n: usize, halves: u16, cb: Exl3Codebook) -> Self {
        Self {
            k,
            n,
            halves,
            cb,
            trellis: r.u16s(trellis_bytes(k, n, halves) / 2),
            suh: r.scales(k),
            svh: r.scales(n),
        }
    }
    pub fn bytes(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        (
            u16_bytes(&self.trellis),
            u16_bytes(&self.suh),
            u16_bytes(&self.svh),
        )
    }
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get())
}

/// `Wq [k, n]` fp16 bits, decoded on every core.
pub fn unpack_par(m: &Matrix) -> Vec<u16> {
    let lut = codebook_lut(m.cb);
    let mut wq = vec![0u16; m.k * m.n];
    let kt = m.k / 16;
    let per = kt.div_ceil(threads());
    std::thread::scope(|s| {
        for (i, band) in wq.chunks_mut(per * 16 * m.n).enumerate() {
            let (lut, trellis) = (&lut, &m.trellis);
            let tk0 = i * per;
            let tk1 = (tk0 + per).min(kt);
            s.spawn(move || unpack_tile_rows(m.halves, lut, trellis, m.n, tk0, tk1, band));
        }
    });
    wq
}

/// The FP64 reference and bound of every row of `xh [rows, k]`, rows spread over the cores.
pub fn refs_par(
    xh: &[u16],
    rows: usize,
    wq: &[u16],
    k: usize,
    n: usize,
    svh: &[u16],
) -> Vec<GemvBound> {
    let mut out: Vec<GemvBound> = (0..rows).map(|_| GemvBound::default()).collect();
    let per = rows.div_ceil(threads());
    std::thread::scope(|s| {
        for (i, chunk) in out.chunks_mut(per).enumerate() {
            s.spawn(move || {
                for (j, b) in chunk.iter_mut().enumerate() {
                    let r = i * per + j;
                    *b = gemv_row_ref(&xh[r * k..(r + 1) * k], wq, k, n, svh);
                }
            });
        }
    });
    out
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

pub fn download_u16(gpu: &dyn GpuBackend, p: DevicePtr, elems: usize) -> Result<Vec<u16>> {
    gpu.synchronize(0)?;
    let mut host = vec![0u8; elems * 2];
    gpu.copy_d2h(p, &mut host)?;
    Ok(host
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

pub fn scratch(gpu: &dyn GpuBackend, bytes: usize) -> Result<W4Scratch> {
    Ok(W4Scratch {
        ws: gpu.alloc(bytes.max(256))?,
        ws_bytes: bytes.max(256),
        tickets: DevicePtr::NULL,
        xq: DevicePtr::NULL,
        xs: DevicePtr::NULL,
    })
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
    pub fn info(&self, name: &str, detail: &str) {
        println!("  [INFO] {name}: {detail}");
    }
}

pub const REPS: usize = 20;
pub const WARMUP: usize = 3;

/// Mean microseconds of `run()` over `REPS` calls after `WARMUP`.
pub fn time_us(gpu: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    gpu.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / REPS as f64)
}
