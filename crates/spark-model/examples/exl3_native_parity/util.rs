// SPDX-License-Identifier: AGPL-3.0-only
//! Shared plumbing for the exl3_native_parity legs: RNG, device up/down
//! helpers, error metrics, and one-call GPU pipeline runners.

use anyhow::Result;
use half::f16;
use spark_model::layers::ops::{exl3_gemm, exl3_gemv};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Gates from the port map's tolerance derivation (vs the f64 truth):
/// fp32-accumulating GEMM legs, and the fp16-MMA-accumulating GEMV legs
/// (also used for fp16-C legs, which add fp16 split-k handoffs).
pub const GEMM_REL_RMS: f64 = 2.5e-3;
pub const GEMM_MAX_Z: f64 = 1.5e-2;
pub const GEMV_REL_RMS: f64 = 8e-3;
pub const GEMV_MAX_Z: f64 = 4e-2;

pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    pub fn u16(&mut self) -> u16 {
        (self.next() >> 24) as u16
    }
    /// Uniform in [0, 1).
    pub fn f(&mut self) -> f32 {
        (((self.next() >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    /// Standard normal (Box-Muller) — the tolerance gates were derived for
    /// N(0,1) activations.
    pub fn gauss(&mut self) -> f32 {
        let u1 = (self.f() as f64).max(1e-12);
        let u2 = self.f() as f64;
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
    /// Random-sign magnitude 0.5..1.5 f16 bits (suh/svh-like).
    pub fn scale_f16(&mut self) -> u16 {
        let m = 0.5 + self.f();
        let s = if self.next() & 1 == 0 { 1.0 } else { -1.0 };
        f16::from_f32(m * s).to_bits()
    }
    pub fn act_f16(&mut self) -> u16 {
        f16::from_f32(self.gauss()).to_bits()
    }
}

pub fn as_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

pub fn down_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

pub fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// (rel_rms, max_z): relative RMS error and max abs error normalized by the
/// truth RMS, over the whole output tensor.
pub fn metrics(y_gpu: &[f64], y64: &[f64]) -> (f64, f64) {
    assert_eq!(y_gpu.len(), y64.len());
    let n = y64.len() as f64;
    let ss: f64 = y64.iter().map(|v| v * v).sum();
    let rms = (ss / n).sqrt().max(1e-30);
    let mut dss = 0.0;
    let mut dmax = 0.0f64;
    for (a, b) in y_gpu.iter().zip(y64.iter()) {
        let d = a - b;
        dss += d * d;
        dmax = dmax.max(d.abs());
    }
    ((dss / ss.max(1e-30)).sqrt(), dmax / rms)
}

/// Everything a leg needs to launch: backend, the shared zeroed locks
/// buffer, and the once-resolved SM count.
pub struct Ctx<'a> {
    pub g: &'a dyn GpuBackend,
    pub locks: DevicePtr,
    pub sms: u32,
}

/// Uploaded tensor set for one (trellis, suh, svh) weight.
pub struct DevWeight {
    pub trellis: DevicePtr,
    pub suh: DevicePtr,
    pub svh: DevicePtr,
}

impl DevWeight {
    pub fn upload(g: &dyn GpuBackend, trellis: &[u16], suh: &[u16], svh: &[u16]) -> Result<Self> {
        Ok(Self {
            trellis: up(g, &as_bytes(trellis))?,
            suh: up(g, &as_bytes(suh))?,
            svh: up(g, &as_bytes(svh))?,
        })
    }
    pub fn free(&self, g: &dyn GpuBackend) {
        for p in [self.trellis, self.suh, self.svh] {
            g.free(p).ok();
        }
    }
}

pub struct PipelineOut {
    /// C as f64 (whatever the on-device dtype was).
    pub y: Vec<f64>,
    /// Raw C bytes for determinism checks.
    pub c_bytes: Vec<u8>,
    /// A_had contents (f16 bits) after the launch.
    pub a_had: Vec<u16>,
}

/// Run the full native pipeline once: upload A, alloc A_had + C, launch
/// gemv (`cfg = Some(_)` forces the config) or gemm (`cfg = None`,
/// `force_shape` optional), sync, download.
#[allow(clippy::too_many_arguments)]
pub fn run_pipeline(
    ctx: &Ctx,
    a_bits: &[u16],
    w: &DevWeight,
    m: usize,
    k: usize,
    n: usize,
    k_bits: u32,
    cb: u32,
    c_fp32: bool,
    gemv_cfg: Option<u32>,
    force_shape: Option<usize>,
) -> Result<PipelineOut> {
    let g = ctx.g;
    assert_eq!(a_bits.len(), m * k);
    let a_d = up(g, &as_bytes(a_bits))?;
    let a_had_d = g.alloc(m * k * 2)?;
    let c_elem = if c_fp32 { 4 } else { 2 };
    let c_d = g.alloc(m * n * c_elem)?;
    let stream = g.default_stream();

    if let Some(cfg) = gemv_cfg {
        let launched = exl3_gemv(
            g, a_d, w.trellis, c_d, m, k, n, k_bits, cb, c_fp32, ctx.locks, w.suh, a_had_d,
            w.svh, Some(cfg), ctx.sms, stream,
        )?;
        anyhow::ensure!(launched, "gemv refused (m={m} k={k} n={n} K={k_bits})");
    } else {
        exl3_gemm(
            g, a_d, w.trellis, c_d, m, k, n, k_bits, cb, c_fp32, ctx.locks, w.suh, a_had_d,
            w.svh, force_shape, ctx.sms, stream,
        )?;
    }
    g.synchronize(stream)?;

    let mut c_bytes = vec![0u8; m * n * c_elem];
    g.copy_d2h(c_d, &mut c_bytes)?;
    let y: Vec<f64> = if c_fp32 {
        c_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect()
    } else {
        c_bytes
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f64())
            .collect()
    };
    let a_had = down_u16(g, a_had_d, m * k)?;
    for p in [a_d, a_had_d, c_d] {
        g.free(p).ok();
    }
    Ok(PipelineOut { y, c_bytes, a_had })
}

/// Print + gate one numeric leg. Returns pass.
pub fn gate_leg(
    label: &str,
    y_gpu: &[f64],
    y64: &[f64],
    rel_rms_gate: f64,
    max_z_gate: f64,
) -> bool {
    let (rel_rms, max_z) = metrics(y_gpu, y64);
    let ok = rel_rms <= rel_rms_gate && max_z <= max_z_gate;
    println!(
        "{label}  rel_rms={rel_rms:.3e} (gate {rel_rms_gate:.1e})  max_z={max_z:.3e} (gate {max_z_gate:.1e})  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}
