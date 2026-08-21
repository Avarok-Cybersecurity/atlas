// SPDX-License-Identifier: AGPL-3.0-only
//! Bit-exactness + speed check: token-parallel causal_conv1d prefill vs the
//! serial one. The accumulation order is identical
//! (`b + s0*w0 + s1*w1 + s2*w2 + s3*w3`), so anything short of BIT-IDENTICAL
//! output is a bug, not rounding.
use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uni(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * ((self.next() >> 11) as f32 / (1u64 << 53) as f32)
    }
}
fn bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}
fn up_u16(g: &dyn GpuBackend, v: &[u16]) -> Result<DevicePtr> {
    let mut b = Vec::with_capacity(v.len() * 2);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_u16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u16>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn main() -> Result<()> {
    let dim: usize = std::env::var("CV_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let seq: usize = std::env::var("CV_SEQ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2700);
    let dconv = 4usize;
    println!("=== causal_conv1d prefill: token-parallel vs serial ===");
    println!("dim={dim} seq_len={seq} d_conv={dconv}");

    let be = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &be;
    let st = g.create_stream()?;
    let mut r = Rng(0x_C0FF_EE11);

    let inp: Vec<u16> = (0..seq * dim).map(|_| bf16(r.uni(-2.0, 2.0))).collect();
    let wt: Vec<u16> = (0..dim * dconv).map(|_| bf16(r.uni(-0.5, 0.5))).collect();
    let bi: Vec<f32> = (0..dim).map(|_| r.uni(-0.1, 0.1)).collect();
    let s0: Vec<f32> = (0..dim * dconv).map(|_| r.uni(-1.0, 1.0)).collect();

    let (pi, pw, pb) = (up_u16(g, &inp)?, up_u16(g, &wt)?, up_f32(g, &bi)?);
    let run = |kern: &str, grid: [u32; 3], blk: [u32; 3]| -> Result<(Vec<u16>, f64)> {
        let ps = up_f32(g, &s0)?; // fresh state each run
        let po = g.alloc(seq * dim * 2)?;
        let h = g.kernel("causal_conv1d", kern)?;
        let go = || -> Result<()> {
            KernelLaunch::new(g, h)
                .grid(grid)
                .block(blk)
                .arg_ptr(ps)
                .arg_ptr(pi)
                .arg_ptr(pw)
                .arg_ptr(pb)
                .arg_ptr(po)
                .arg_u32(dim as u32)
                .arg_u32(dconv as u32)
                .arg_u32(seq as u32)
                .arg_u32(dim as u32)
                .arg_u32(dim as u32)
                .launch(st)?;
            Ok(())
        };
        go()?;
        g.synchronize(st)?;
        let out = dn_u16(g, po, seq * dim)?;
        for _ in 0..3 {
            go()?;
        }
        g.synchronize(st)?;
        let t = std::time::Instant::now();
        for _ in 0..10 {
            go()?;
        }
        g.synchronize(st)?;
        Ok((out, t.elapsed().as_secs_f64() * 1000.0 / 10.0))
    };

    let (a, ta) = run(
        "causal_conv1d_update_prefill",
        [div_ceil(dim as u32, 256), 1, 1],
        [256, 1, 1],
    )?;
    let (b, tb) = run(
        "causal_conv1d_update_prefill_tp",
        [div_ceil(dim as u32, 32), div_ceil(seq as u32, 8 * 8), 1],
        [32, 8, 1],
    )?;

    let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    println!("  serial          {ta:8.3} ms/iter");
    println!("  token-parallel  {tb:8.3} ms/iter   ({:.2}x)", ta / tb);
    println!("  mismatched elements: {diff} / {}", a.len());
    if diff != 0 {
        bail!("NOT bit-identical: {diff} elements differ");
    }
    println!("\nPASS: bit-identical");
    Ok(())
}
