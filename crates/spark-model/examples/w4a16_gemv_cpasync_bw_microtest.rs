// SPDX-License-Identifier: AGPL-3.0-only
//! Cold-DRAM bandwidth microtest: `w4a16_gemv` vs `w4a16_gemv_sw` vs
//! `w4a16_gemv_cpasync` on Qwen3.6-35B M=1 shapes.
//!
//! Exit 0 if cpasync is not slower than base gemv by more than 3% on the
//! production shapes. Informational GB/s is always printed.

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const WARMUP: usize = 8;
const ITERS: usize = 40;
const COLD_CYCLE_BYTES: usize = 256 << 20;

const SHAPES: &[(&str, u32, u32)] = &[
    ("gdn in_proj N=12288 K=2048", 12288, 2048),
    ("attn Q      N=8192  K=2048", 8192, 2048),
    ("gdn out     N=2048  K=4096", 2048, 4096),
];

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
    fn unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 23) as f32) * 2.0 - 1.0
    }
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
    outs: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, outs), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bs)
        .arg_f32(1.0)
        .arg_ptr(c)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

struct Timed {
    name: &'static str,
    us: f64,
    gbps: f64,
}

fn time_kernel(
    g: &dyn GpuBackend,
    name: &'static str,
    kh: KernelHandle,
    a: DevicePtr,
    copies: &[(DevicePtr, DevicePtr)],
    c: DevicePtr,
    n: u32,
    k: u32,
    outs: u32,
    weight_bytes: usize,
) -> Result<Timed> {
    let go = |i: usize| {
        launch(
            g,
            kh,
            a,
            copies[i % copies.len()].0,
            copies[i % copies.len()].1,
            c,
            n,
            k,
            outs,
        )
    };
    for i in 0..WARMUP {
        go(i)?;
    }
    g.synchronize(0)?;
    let t0 = Instant::now();
    for i in 0..ITERS {
        go(WARMUP + i)?;
    }
    g.synchronize(0)?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    let gbps = weight_bytes as f64 / (us * 1e-6) / 1e9;
    Ok(Timed { name, us, gbps })
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let peak: f64 = std::env::var("ATLAS_PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(273.0);

    let m1 = g.kernel("w4a16_gemv", "w4a16_gemv");
    let sw = g.kernel("w4a16_gemv", "w4a16_gemv_sw");
    let cp = g.kernel("w4a16_gemv", "w4a16_gemv_cpasync");
    let (Ok(m1), Ok(sw), Ok(cp)) = (m1, sw, cp) else {
        eprintln!("w4a16 GEMV kernels absent — SKIP");
        std::process::exit(2);
    };

    eprintln!("W4A16 M=1 cp.async cold-DRAM  peak {peak:.0} GB/s, {ITERS} iters\n");

    let mut rng = XorShift(0x9E3779B97F4A7C15);
    let mut ok = true;
    for &(label, n, k) in SHAPES {
        let (n_us, k_us) = (n as usize, k as usize);
        let packed_bytes = n_us * k_us / 2;
        let scale_bytes = n_us * k_us / 16;
        let weight_bytes = packed_bytes + scale_bytes;
        let copies_n = (COLD_CYCLE_BYTES.div_ceil(weight_bytes)).clamp(1, 16);

        let a_host: Vec<u16> = (0..k_us)
            .map(|_| f32_to_bf16_bits(rng.unit_f32()))
            .collect();
        let b_host: Vec<u8> = (0..packed_bytes).map(|_| rng.byte()).collect();
        let bs_host: Vec<u8> = (0..scale_bytes)
            .map(|_| 0x18 + (rng.byte() & 0x0F))
            .collect();

        let a = g.alloc(k_us * 2)?;
        let c = g.alloc(n_us * 2)?;
        let a_bytes: Vec<u8> = a_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d(&a_bytes, a)?;
        let mut copies = Vec::with_capacity(copies_n);
        for _ in 0..copies_n {
            let b = g.alloc(packed_bytes)?;
            let bs = g.alloc(scale_bytes)?;
            g.copy_h2d(&b_host, b)?;
            g.copy_h2d(&bs_host, bs)?;
            copies.push((b, bs));
        }

        let floor_us = weight_bytes as f64 / (peak * 1e9) * 1e6;
        eprintln!(
            "── {label}  weights {:.2} MB × {copies_n} copies, floor {floor_us:.1} us ──",
            weight_bytes as f64 / 1e6
        );

        let t_m1 = time_kernel(g, "gemv M=1", m1, a, &copies, c, n, k, 4, weight_bytes)?;
        let t_sw = time_kernel(g, "gemv_sw", sw, a, &copies, c, n, k, 8, weight_bytes)?;
        let t_cp = time_kernel(g, "gemv_cpasync", cp, a, &copies, c, n, k, 4, weight_bytes)?;
        for t in [&t_m1, &t_sw, &t_cp] {
            eprintln!(
                "  {:<16} {:>8.2} us  {:>6.1} GB/s  {:>5.1}% peak  {:>5.2}x floor",
                t.name,
                t.us,
                t.gbps,
                100.0 * t.gbps / peak,
                t.us / floor_us,
            );
        }
        let vs = t_cp.us / t_m1.us;
        eprintln!("  cpasync / gemv = {vs:.3}x   (target <1.0; fail if >1.03)\n");
        if vs > 1.03 {
            ok = false;
        }

        g.free(a)?;
        g.free(c)?;
        for (b, bs) in copies {
            g.free(b)?;
            g.free(bs)?;
        }
    }

    if ok {
        eprintln!("PASS — cpasync is not a bandwidth regression vs w4a16_gemv");
        Ok(())
    } else {
        eprintln!("FAIL — cpasync is >3% slower than w4a16_gemv on a production shape");
        std::process::exit(1);
    }
}
