// SPDX-License-Identifier: AGPL-3.0-only

//! Oracle for the SHARED-expert path of moe_expert_silu_down_shared_fp8:
//!   s_act[k] = silu(gate_in[k]) * up_in[k];   out[n] = Σ_k s_act[k]·E4M3(down_w[n*K+k])·bscale[n/128,k/128]
//! top_k=0 → is_shared only. Grid (ceil(N/8), 1, 1), Block 128.
//! N=hidden (down output), K=moe_intermediate (down input).
//!
//! cargo run --release -p spark-model --example moe_silu_down_microtest --features cuda,gpu-examples -- [N] [K] [seed]

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const FP8_BLOCK: usize = 128;
const COSINE_GATE: f64 = 0.999;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32)
    }
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16(f: f32) -> u16 {
    let b = f.to_bits();
    ((b.wrapping_add(0x7FFF + ((b >> 16) & 1))) >> 16) as u16
}
fn e4m3_to_f32(byte: u8) -> f32 {
    let s = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((byte >> 3) & 0x0F) as i32;
    let m = (byte & 0x07) as i32;
    if e == 0 {
        s * (m as f32 / 8.0) * 2f32.powi(-6)
    } else if e == 0x0F && m == 0x07 {
        0.0
    } else {
        s * (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
    }
}
fn f32_to_e4m3(v: f32) -> u8 {
    let mut bb = 0u8;
    let mut be = f32::INFINITY;
    for b in 0..=255u8 {
        let d = e4m3_to_f32(b);
        if d.is_finite() {
            let e = (d - v).abs();
            if e < be {
                be = e;
                bb = b;
            }
        }
    }
    bb
}
fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}
fn u16le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).map_or(2048, |s| s.parse().unwrap());
    let k: usize = a.get(2).map_or(512, |s| s.parse().unwrap());
    let seed: u64 = a.get(3).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    let kb = k.div_ceil(FP8_BLOCK);
    let nb = n.div_ceil(FP8_BLOCK);
    println!("=== moe_silu_down (shared) microtest: N={n} K={k} seed=0x{seed:X} ===");

    let mut rng = Rng(seed);
    let gate: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16(rng.uniform(-2.0, 2.0)))
        .collect();
    let upv: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16(rng.uniform(-2.0, 2.0)))
        .collect();
    let dw: Vec<u8> = (0..n * k)
        .map(|_| f32_to_e4m3(rng.uniform(-0.4, 0.4)))
        .collect();
    let ds: Vec<f32> = (0..nb * kb).map(|_| rng.uniform(0.5, 1.5)).collect();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let gi = up(gpu, &u16le(&gate))?;
    let ui = up(gpu, &u16le(&upv))?;
    let dwp = up(gpu, &dw)?;
    let dsp = up(gpu, &f32le(&ds))?;
    let outp = gpu.alloc(n * 2)?;
    let dummy = gpu.alloc(64)?;

    let handle = gpu.kernel(
        "moe_shared_expert_fused_fp8",
        "moe_expert_silu_down_shared_fp8",
    )?;
    KernelLaunch::new(gpu, handle)
        .grid([div_ceil(n as u32, 8), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(dummy)
        .arg_ptr(dummy) // gate_out, up_out (routed)
        .arg_ptr(dummy)
        .arg_ptr(dummy) // down weight/scale ptrs (routed)
        .arg_ptr(dummy)
        .arg_ptr(dummy) // output (routed), expert_indices
        .arg_ptr(gi)
        .arg_ptr(ui) // sh_gate_in, sh_up_in
        .arg_ptr(dwp)
        .arg_ptr(dsp)
        .arg_ptr(outp) // sh_down weight/scale/out
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(0)
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let mut raw = vec![0u8; n * 2];
    gpu.copy_d2h(outp, &mut raw)?;
    let og: Vec<u16> = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    // CPU ref
    let mut s_act = vec![0f32; k];
    for i in 0..k {
        let gf = bf16_to_f32(gate[i]);
        let uf = bf16_to_f32(upv[i]);
        s_act[i] = (gf / (1.0 + (-gf).exp())) * uf;
    }
    let oc: Vec<u16> = (0..n)
        .map(|nn| {
            let nblk = nn / FP8_BLOCK;
            let mut acc = 0f32;
            for kk in 0..k {
                acc += s_act[kk] * e4m3_to_f32(dw[nn * k + kk]) * ds[nblk * kb + kk / FP8_BLOCK];
            }
            f32_to_bf16(acc)
        })
        .collect();

    let (mut d, mut na, mut nb2) = (0f64, 0f64, 0f64);
    for i in 0..n {
        let x = bf16_to_f32(og[i]) as f64;
        let y = bf16_to_f32(oc[i]) as f64;
        d += x * y;
        na += x * x;
        nb2 += y * y;
    }
    let c = if na == 0.0 || nb2 == 0.0 {
        f64::NAN
    } else {
        d / (na.sqrt() * nb2.sqrt())
    };
    println!("down_cos={c:.6}");
    for p in [gi, ui, dwp, dsp, outp, dummy] {
        gpu.free(p).ok();
    }
    if c >= COSINE_GATE {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL (shared silu_down GEMV diverges)");
        std::process::exit(1);
    }
}
