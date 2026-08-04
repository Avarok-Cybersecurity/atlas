// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone oracle for the SHARED-expert path of moe_expert_gate_up_shared_fp8
//! (moe_shared_expert_fused_fp8.cu) — the always-on expert whose ~1.6% value
//! drift is the prime suspect for the HIP 35B MoE moe_out divergence.
//!
//! Launch with top_k=0 so is_shared=(slot==0)=true for the single grid.y=0
//! block → only the shared expert runs (routed ptr tables unused). Grid
//! (ceil(N/8), 1, 2), Block 128. proj=blockIdx.z (0=gate,1=up).
//!
//!   sh_gate_out[n] = Σ_k A[k]·E4M3(sh_gate_w[n*K+k])·sh_gate_bscale[n/128, k/128]
//!   sh_up_out[n]   = same with the up weights.   FP32 accum, BF16 out.
//!
//! Usage: cargo run --release -p spark-model --example moe_shared_microtest \
//!          --features cuda,gpu-examples -- [N] [K] [seed]
//! Exit 0 = PASS (gate & up cosine >= gate), 1 = FAIL.

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
    let bits = f.to_bits();
    ((bits.wrapping_add(0x7FFF + ((bits >> 16) & 1))) >> 16) as u16
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
    let mut best = 0u8;
    let mut be = f32::INFINITY;
    for b in 0..=255u8 {
        let d = e4m3_to_f32(b);
        if d.is_finite() {
            let e = (d - v).abs();
            if e < be {
                be = e;
                best = b;
            }
        }
    }
    best
}
fn up_bytes(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
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
    let n: usize = a.get(1).map_or(512, |s| s.parse().unwrap());
    let k: usize = a.get(2).map_or(2048, |s| s.parse().unwrap());
    let seed: u64 = a.get(3).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    let kb = k.div_ceil(FP8_BLOCK);
    let nb = n.div_ceil(FP8_BLOCK);
    println!("=== moe_shared (gate_up) microtest: N={n} K={k} seed=0x{seed:X} ===");

    let mut rng = Rng(seed);
    let av: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16(rng.uniform(-1.0, 1.0)))
        .collect();
    let gw: Vec<u8> = (0..n * k)
        .map(|_| f32_to_e4m3(rng.uniform(-0.4, 0.4)))
        .collect();
    let uw: Vec<u8> = (0..n * k)
        .map(|_| f32_to_e4m3(rng.uniform(-0.4, 0.4)))
        .collect();
    let gs: Vec<f32> = (0..nb * kb).map(|_| rng.uniform(0.5, 1.5)).collect();
    let us: Vec<f32> = (0..nb * kb).map(|_| rng.uniform(0.5, 1.5)).collect();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let a_p = up_bytes(gpu, &u16le(&av))?;
    let gw_p = up_bytes(gpu, &gw)?;
    let uw_p = up_bytes(gpu, &uw)?;
    let gs_p = up_bytes(gpu, &f32le(&gs))?;
    let us_p = up_bytes(gpu, &f32le(&us))?;
    let go_p = gpu.alloc(n * 2)?;
    let uo_p = gpu.alloc(n * 2)?;
    // Dummy routed tables / outputs (unused when is_shared).
    let dummy = gpu.alloc(64)?;

    let handle = gpu.kernel(
        "moe_shared_expert_fused_fp8",
        "moe_expert_gate_up_shared_fp8",
    )?;
    KernelLaunch::new(gpu, handle)
        .grid([div_ceil(n as u32, 8), 1, 2])
        .block([128, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(dummy)
        .arg_ptr(dummy)
        .arg_ptr(dummy) // gate routed: w_ptrs, scale_ptrs, out
        .arg_ptr(dummy)
        .arg_ptr(dummy)
        .arg_ptr(dummy) // up routed
        .arg_ptr(dummy) // expert_indices
        .arg_ptr(gw_p)
        .arg_ptr(gs_p)
        .arg_ptr(go_p) // shared gate: w, scale, out
        .arg_ptr(uw_p)
        .arg_ptr(us_p)
        .arg_ptr(uo_p) // shared up
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(0) // N, K, top_k=0
        .launch(stream)?;
    gpu.synchronize(stream)?;

    let read = |p: DevicePtr| -> Vec<u16> {
        let mut raw = vec![0u8; n * 2];
        gpu.copy_d2h(p, &mut raw).ok();
        raw.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let go = read(go_p);
    let uo = read(uo_p);

    let cpu = |w: &[u8], s: &[f32]| -> Vec<u16> {
        (0..n)
            .map(|nn| {
                let nblk = nn / FP8_BLOCK;
                let mut acc = 0f32;
                for kk in 0..k {
                    let sc = s[nblk * kb + kk / FP8_BLOCK];
                    acc += bf16_to_f32(av[kk]) * e4m3_to_f32(w[nn * k + kk]) * sc;
                }
                f32_to_bf16(acc)
            })
            .collect()
    };
    let go_c = cpu(&gw, &gs);
    let uo_c = cpu(&uw, &us);
    let cosv = |g: &[u16], c: &[u16]| -> f64 {
        let (mut d, mut na, mut nb_) = (0f64, 0f64, 0f64);
        for i in 0..n {
            let x = bf16_to_f32(g[i]) as f64;
            let y = bf16_to_f32(c[i]) as f64;
            d += x * y;
            na += x * x;
            nb_ += y * y;
        }
        if na == 0.0 || nb_ == 0.0 {
            return f64::NAN;
        }
        d / (na.sqrt() * nb_.sqrt())
    };
    let cg = cosv(&go, &go_c);
    let cu = cosv(&uo, &uo_c);
    println!("gate_cos={cg:.6}  up_cos={cu:.6}");
    for p in [a_p, gw_p, uw_p, gs_p, us_p, go_p, uo_p, dummy] {
        gpu.free(p).ok();
    }
    if cg >= COSINE_GATE && cu >= COSINE_GATE {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL (shared-expert gate/up GEMV diverges)");
        std::process::exit(1);
    }
}
