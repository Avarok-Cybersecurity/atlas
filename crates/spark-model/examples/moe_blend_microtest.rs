// SPDX-License-Identifier: AGPL-3.0-only

//! Oracle for moe_weighted_sum_blend (moe_expert_gemv.cu) — the MoE combine:
//!   sig = sigmoid(Σ_k input[k]·gate_weight[k])
//!   out[j] = Σ_e weights[e]·expert_out[e*hidden+j] + sig·shared_out[j]
//! Grid (ceil(hidden/256),1,1), Block 256. Tests both gated + NULL-gate modes.
//!
//! cargo run --release -p spark-model --example moe_blend_microtest --features cuda,gpu-examples -- [hidden] [top_k] [K] [seed]

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

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
fn u16le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn up(gpu: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(b.len().max(16))?;
    gpu.copy_h2d(b, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let hidden: usize = a.get(1).map_or(2048, |s| s.parse().unwrap());
    let top_k: usize = a.get(2).map_or(8, |s| s.parse().unwrap());
    let k: usize = a.get(3).map_or(2048, |s| s.parse().unwrap());
    let seed: u64 = a.get(4).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    println!(
        "=== moe_weighted_sum_blend microtest: hidden={hidden} top_k={top_k} K={k} seed=0x{seed:X} ==="
    );

    let mut rng = Rng(seed);
    let eo: Vec<u16> = (0..top_k * hidden)
        .map(|_| f32_to_bf16(rng.uniform(-1.0, 1.0)))
        .collect();
    let ew: Vec<f32> = (0..top_k).map(|_| rng.uniform(0.0, 0.3)).collect();
    let so: Vec<u16> = (0..hidden)
        .map(|_| f32_to_bf16(rng.uniform(-1.0, 1.0)))
        .collect();
    let inp: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16(rng.uniform(-1.0, 1.0)))
        .collect();
    let gw: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16(rng.uniform(-0.5, 0.5)))
        .collect();

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let eop = up(gpu, &u16le(&eo))?;
    let ewp = up(gpu, &f32le(&ew))?;
    let sop = up(gpu, &u16le(&so))?;
    let inpp = up(gpu, &u16le(&inp))?;
    let gwp = up(gpu, &u16le(&gw))?;

    let run = |gate_null: bool| -> Result<f64> {
        let outp = gpu.alloc(hidden * 2)?;
        let handle = gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?;
        let gwarg = if gate_null { DevicePtr(0) } else { gwp };
        KernelLaunch::new(gpu, handle)
            .grid([div_ceil(hidden as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(outp)
            .arg_ptr(eop)
            .arg_ptr(ewp)
            .arg_ptr(sop)
            .arg_ptr(inpp)
            .arg_ptr(gwarg)
            .arg_u32(hidden as u32)
            .arg_u32(top_k as u32)
            .arg_u32(k as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut raw = vec![0u8; hidden * 2];
        gpu.copy_d2h(outp, &mut raw)?;
        let og: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        // CPU ref
        let sig = if gate_null {
            1.0f32
        } else {
            let mut d = 0f32;
            for kk in 0..k {
                d += bf16_to_f32(inp[kk]) * bf16_to_f32(gw[kk]);
            }
            1.0 / (1.0 + (-d).exp())
        };
        let (mut dt, mut na, mut nb) = (0f64, 0f64, 0f64);
        for j in 0..hidden {
            let mut acc = 0f32;
            for e in 0..top_k {
                acc += ew[e] * bf16_to_f32(eo[e * hidden + j]);
            }
            acc += sig * bf16_to_f32(so[j]);
            let g = bf16_to_f32(og[j]) as f64;
            let c = f32_to_bf16(acc);
            let cy = bf16_to_f32(c) as f64;
            dt += g * cy;
            na += g * g;
            nb += cy * cy;
        }
        gpu.free(outp).ok();
        Ok(if na == 0.0 || nb == 0.0 {
            f64::NAN
        } else {
            dt / (na.sqrt() * nb.sqrt())
        })
    };
    let cg = run(false)?;
    let cn = run(true)?;
    println!("gated_cos={cg:.6}  nullgate_cos={cn:.6}");
    for p in [eop, ewp, sop, inpp, gwp] {
        gpu.free(p).ok();
    }
    if cg >= COSINE_GATE && cn >= COSINE_GATE {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL (blend diverges)");
        std::process::exit(1);
    }
}
