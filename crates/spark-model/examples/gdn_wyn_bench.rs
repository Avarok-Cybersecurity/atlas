// SPDX-License-Identifier: AGPL-3.0-only
//! Microbench: chain-verify GDN at K∈{5..8} — one pool-layout wyN launch
//! (`gated_delta_rule_wy5..wy8`) vs the serial per-token fallback it
//! replaces (K× `gated_delta_rule_decode` + K× h-intermediate copy_d2d).
//! 27B GDN dims: NK=16, NV=32, KD=VD=128 (h_state 2 MiB FP32).
//!
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b \
//!   cargo run -p spark-model --release --example gdn_wyn_bench \
//!       --features cuda,gpu-examples
//!
//! NOTE: the qwen3.6-27b `gated_delta_rule_decode` override takes FP32
//! q/k/v (matching the production fallback's FP32 conv output); the wyN
//! kernels take BF16 (matching the fused arms' BF16 conv). Both are fed
//! from pre-uploaded buffers so only kernel + copy time is measured.
use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const H_NUMEL: usize = NV * KD * VD; // 524288 floats = 2 MiB
const ITERS: usize = 300;
const WARMUP: usize = 30;

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len())?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let decode_k = g.kernel("gated_delta_rule", "gated_delta_rule_decode")?;

    let kmax = 8usize;
    // Deterministic pseudo-random inputs (values irrelevant for timing).
    let h0: Vec<u8> = (0..H_NUMEL)
        .flat_map(|i| (((i % 251) as f32 - 125.0) * 1e-3).to_le_bytes())
        .collect();
    let qkv_f32: Vec<u8> = (0..kmax * NK * KD)
        .flat_map(|i| (((i % 17) as f32 - 8.0) * 0.05).to_le_bytes())
        .collect();
    let v_f32: Vec<u8> = (0..kmax * NV * VD)
        .flat_map(|i| (((i % 13) as f32 - 6.0) * 0.05).to_le_bytes())
        .collect();
    let qkv_bf16: Vec<u8> = (0..kmax * NK * KD)
        .flat_map(|i| {
            bf16::from_f32(((i % 17) as f32 - 8.0) * 0.05)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let v_bf16: Vec<u8> = (0..kmax * NV * VD)
        .flat_map(|i| {
            bf16::from_f32(((i % 13) as f32 - 6.0) * 0.05)
                .to_bits()
                .to_le_bytes()
        })
        .collect();
    let gb: Vec<u8> = (0..kmax * NV)
        .flat_map(|i| (0.9f32 + ((i % 7) as f32) * 0.01).to_le_bytes())
        .collect();

    let h = up(g, &h0)?;
    let qf = up(g, &qkv_f32)?;
    let kf = up(g, &qkv_f32)?;
    let vf = up(g, &v_f32)?;
    let qb = up(g, &qkv_bf16)?;
    let kb = up(g, &qkv_bf16)?;
    let vb = up(g, &v_bf16)?;
    let gate = up(g, &gb)?;
    let beta = up(g, &gb)?;
    let out = g.alloc(kmax * NV * VD * 2)?;
    let inter_pool = g.alloc((kmax - 1) * H_NUMEL * 4)?;

    for &k in &[5usize, 6, 7, 8] {
        let wyn_k = g.kernel("gated_delta_rule_wyn", &format!("gated_delta_rule_wy{k}"))?;

        // ── (a) serial fallback: K× decode + K× h copy_d2d ──
        let serial = |iters: usize| -> Result<f64> {
            g.synchronize(0)?;
            let t0 = Instant::now();
            for _ in 0..iters {
                for t in 0..k {
                    KernelLaunch::new(g, decode_k)
                        .grid([NV as u32, 1, 1])
                        .block([128, 1, 1])
                        .arg_ptr(h)
                        .arg_ptr(qf.offset(t * NK * KD * 4))
                        .arg_ptr(kf.offset(t * NK * KD * 4))
                        .arg_ptr(vf.offset(t * NV * VD * 4))
                        .arg_ptr(gate.offset(t * NV * 4))
                        .arg_ptr(beta.offset(t * NV * 4))
                        .arg_ptr(out)
                        .arg_u32(1)
                        .arg_u32(NK as u32)
                        .arg_u32(NV as u32)
                        .arg_u32(KD as u32)
                        .arg_u32(VD as u32)
                        .launch(0)?;
                    g.copy_d2d_async(
                        h,
                        inter_pool.offset((t % (kmax - 1)) * H_NUMEL * 4),
                        H_NUMEL * 4,
                        0,
                    )?;
                }
            }
            g.synchronize(0)?;
            Ok(t0.elapsed().as_secs_f64() / iters as f64 * 1e6)
        };

        // ── (b) one wyN launch ──
        let fused = |iters: usize| -> Result<f64> {
            g.synchronize(0)?;
            let t0 = Instant::now();
            for _ in 0..iters {
                KernelLaunch::new(g, wyn_k)
                    .grid([NV as u32, 1, 1])
                    .block([128, 1, 1])
                    .arg_ptr(h)
                    .arg_ptr(qb)
                    .arg_ptr(kb)
                    .arg_ptr(vb)
                    .arg_ptr(gate)
                    .arg_ptr(beta)
                    .arg_ptr(out)
                    .arg_ptr(inter_pool)
                    .arg_u32(H_NUMEL as u32)
                    .arg_u32(1)
                    .arg_u32(NK as u32)
                    .arg_u32(NV as u32)
                    .arg_u32(KD as u32)
                    .arg_u32(VD as u32)
                    .arg_u32((NK * KD) as u32)
                    .arg_u32((NV * VD) as u32)
                    .arg_u32(NV as u32)
                    .launch(0)?;
            }
            g.synchronize(0)?;
            Ok(t0.elapsed().as_secs_f64() / iters as f64 * 1e6)
        };

        serial(WARMUP)?;
        let s_us = serial(ITERS)?;
        fused(WARMUP)?;
        let f_us = fused(ITERS)?;
        println!(
            "K={k}  serial(K×decode+K×copy)={s_us:8.2} µs   wy{k}={f_us:8.2} µs   speedup={:.2}x",
            s_us / f_us
        );
    }

    for p in [h, qf, kf, vf, qb, kb, vb, gate, beta, out, inter_pool] {
        let _ = g.free(p);
    }
    Ok(())
}
