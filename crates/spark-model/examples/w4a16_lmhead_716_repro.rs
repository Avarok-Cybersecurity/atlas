// SPDX-License-Identifier: AGPL-3.0-only

//! Repro rig for the decode lm_head tile-GEMM CUDA 716 fault (gb10-concurrency
//! campaign, STATE.md "lm_head -> tile GEMM: BUILT TWICE, FAILED TWICE").
//!
//! The standalone bench (`w4a16_m17_bench`) runs the exact shape clean, so this
//! rig reproduces the SERVE-side launch environment piece by piece:
//!   leg 1 solo          — lm_head GEMM alone on the legacy NULL stream (control)
//!   leg 2 same-stream   — decode-step kernel sequence (ssm gemms + lm_head +
//!                         argmax_bf16_batch + 4B D2H) on ONE created stream,
//!                         M cycling the ladder 2/4/8/12/16 like a serve ramp
//!   leg 3 cross-stream  — gemms on stream A, argmax on stream B (mirrors
//!                         `argmax_batch_dispatch` using `default_stream()`
//!                         while decode runs on its own stream)
//!   leg 4 conc-prefill  — leg 2 plus prefill-shaped `w4a16_gemm_t_m128` and
//!                         `memset_async` interleaved on a third stream
//!   leg 5 undersized-C  — DELIBERATE overrun: C sized 4 rows, M=16 — records
//!                         the error signature a real overrun produces
//!
//! Buffer sizes mirror the serve exactly: logits = 32 rows x vocab x 2
//! (`buffers/sizes.rs` logits_tokens = m.min(32)); twin = [K/2, N] + [K/16, N]
//! fresh allocations (`transpose_for_gemm`); launch geometry from
//! `gemm_dense.rs::w4a16_gemm_n128` and `sampling.rs::argmax_bf16_batch`.
//!
//!   ATLAS_TARGET_MODEL=qwen3.6-27b cargo run -p spark-model --release \
//!       --example w4a16_lmhead_716_repro --features cuda,gpu-examples
//!
//! Env: ATLAS_REPRO_ITERS (default 300 per leg), ATLAS_REPRO_LEG (run one leg).

use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const V: u32 = 248_320; // lm_head N (vocab)
const H: u32 = 5_120; // hidden (K)
const QKVZ_N: u32 = 16_384;
const OUTP_N: u32 = 5_120;
const OUTP_K: u32 = 6_144;
const FFN_N: u32 = 17_408;
const PRE_M: u32 = 512; // prefill chunk rows for the m128 leg
const LADDER: &[u32] = &[2, 4, 8, 12, 16];

struct W {
    b: DevicePtr,
    bs: DevicePtr,
}

fn twin(g: &dyn GpuBackend, n: u32, k: u32) -> Result<W> {
    let (n, k) = (n as usize, k as usize);
    let b = g.alloc(n * k / 2)?;
    let bs = g.alloc(n * k / 16)?;
    g.memset(b, 0x5A, n * k / 2)?;
    g.memset(bs, 0x5A, n * k / 16)?;
    Ok(W { b, bs })
}

/// Mirror of `gemm_dense.rs::w4a16_gemm_n128`: grid (N/128, M/64), block 128.
#[allow(clippy::too_many_arguments)]
fn gemm_t(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    w: &W,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.b)
        .arg_ptr(w.bs)
        .arg_f32(1.0)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Mirror of `sampling.rs::argmax_bf16_batch`: grid (rows), block 1024.
fn argmax(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    logits: DevicePtr,
    out: DevicePtr,
    rows: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([rows, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(out)
        .arg_u32(V)
        .arg_u32(V)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn run_leg(
    g: &dyn GpuBackend,
    leg: &str,
    iters: usize,
    kt: KernelHandle,
    kt64: KernelHandle,
    km128: Option<KernelHandle>,
    karg: KernelHandle,
    s_dec: u64,
    s_arg: u64,
    s_pre: Option<u64>,
    c_logits_rows: u32,
) -> Result<()> {
    // Serve-order allocation: twins at "model load", arena after.
    let w_lm = twin(g, V, H)?;
    let w_qkvz = twin(g, QKVZ_N, H)?;
    let w_outp = twin(g, OUTP_N, OUTP_K)?;
    let w_ffn = twin(g, FFN_N, H)?;
    let a_norm = g.alloc(64 * H as usize * 2)?;
    let c_ssm = g.alloc(64 * QKVZ_N as usize * 2)?;
    let c_out = g.alloc(64 * OUTP_N as usize * 2)?;
    let c_logits = g.alloc(c_logits_rows as usize * V as usize * 2)?;
    let a_pre = g.alloc(PRE_M as usize * H as usize * 2)?;
    let c_pre = g.alloc(PRE_M as usize * FFN_N as usize * 2)?;
    let amax_out = g.alloc(256)?;
    for (p, b) in [
        (a_norm, 64 * H as usize * 2),
        (a_pre, PRE_M as usize * H as usize * 2),
    ] {
        g.memset(p, 0x5A, b)?;
    }

    let mut host = [0u8; 64];
    let r = (|| -> Result<()> {
        for it in 0..iters {
            let m = LADDER[it % LADDER.len()];
            // lm_head kernel alternates _t / _t_k64 (serve failed with both)
            let klm = if it % 2 == 0 { kt } else { kt64 };
            if let (Some(kp), Some(sp)) = (km128, s_pre) {
                // prefill leg: m128 tile GEMM + the memsets prefill issues
                g.memset_async(c_pre, 0, PRE_M as usize * FFN_N as usize * 2, sp)?;
                KernelLaunch::new(g, kp)
                    .grid([div_ceil(FFN_N, 128), div_ceil(PRE_M, 128), 1])
                    .block([128, 1, 1])
                    .arg_ptr(a_pre)
                    .arg_ptr(w_ffn.b)
                    .arg_ptr(w_ffn.bs)
                    .arg_f32(1.0)
                    .arg_ptr(c_pre)
                    .arg_u32(PRE_M)
                    .arg_u32(FFN_N)
                    .arg_u32(H)
                    .launch(sp)?;
            }
            // decode step: a few ssm-shaped tile GEMMs, then the lm_head GEMM
            for _ in 0..4 {
                gemm_t(g, kt, a_norm, &w_qkvz, c_ssm, m, QKVZ_N, H, s_dec)?;
                gemm_t(g, kt64, a_norm, &w_outp, c_out, m, OUTP_N, OUTP_K, s_dec)?;
            }
            gemm_t(g, klm, a_norm, &w_lm, c_logits, m, V, H, s_dec)?;
            argmax(g, karg, c_logits, amax_out, m, s_arg)?;
            // 4*m-byte D2H — the exact call that reports 716 in the serve
            g.copy_d2h(amax_out, &mut host[..4 * m as usize])?;
        }
        g.synchronize(s_dec)?;
        g.synchronize(s_arg)?;
        if let Some(sp) = s_pre {
            g.synchronize(sp)?;
        }
        Ok(())
    })();
    match &r {
        Ok(()) => eprintln!("leg {leg}: PASS ({iters} iters)"),
        Err(e) => eprintln!("leg {leg}: FAIL — {e}"),
    }
    for p in [
        w_lm.b, w_lm.bs, w_qkvz.b, w_qkvz.bs, w_outp.b, w_outp.bs, w_ffn.b, w_ffn.bs, a_norm,
        c_ssm, c_out, c_logits, a_pre, c_pre, amax_out,
    ] {
        let _ = g.free(p);
    }
    r
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let iters: usize = std::env::var("ATLAS_REPRO_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let only = std::env::var("ATLAS_REPRO_LEG").ok();

    let kt = g.kernel("w4a16", "w4a16_gemm_t")?;
    let kt64 = g.kernel("w4a16", "w4a16_gemm_t_k64")?;
    let km128 = g.kernel("w4a16", "w4a16_gemm_t_m128").ok();
    let karg = g.kernel("argmax", "argmax_bf16_batch")?;
    let s_a = g.create_stream()?;
    let s_b = g.create_stream()?;
    let s_c = g.create_stream()?;

    let legs: &[(&str, u64, u64, Option<u64>, u32)] = &[
        ("1-solo-null-stream", 0, 0, None, 32),
        ("2-same-stream", s_a, s_a, None, 32),
        ("3-cross-stream", s_a, s_b, None, 32),
        ("4-conc-prefill", s_a, s_b, Some(s_c), 32),
        ("5-undersized-C", s_a, s_a, None, 4),
    ];
    for &(name, sd, sa, sp, crows) in legs {
        if let Some(ref o) = only {
            if !name.starts_with(o.as_str()) {
                continue;
            }
        }
        if run_leg(g, name, iters, kt, kt64, km128, karg, sd, sa, sp, crows).is_err() {
            eprintln!("context may be sticky-dead after a fault; stopping here.");
            break;
        }
    }
    Ok(())
}
