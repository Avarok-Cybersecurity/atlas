// SPDX-License-Identifier: AGPL-3.0-only
//! Widened-K legs ("K ladder"): every arm the native EXL3 path serves at the
//! higher-bpw branches of turboderp/Qwen3.8-Flash-Next-exl3 — 3.05 (experts
//! K=3, dense K=5), 4.05 (experts K=4, dense/lm_head K=6), 5.05 (experts
//! K=5) and 6.05 (experts K=6, dense K=8) — exercised at REAL qwen4_exp
//! shapes against the same reconstruct->f64 references the K=4 legs use:
//!
//!  1. gemm at K in {3,5,6,8}: GDN qkv [2560->10240], out [6144->2560],
//!     expert gate [2560->640] and down [640->2560] (m=64, f32 C; the K=6
//!     qkv shape also in f16 C).
//!  2. mgemm weighted routing at K in {3,5,6} (`legs::leg_mgemm_k`).
//!  3. the PRODUCTION 3x-mgemm decode pipeline at K in {5,6}, T in {1,8}.
//!  4. the PRODUCTION fused `exl3_moe` prefill tier at K in {5,6}: T=3
//!     (no-sync shortcut) and T=64 (host-sync fused, asserted fused-only) —
//!     the new fixed-K k5/k6 instances.
//!  5. lm_head-style gemm [2560 -> 248320] K=6 at m=1 (f32 C — the
//!     `project_single_fp32` fallthrough; K=6 has no GEMV) and m=64 (f16 C —
//!     `project`), checked on four 128-column blocks (first, second, middle,
//!     last) so the f64 truth stays at [k x 512] instead of 5 GB.
//!  6. the PRODUCTION `exl3_dense_linear` at the GDN/attention shapes with
//!     K=6, m in {1, 8, 64} — m<=8 must land on the f32-C GEMM tier.
//!
//! Gates: GEMM legs at the GEMM pair (rel 2.5e-3 / z 1.5e-2), f16-C and
//! bf16-egress legs at the GEMV pair (rel 8e-3 / z 4e-2), MoE pipelines at
//! the legs_moe pair (rel 8e-3 / z 8e-2). Higher K decodes FINER weights
//! (smaller quantization step), so these gates are if anything conservative
//! relative to the K=4 derivation.

use std::sync::Arc;

use anyhow::Result;
use half::{bf16, f16};
use spark_model::layers::ops::{
    Exl3DenseOut, Exl3DenseStage, Exl3DenseWeight, Exl3LaunchState, Exl3MoeOverflowCtx,
    exl3_dense_linear,
};

use crate::legs::{gen_weight, leg_mgemm_k};
use crate::legs_moe::{self, ProjSet, ref_token};
use crate::legs_moe_prefill;
use crate::truth::{cb_enum, decode_what_f64, truth_matmul};
use crate::util::{
    Ctx, DevWeight, GEMM_MAX_Z, GEMM_REL_RMS, GEMV_MAX_Z, GEMV_REL_RMS, Lcg, as_bytes, gate_leg,
    run_pipeline, up,
};

const MOE_REL_RMS: f64 = 8e-3;
const MOE_MAX_Z: f64 = 8e-2;
const CB: u32 = 2;

/// Leg 1: plain gemm at real shapes, K in {3,5,6,8}.
fn leg_gemm_shapes(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let mut ok = true;
    let shapes: [(usize, usize, &str); 4] = [
        (2560, 10240, "gdn qkv"),
        (6144, 2560, "gdn out / attn o"),
        (2560, 640, "expert gate/up"),
        (640, 2560, "expert down"),
    ];
    let m = 64usize;
    for k_bits in [3u32, 5, 6, 8] {
        for (k, n, label) in shapes {
            let (trellis, suh, svh) = gen_weight(rng, k, n, k_bits);
            let a: Vec<u16> = (0..m * k).map(|_| rng.act_f16()).collect();
            let what = decode_what_f64(&trellis, k, n, k_bits, cb_enum(CB));
            let y64 = truth_matmul(&a, &suh, &svh, &what, m, k, n, 1.0);
            let w = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
            let out = run_pipeline(ctx, &a, &w, m, k, n, k_bits, CB, true, None, None)?;
            ok &= gate_leg(
                &format!("kladder gemm [{m}x{k}x{n}] {label} K={k_bits} cb2 f32"),
                &out.y,
                &y64,
                GEMM_REL_RMS,
                GEMM_MAX_Z,
            );
            if k_bits == 6 && n == 10240 {
                let out16 = run_pipeline(ctx, &a, &w, m, k, n, k_bits, CB, false, None, None)?;
                ok &= gate_leg(
                    &format!("kladder gemm [{m}x{k}x{n}] {label} K=6 cb2 f16-C (loose gate)"),
                    &out16.y,
                    &y64,
                    GEMV_REL_RMS,
                    GEMV_MAX_Z,
                );
            }
            w.free(ctx.g);
        }
    }
    Ok(ok)
}

/// Leg 3: production decode pipeline (3x mgemm) at K in {5,6}.
fn leg_moe_decode_k(ctx: &Ctx, rng: &mut Lcg, k_bits: u32) -> Result<bool> {
    const H: usize = 2560;
    const I: usize = 640;
    const E: usize = 8;
    const TOP_K: usize = 3;
    let g = ctx.g;
    let mut ok = true;
    let gate = ProjSet::generate_k(ctx, rng, E, H, I, k_bits)?;
    let upp = ProjSet::generate_k(ctx, rng, E, H, I, k_bits)?;
    let down = ProjSet::generate_k(ctx, rng, E, I, H, k_bits)?;
    let (gate_t, gate_own) = gate.table(ctx, 0)?;
    let (up_t, up_own) = upp.table(ctx, 0)?;
    let (down_t, down_own) = down.table(ctx, 0)?;
    let tables = [gate_t, up_t, down_t];
    let sl = legs_moe::alloc_slabs(ctx, 8 * TOP_K, 8)?;
    for t in [1usize, 8] {
        let s = t * TOP_K;
        let input_bf16: Vec<u16> = (0..t * H)
            .map(|_| bf16::from_f32(rng.gauss()).to_bits())
            .collect();
        let input_f16: Vec<u16> = input_bf16
            .iter()
            .map(|&b| f16::from_f32(bf16::from_bits(b).to_f32()).to_bits())
            .collect();
        let ids: Vec<u32> = (0..s).map(|_| (rng.next() % E as u64) as u32).collect();
        let mut probs: Vec<f32> = (0..s).map(|_| 0.05 + rng.f()).collect();
        for chunk in probs.chunks_mut(TOP_K) {
            let sum: f32 = chunk.iter().sum();
            for v in chunk {
                *v /= sum;
            }
        }
        let (y_gpu, _) =
            legs_moe::run_native(ctx, &sl, &tables, &input_bf16, &ids, &probs, t, 0, E)?;
        let mut y64 = Vec::with_capacity(t * H);
        for tok in 0..t {
            y64.extend(ref_token(
                &input_f16[tok * H..(tok + 1) * H],
                &ids[tok * TOP_K..(tok + 1) * TOP_K],
                &probs[tok * TOP_K..(tok + 1) * TOP_K],
                &gate,
                &upp,
                &down,
                0,
                E,
            ));
        }
        ok &= gate_leg(
            &format!("kladder moe-decode 3x-mgemm [{H}x{I}] K={k_bits} E={E} top_k={TOP_K} T={t}"),
            &y_gpu,
            &y64,
            MOE_REL_RMS,
            MOE_MAX_Z,
        );
    }
    for p in sl.owned.iter() {
        g.free(*p).ok();
    }
    for p in gate_own.into_iter().chain(up_own).chain(down_own) {
        g.free(p).ok();
    }
    gate.free(g);
    upp.free(g);
    down.free(g);
    Ok(ok)
}

/// Leg 4: production fused prefill tier at K in {5,6} (fixed-K instances).
fn leg_moe_prefill_k(ctx: &Ctx, rng: &mut Lcg, k_bits: u32) -> Result<bool> {
    use legs_moe_prefill::{H, I, TOP_K};
    const E: usize = 16;
    let g = ctx.g;
    let mut ok = true;
    let gate = ProjSet::generate_k(ctx, rng, E, H, I, k_bits)?;
    let upp = ProjSet::generate_k(ctx, rng, E, H, I, k_bits)?;
    let down = ProjSet::generate_k(ctx, rng, E, I, H, k_bits)?;
    let (gate_t, gate_own) = gate.table(ctx, 0)?;
    let (up_t, up_own) = upp.table(ctx, 0)?;
    let (down_t, down_own) = down.table(ctx, 0)?;
    let tables = [gate_t, up_t, down_t];
    let host = |p: &ProjSet| -> Vec<[u64; 3]> {
        p.dev
            .iter()
            .map(|d| [d.trellis.0, d.suh.0, d.svh.0])
            .collect()
    };
    let (gh, uh, dh) = (host(&gate), host(&upp), host(&down));
    let ov = Exl3MoeOverflowCtx {
        gate_host: &gh,
        up_host: &uh,
        down_host: &dh,
    };
    let sl = legs_moe_prefill::alloc_slabs(ctx)?;
    for t in [3usize, 64] {
        let s = t * TOP_K;
        let (input_bf16, input_f16, probs) = legs_moe_prefill::gen_inputs(rng, t);
        let ids: Vec<u32> = (0..s).map(|_| (rng.next() % E as u64) as u32).collect();
        let (y_gpu, _, num_active, n_ov) = legs_moe_prefill::run_native(
            ctx,
            &sl,
            &tables,
            &ov,
            &input_bf16,
            &ids,
            &probs,
            t,
            0,
            E,
        )?;
        let y64 = legs_moe_prefill::ref_all(&input_f16, &ids, &probs, &gate, &upp, &down, t, 0, E);
        ok &= gate_leg(
            &format!("kladder moe-prefill fused [{H}x{I}] K={k_bits} E={E} top_k={TOP_K} T={t}"),
            &y_gpu,
            &y64,
            MOE_REL_RMS,
            MOE_MAX_Z,
        );
        let tier_ok = if t == 3 {
            num_active == -1 && n_ov == 0
        } else {
            num_active > 0 && n_ov == 0
        };
        println!(
            "kladder moe-prefill K={k_bits} T={t} tier (num_active={num_active}, \
             overflow={n_ov}) as expected = {tier_ok}"
        );
        ok &= tier_ok;
    }
    for p in sl.owned.iter() {
        g.free(*p).ok();
    }
    for p in gate_own.into_iter().chain(up_own).chain(down_own) {
        g.free(p).ok();
    }
    gate.free(g);
    upp.free(g);
    down.free(g);
    Ok(ok)
}

/// Leg 5: lm_head geometry [2560 -> 248320] at K=6, m=1 (f32 C) and m=64
/// (f16 C), truth on four 128-column blocks.
fn leg_lm_head_k6(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let (k, n, k_bits) = (2560usize, 248_320usize, 6u32);
    let kt = 16 * k_bits as usize;
    let tiles_n = n / 16;
    let (trellis, suh, svh) = gen_weight(rng, k, n, k_bits);
    let w = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
    // Column blocks: first, second, a middle one, the last (pad tail).
    let blocks: [usize; 4] = [0, 1, n / 256, n / 128 - 1];
    let n_red = 128 * blocks.len();
    let mut trellis_red = Vec::with_capacity((k / 16) * (n_red / 16) * kt);
    let mut svh_red = Vec::with_capacity(n_red);
    for tr in 0..k / 16 {
        for &b in &blocks {
            for tc in 8 * b..8 * b + 8 {
                let base = (tr * tiles_n + tc) * kt;
                trellis_red.extend_from_slice(&trellis[base..base + kt]);
            }
        }
    }
    for &b in &blocks {
        svh_red.extend_from_slice(&svh[128 * b..128 * b + 128]);
    }
    let what_red = decode_what_f64(&trellis_red, k, n_red, k_bits, cb_enum(CB));
    let mut ok = true;
    for (m, c_fp32) in [(1usize, true), (64usize, false)] {
        let a: Vec<u16> = (0..m * k).map(|_| rng.act_f16()).collect();
        let y64 = truth_matmul(&a, &suh, &svh_red, &what_red, m, k, n_red, 1.0);
        let out = run_pipeline(ctx, &a, &w, m, k, n, k_bits, CB, c_fp32, None, None)?;
        let mut y_gpu = Vec::with_capacity(m * n_red);
        for r in 0..m {
            for &b in &blocks {
                y_gpu.extend_from_slice(&out.y[r * n + 128 * b..r * n + 128 * b + 128]);
            }
        }
        let (rel, z) = if c_fp32 {
            (GEMM_REL_RMS, GEMM_MAX_Z)
        } else {
            (GEMV_REL_RMS, GEMV_MAX_Z)
        };
        ok &= gate_leg(
            &format!(
                "kladder lm_head gemm [{m}x{k}x{n}] K=6 cb2 {} (blocks {blocks:?})",
                if c_fp32 { "f32" } else { "f16-C" }
            ),
            &y_gpu,
            &y64,
            rel,
            z,
        );
    }
    w.free(ctx.g);
    Ok(ok)
}

/// Leg 6: production dense-linear dispatch at the GDN/attention shapes, K=6.
fn leg_dense_k6(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let g = ctx.g;
    let stream = g.default_stream();
    let k_bits = 6u32;
    let launch = Arc::new(Exl3LaunchState::new(g)?);
    let stage = Exl3DenseStage::new(g, launch.clone(), 256, 6144, 12288)?;
    let shapes: [(usize, usize, &str); 5] = [
        (2560, 10240, "gdn in_proj_qkv"),
        (2560, 6144, "gdn in_proj_z"),
        (6144, 2560, "gdn out_proj / attn o_proj"),
        (2560, 12288, "attn q_proj"),
        (2560, 512, "attn k/v_proj"),
    ];
    let m_max = 64usize;
    let dst = g.alloc(m_max * 12288 * 2)?;
    let mut ok = true;
    for (k, n, label) in shapes {
        let (trellis, suh, svh) = gen_weight(rng, k, n, k_bits);
        let what = decode_what_f64(&trellis, k, n, k_bits, cb_enum(CB));
        let dev = DevWeight::upload(g, &trellis, &suh, &svh)?;
        let w = Exl3DenseWeight {
            trellis: dev.trellis,
            suh: dev.suh,
            svh: dev.svh,
            in_dim: k,
            out_dim: n,
            k_bits,
            cb: CB,
        };
        let a_bf16: Vec<u16> = (0..m_max * k)
            .map(|_| bf16::from_f32(rng.gauss()).to_bits())
            .collect();
        let a_f16: Vec<u16> = a_bf16
            .iter()
            .map(|&b| f16::from_f32(bf16::from_bits(b).to_f32()).to_bits())
            .collect();
        let a_d = up(g, &as_bytes(&a_bf16))?;
        for m in [1usize, 8, 64] {
            exl3_dense_linear(g, &w, a_d, Exl3DenseOut::contiguous(dst), m, &stage, stream)?;
            g.synchronize(stream)?;
            let mut bytes = vec![0u8; m * n * 2];
            g.copy_d2h(dst, &mut bytes)?;
            let y: Vec<f64> = bytes
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
                .collect();
            let y64 = truth_matmul(&a_f16[..m * k], &suh, &svh, &what, m, k, n, 1.0);
            ok &= gate_leg(
                &format!(
                    "kladder dense [{k}->{n}] {label} K=6 m={m} ({})",
                    if m <= 8 {
                        "f32-C GEMM tier, no GEMV at K=6"
                    } else {
                        "f16-C GEMM tier"
                    }
                ),
                &y,
                &y64,
                GEMV_REL_RMS,
                GEMV_MAX_Z,
            );
        }
        g.free(a_d).ok();
        dev.free(g);
    }
    g.free(dst).ok();
    stage.release(g)?;
    launch.release(g)?;
    Ok(ok)
}

pub fn run(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let mut ok = true;
    ok &= leg_gemm_shapes(ctx, rng)?;
    for k_bits in [3u32, 5, 6] {
        ok &= leg_mgemm_k(ctx, rng, k_bits)?;
    }
    for k_bits in [5u32, 6] {
        ok &= leg_moe_decode_k(ctx, rng, k_bits)?;
        ok &= leg_moe_prefill_k(ctx, rng, k_bits)?;
    }
    ok &= leg_lm_head_k6(ctx, rng)?;
    ok &= leg_dense_k6(ctx, rng)?;
    Ok(ok)
}
