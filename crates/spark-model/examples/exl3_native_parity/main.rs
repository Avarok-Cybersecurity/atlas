// SPDX-License-Identifier: AGPL-3.0-only
//! Parity + bench gate for the NATIVE EXL3 (QTIP trellis) matmul path
//! (module `exl3_matmul`: fused-rotation gemm/mgemm/gemv over packed
//! trellis codes — no weight reconstruction).
//!
//! Legs:
//!  A. rotation-EXACT: the fused kernels' A_had scratch (input rotation:
//!     fp16 suh pre-scale, fp32 H128, one terminal fp16 rounding) must be
//!     BIT-IDENTICAL to a CPU replica of `had_hf_r_128_inner`'s op order.
//!  B. GEMV numeric oracle (m <= 8, fp16 MMA accumulation): vs a CPU f64
//!     truth built from the parity-proven `cpu_ref::decode_tile`; gates
//!     rel_rms <= 8e-3, max_z <= 4e-2 (map derivation).
//!  C. GEMM numeric oracle (fp32 MMA accumulation): K sweep 2..8, all four
//!     tile shapes, fp32/fp16 C; gates rel_rms <= 2.5e-3, max_z <= 1.5e-2;
//!     plus determinism, negative controls (wrong codebook / corrupted
//!     tile must BLOW the gate) and a bf16-materialized calibration
//!     backstop (err_native <= 2*err_materialized + 1e-3).
//!  D. REAL tensor ([2560x640] K=4 MUL1 qwen4_exp expert) through gemv and
//!     gemm. E. mgemm smoke: 4 experts, weighted token routing.
//!  F. decode-MoE: the PRODUCTION 3x-mgemm routed-expert pipeline
//!     (stage/replicate/gate/up/silu/down with folded probs) — 8 experts
//!     K=4 MUL1 at real geometry, T in {1,4,8}, top_k=3, vs a
//!     reconstruct->f64->silu->weighted-sum reference; EP sub-leg with
//!     remote experts masked -1 + exact-zero all-remote token + negative
//!     control (legs_moe.rs).
//!  G. prefill-MoE: the PRODUCTION sort-by-expert tier (Atlas counting sort
//!     + exl3_moe_stage_sorted + the fused persistent exl3_moe kernel + the
//!     reconstruct overflow path) — 16 experts K=4 MUL1, top_k=4, T in
//!     {3 (no-sync shortcut), 64 (host-sync fused), 64-EP (sentinel tail +
//!     exact-zero + control), 192 skewed (overflow >128 rows, asserted via
//!     stats)} (legs_moe_prefill.rs).
//!
//! With EXL3_BENCH=1 also times gemv/gemm at qwen4_exp decode/prefill
//! shapes (20 warmup + 200 timed launches, us/launch).
//!
//! Exit: 0 pass, 1 any leg failed or a control went vacuous, 2 kernels
//! absent from this target's module set.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-flash-next \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example exl3_native_parity

mod bench;
mod legs;
mod legs_moe;
mod legs_moe_prefill;
mod legs_moe_prefill_debug;
mod truth;
mod util;

use anyhow::Result;
use spark_model::layers::ops::exl3_locks_alloc;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::GpuBackend;

use crate::truth::{cb_enum, decode_what_f64, exact_a_had, truth_matmul};
use crate::util::{Ctx, DevWeight, GEMV_MAX_Z, GEMV_REL_RMS, Lcg, gate_leg, run_pipeline};

/// Leg A: bit-exact input rotation, via the A_had scratch each fused kernel
/// writes before its matmul stages (A_had never aliases A here).
fn leg_rotation_exact(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let mut ok = true;
    // (label, m, k, n, gemv_cfg)
    let cases: [(&str, usize, usize, usize, Option<u32>); 3] = [
        ("gemv m=1 [2560->640] cfg0", 1, 2560, 640, Some(0)),
        ("gemv m=8 [2560->640] cfg1", 8, 2560, 640, Some(1)),
        ("gemm m=17 [128->128] (shape1)", 17, 128, 128, None),
    ];
    for (label, m, k, n, cfg) in cases {
        let trellis: Vec<u16> = (0..(k / 16) * (n / 16) * 16 * 4)
            .map(|_| rng.u16())
            .collect();
        let suh: Vec<u16> = (0..k).map(|_| rng.scale_f16()).collect();
        let svh: Vec<u16> = (0..n).map(|_| rng.scale_f16()).collect();
        let a: Vec<u16> = (0..m * k).map(|_| rng.act_f16()).collect();
        let w = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
        let out = run_pipeline(ctx, &a, &w, m, k, n, 4, 2, true, cfg, None)?;
        w.free(ctx.g);
        let cpu = exact_a_had(&a, &suh, m, k);
        let diffs = out
            .a_had
            .iter()
            .zip(cpu.iter())
            .filter(|(x, y)| x != y)
            .count();
        let identical = diffs == 0;
        println!(
            "A_had exact {label}: bit-identical={identical} (diff {diffs}/{})",
            m * k
        );
        ok &= identical;
    }
    Ok(ok)
}

/// Leg B: GEMV numeric oracle, all (K, cb) x m x cfg envelopes at the
/// qwen4_exp expert shape, fp32 C (+ one fp16-C arm + determinism).
fn leg_gemv(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let mut ok = true;
    let (k, n) = (2560usize, 640usize);
    for (k_bits, cb) in [(2u32, 2u32), (3, 2), (4, 2), (4, 1)] {
        let trellis: Vec<u16> = (0..(k / 16) * (n / 16) * 16 * k_bits as usize)
            .map(|_| rng.u16())
            .collect();
        let suh: Vec<u16> = (0..k).map(|_| rng.scale_f16()).collect();
        let svh: Vec<u16> = (0..n).map(|_| rng.scale_f16()).collect();
        let a8: Vec<u16> = (0..8 * k).map(|_| rng.act_f16()).collect();
        let what = decode_what_f64(&trellis, k, n, k_bits, cb_enum(cb));
        let y64_8 = truth_matmul(&a8, &suh, &svh, &what, 8, k, n, 1.0);
        let w = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
        for m in [1usize, 4, 8] {
            for cfg in [0u32, 1] {
                let out = run_pipeline(
                    ctx,
                    &a8[..m * k],
                    &w,
                    m,
                    k,
                    n,
                    k_bits,
                    cb,
                    true,
                    Some(cfg),
                    None,
                )?;
                ok &= gate_leg(
                    &format!("gemv [2560x640] K={k_bits} cb{cb} m={m} cfg{cfg} f32"),
                    &out.y,
                    &y64_8[..m * n],
                    GEMV_REL_RMS,
                    GEMV_MAX_Z,
                );
            }
        }
        if (k_bits, cb) == (4, 2) {
            // fp16-C arm + determinism at the served config.
            let out = run_pipeline(ctx, &a8[..k], &w, 1, k, n, 4, 2, false, Some(0), None)?;
            ok &= gate_leg(
                "gemv [2560x640] K=4 cb2 m=1 cfg0 f16-C",
                &out.y,
                &y64_8[..n],
                GEMV_REL_RMS,
                GEMV_MAX_Z,
            );
            let d1 = run_pipeline(ctx, &a8, &w, 8, k, n, 4, 2, true, Some(0), None)?;
            let d2 = run_pipeline(ctx, &a8, &w, 8, k, n, 4, 2, true, Some(0), None)?;
            let det = d1.c_bytes == d2.c_bytes;
            println!("gemv determinism (two launches bit-identical) = {det}");
            ok &= det;
        }
        w.free(ctx.g);
    }
    Ok(ok)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    // Probe: absent = this target set doesn't carry the module.
    if g.kernel("exl3_matmul", "exl3_gemm_k4_cb2_sh2_f32").is_err() {
        println!("exl3_matmul kernels absent from this target set — SKIP");
        std::process::exit(2);
    }

    let locks = exl3_locks_alloc(g)?;
    let sms = g.sm_count()?;
    println!("exl3 native matmul parity — sm_count={sms}, locks buffer zeroed once");
    let ctx = Ctx { g, locks, sms };
    let mut rng = Lcg(0x5EED_D06E);

    if std::env::var("EXL3_PF_DEBUG").as_deref() == Ok("1") {
        legs_moe_prefill_debug::debug_pf(&ctx, &mut rng)?;
        std::process::exit(3);
    }

    let mut clean = true;
    clean &= leg_rotation_exact(&ctx, &mut rng)?;
    clean &= leg_gemv(&ctx, &mut rng)?;
    clean &= legs::leg_gemm(&ctx, &mut rng)?;
    match legs::leg_real(&ctx, &mut rng) {
        Ok(Some(ok)) => clean &= ok,
        Ok(None) => {}
        Err(e) => {
            println!("REAL leg FAILED to run: {e:#}");
            clean = false;
        }
    }
    clean &= legs::leg_mgemm(&ctx, &mut rng)?;
    clean &= legs_moe::leg_moe_decode(&ctx, &mut rng)?;
    clean &= legs_moe_prefill::leg_moe_prefill(&ctx, &mut rng)?;

    if std::env::var("EXL3_BENCH").as_deref() == Ok("1") {
        bench::run(&ctx, &mut rng)?;
    }

    if clean {
        println!("PASS — native EXL3 matmul matches the CPU truth at every leg.");
        Ok(())
    } else {
        println!("FAIL — native EXL3 matmul diverged (or a control went vacuous).");
        std::process::exit(1);
    }
}
