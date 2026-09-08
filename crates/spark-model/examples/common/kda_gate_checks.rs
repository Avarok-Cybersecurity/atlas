// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `kda_gate_microtest.rs` to keep it under the 500-LoC cap.
//! Test-only harness code; no serving path runs any of it.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Result, bail};
use half::bf16;
use serde_json::Value;
use spark_model::layers::glm5next_kda_ref::{KdaDims, bounded_gate};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// A. + B. — fp32 and bf16 kernels vs the HF production-geometry golden (H=64, D=128, T=2).
pub(crate) fn check_production(
    g: &dyn GpuBackend,
    kf: KernelHandle,
    kb: KernelHandle,
) -> Result<bool> {
    let v: Value = serde_json::from_str(&PROD_GOLDEN)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    assert_eq!(
        (h, d),
        (PROD_H, PROD_D),
        "golden is not production geometry"
    );
    let lb = f["lower_bound"].as_f64().unwrap() as f32;

    let g_raw = json_arr(&v, "inputs", "g_raw");
    let dt_bias = json_arr(&v, "inputs", "dt_bias");
    let a_log = json_arr(&v, "inputs", "A_log");
    let want = json_arr(&v, "outputs", "gate");
    assert_eq!(dt_bias.len(), h * d, "dt_bias must be per-channel");
    assert_eq!(a_log.len(), h, "A_log must be per-head");

    // Oracle floor, GPU not involved: how far apart are the Rust CPU reference and HF on this
    // same tensor? This is the irreducible libm spread the GPU result is measured against.
    let cpu_ref = bounded_gate(
        &g_raw,
        &dt_bias,
        &a_log,
        KdaDims {
            hidden: 0,
            heads: h,
            head_dim: d,
            tokens: t,
        },
        lb,
    );
    let floor = compare(&cpu_ref, &want);
    report("oracle floor: CPU reference vs HF (no GPU)", &floor, "f32");

    let got = run_f32(g, kf, &g_raw, &dt_bias, &a_log, t, h, d, lb)?;
    let e = compare(&got, &want);
    report(&format!("production H={h} D={d} T={t} vs HF"), &e, "f32");

    // bf16 entry point. Its error floor is the bf16 rounding of `g_raw`, so it is scored
    // against the golden recomputed from bf16-rounded input, not against the fp32 golden.
    let n = t * h * d;
    let (dg, db, da) = (
        up_bf16(g, &g_raw)?,
        up_f32(g, &dt_bias)?,
        up_f32(g, &a_log)?,
    );
    let out = g.alloc(n * 4)?;
    launch_gate(g, kb, dg, db, da, out, t, h, d, lb)?;
    g.synchronize(0)?;
    let got_bf16 = down_f32(g, out, n)?;

    let g_rounded: Vec<f32> = g_raw.iter().map(|x| bf16::from_f32(*x).to_f32()).collect();
    let want_bf16 = bounded_gate(
        &g_rounded,
        &dt_bias,
        &a_log,
        KdaDims {
            hidden: 0,
            heads: h,
            head_dim: d,
            tokens: t,
        },
        lb,
    );
    let eb = compare(&got_bf16, &want_bf16);
    report("production bf16 vs bf16-rounded ref", &eb, "bf16");
    let eb_vs_hf = compare(&got_bf16, &want);
    report(
        "production bf16 vs fp32 HF (input-rounding floor)",
        &eb_vs_hf,
        "bf16",
    );

    // The kernel's error against HF must sit in the same band as the CPU reference's own.
    let ratio = if floor.max_abs > 0.0 {
        e.max_abs / floor.max_abs
    } else {
        1.0
    };
    let rel_ratio = if floor.max_rel > 0.0 {
        e.max_rel / floor.max_rel
    } else {
        1.0
    };
    println!(
        "  GPU-vs-HF / CPU-ref-vs-HF ratio             max_abs={ratio:.3} max_rel={rel_ratio:.3} (bound {MAX_FLOOR_RATIO})"
    );
    Ok(within(&e) && within(&eb) && ratio <= MAX_FLOOR_RATIO && rel_ratio <= MAX_FLOOR_RATIO)
}

/// C. — boundary behaviour.
pub(crate) fn check_boundaries(g: &dyn GpuBackend, k: KernelHandle) -> Result<bool> {
    let mut rng = Lcg(0xB0173);
    let mut ok = true;
    let dims = KdaDims {
        hidden: 0,
        heads: PROD_H,
        head_dim: PROD_D,
        tokens: 4,
    };
    let n = 4 * PROD_H * PROD_D;

    // C1 — `lower_bound` is read from the argument, not compiled in as -5.
    // The gate is exactly linear in `lower_bound`, so a hardcoded -5 is detectable as a
    // proportionality violation as well as a mismatch against the reference.
    let g_raw = rng.vec(n);
    let dt_bias = rng.vec(PROD_H * PROD_D);
    let a_log: Vec<f32> = (0..PROD_H).map(|h| 0.4 - 0.9 * h as f32 / 63.0).collect();
    let base = run_f32(g, k, &g_raw, &dt_bias, &a_log, 4, PROD_H, PROD_D, -5.0)?;
    for &lb in &[-3.25f32, -1.0, -12.5, -0.25] {
        let got = run_f32(g, k, &g_raw, &dt_bias, &a_log, 4, PROD_H, PROD_D, lb)?;
        let want = bounded_gate(&g_raw, &dt_bias, &a_log, dims, lb);
        let e = compare(&got, &want);
        report(&format!("lower_bound={lb:<6} vs CPU reference"), &e, "f32");
        ok &= within(&e);

        let scale = lb / -5.0;
        let prop = base
            .iter()
            .zip(&got)
            .map(|(b, x)| ((*b as f64) * scale as f64 - *x as f64).abs())
            .fold(0.0f64, f64::max);
        if prop > 1e-6 {
            println!("    ! lower_bound={lb} not proportional to the -5.0 run: {prop:.3e}");
            ok = false;
        }
        if got.iter().any(|v| *v < lb - 1e-6 || *v > 1e-30) {
            println!("    ! lower_bound={lb} produced a value outside [{lb}, 0]");
            ok = false;
        }
    }

    // C2 — per-channel `dt_bias`. A kernel that broadcast one bias per head would produce
    // identical output for a constant-per-head bias and a varying-per-channel one.
    let flat: Vec<f32> = (0..PROD_H)
        .flat_map(|h| std::iter::repeat_n(dt_bias[h * PROD_D], PROD_D))
        .collect();
    let varying = run_f32(g, k, &g_raw, &dt_bias, &a_log, 4, PROD_H, PROD_D, -5.0)?;
    let constant = run_f32(g, k, &g_raw, &flat, &a_log, 4, PROD_H, PROD_D, -5.0)?;
    let spread = compare(&varying, &constant).max_abs;
    println!(
        "  per-channel vs per-head-broadcast dt_bias   divergence={spread:.3e} (must be large)"
    );
    if spread < 0.1 {
        println!("    ! dt_bias channel axis appears collapsed");
        ok = false;
    }

    // C3 — distinct `A_log` per head. Same argument on the head axis.
    let same_a = vec![a_log[0]; PROD_H];
    let distinct = run_f32(g, k, &g_raw, &dt_bias, &a_log, 4, PROD_H, PROD_D, -5.0)?;
    let uniform = run_f32(g, k, &g_raw, &dt_bias, &same_a, 4, PROD_H, PROD_D, -5.0)?;
    let spread_a = compare(&distinct, &uniform).max_abs;
    println!(
        "  distinct vs uniform A_log                   divergence={spread_a:.3e} (must be large)"
    );
    if spread_a < 0.1 {
        println!("    ! A_log head axis appears collapsed");
        ok = false;
    }

    // C4 — sigmoid saturation, both tails, plus values that overflow `exp`.
    let extremes: [f32; 8] = [-1.0e4, -800.0, -80.0, -1.0, 1.0, 80.0, 800.0, 1.0e4];
    let sat_n = PROD_H * PROD_D;
    let g_sat: Vec<f32> = (0..sat_n).map(|i| extremes[i % 8]).collect();
    let zero_bias = vec![0.0f32; PROD_H * PROD_D];
    let ones_a = vec![0.0f32; PROD_H]; // decay = exp(0) = 1
    let got = run_f32(g, k, &g_sat, &zero_bias, &ones_a, 1, PROD_H, PROD_D, -5.0)?;
    let want = bounded_gate(
        &g_sat,
        &zero_bias,
        &ones_a,
        KdaDims {
            hidden: 0,
            heads: PROD_H,
            head_dim: PROD_D,
            tokens: 1,
        },
        -5.0,
    );
    let e = compare(&got, &want);
    report("saturation sweep vs CPU reference", &e, "f32");
    ok &= within(&e);
    if got.iter().any(|v| !v.is_finite()) {
        println!("    ! saturation produced a non-finite value");
        ok = false;
    }
    let hit_lb = got.iter().filter(|v| **v <= -5.0 + 1e-6).count();
    let hit_zero = got.iter().filter(|v| v.abs() <= 1e-30).count();
    println!(
        "  saturation coverage                         at lower_bound={hit_lb}/{sat_n} at zero={hit_zero}/{sat_n}"
    );
    if hit_lb == 0 || hit_zero == 0 {
        println!("    ! saturation sweep did not reach both tails");
        ok = false;
    }
    Ok(ok)
}
