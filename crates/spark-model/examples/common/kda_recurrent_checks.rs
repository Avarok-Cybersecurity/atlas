// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `kda_recurrent_microtest.rs` to keep it under the 500-LoC cap.
//! Test-only harness code; no serving path runs any of it.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Result, bail};
use half::bf16;
use serde_json::Value;
use spark_model::layers::glm5next_kda_ref::{KdaDims, kda_recurrent_prenorm};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// A (non-zero carried state) + B (per-step output AND state) at H=64, D=128, against HF.
pub(crate) fn check_production(rec: &Rec, bf16_inputs: bool) -> Result<bool> {
    let v: Value = serde_json::from_str(&PROD)?;
    let f = &v["fixture"];
    let (h, d, steps) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["steps"].as_u64().unwrap() as usize,
    );
    assert_eq!(
        (h, d),
        (PROD_H, PROD_D),
        "golden is not production geometry"
    );
    let stride = f["sample_stride"].as_u64().unwrap() as usize;
    let lb = f["lower_bound"].as_f64().unwrap() as f32;

    // The generator's LCG must be reproduced exactly or the whole fixture is different.
    let probe: Vec<f32> = v["lcg_probe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let mut pr = Lcg(0x5EED_2EC0);
    let mine = pr.vec(probe.len());
    if mine
        .iter()
        .zip(&probe)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        println!("    ! LCG mismatch: rust {mine:?} vs python {probe:?}");
        return Ok(false);
    }
    println!(
        "  LCG reproduction: {} draws bit-identical to the generator",
        probe.len()
    );

    // Same draw order as the generator.
    let mut rng = Lcg(0x5EED_2EC0);
    let state0: Vec<f32> = rng.vec(h * d * d).iter().map(|x| x * 0.05).collect();
    let q_s: Vec<Vec<f32>> = (0..steps).map(|_| rng.vec(h * d)).collect();
    let k_s: Vec<Vec<f32>> = (0..steps).map(|_| rng.vec(h * d)).collect();
    let v_s: Vec<Vec<f32>> = (0..steps).map(|_| rng.vec(h * d)).collect();
    let g_s: Vec<Vec<f32>> = (0..steps)
        .map(|_| {
            rng.vec(h * d)
                .iter()
                .map(|x| lb * (1.0 / (1.0 + (-(x * 3.0)).exp())))
                .collect()
        })
        .collect();
    let b_s: Vec<Vec<f32>> = (0..steps)
        .map(|_| {
            rng.vec(h)
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect()
        })
        .collect();

    let l2 = |x: &[f32]| -> Vec<f32> {
        x.chunks_exact(d)
            .flat_map(|r| {
                let inv = 1.0 / (r.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
                r.iter().map(move |a| a * inv)
            })
            .collect()
    };

    let dims1 = KdaDims {
        hidden: 0,
        heads: h,
        head_dim: d,
        tokens: 1,
    };
    let mut state = state0.clone();
    let mut cpu_state = state0.clone();
    let mut ok = true;
    let dtype = if bf16_inputs { "bf16" } else { "f32" };

    // The CPU reference must see EXACTLY the inputs the GPU sees, or the comparison mixes
    // input rounding into what is supposed to be kernel error.
    let as_seen = |x: &[f32]| -> Vec<f32> {
        if bf16_inputs {
            x.iter().map(|a| bf16::from_f32(*a).to_f32()).collect()
        } else {
            x.to_vec()
        }
    };

    for s in 0..steps {
        let (qn, kn) = (l2(&q_s[s]), l2(&k_s[s]));
        let o = rec.step(
            &qn,
            &kn,
            &v_s[s],
            &g_s[s],
            &b_s[s],
            &mut state,
            h,
            d,
            bf16_inputs,
        )?;
        let cpu_o = kda_recurrent_prenorm(
            &as_seen(&qn),
            &as_seen(&kn),
            &as_seen(&v_s[s]),
            &g_s[s],
            &b_s[s],
            dims1,
            &mut cpu_state,
        );

        let want_o = arr(&v, "outputs", &format!("o_step{s}"));
        let floor = compare(&cpu_o, &want_o);
        let e = compare(&o, &want_o);
        let kern = compare(&o, &cpu_o);
        report(
            &format!("prod step{s}: CPU-ref vs HF (floor, no GPU)"),
            &floor,
            dtype,
        );
        report(
            &format!("prod step{s}: GPU o vs HF (floor + kernel)"),
            &e,
            dtype,
        );
        report(
            &format!("prod step{s}: GPU o vs CPU-ref, SAME inputs (kernel only)"),
            &kern,
            dtype,
        );
        let state_kern = compare(&state, &cpu_state);
        report(
            &format!("prod step{s}: GPU state vs CPU-ref state (kernel only)"),
            &state_kern,
            dtype,
        );

        let want_samp = arr(&v, "outputs", &format!("state_sample{s}"));
        let samp: Vec<f32> = state.iter().step_by(stride).copied().collect();
        let es = compare(&samp, &want_samp);
        report(&format!("prod step{s}: state sample vs HF"), &es, dtype);

        let want_ck = v["state_checksums"][s].as_f64().unwrap();
        let got_ck = checksum(&state);
        let ck_rel = (got_ck - want_ck).abs() / want_ck.abs().max(1.0);
        println!(
            "  prod step{s}: full-state fp64 checksum rel_err={ck_rel:.3e}  (got {got_ck:.6e})"
        );

        let r = ratio(&e, &floor);
        println!("  prod step{s}: GPU/floor max_abs ratio = {r:.3}");
        if bf16_inputs {
            // Against the fp32 golden the bf16 path is dominated by INPUT ROUNDING, so that
            // figure is reported, not gated. What IS gated is kernel-only error: given the
            // same bf16-rounded inputs, the GPU must track the CPU reference to fp32 accuracy.
            ok &= within(&kern) && within(&state_kern);
        } else {
            ok &= within(&e) && within(&es) && r <= MAX_FLOOR_RATIO && ck_rel < 1e-6;
        }
    }
    Ok(ok)
}

/// D — prove each semantic independently by breaking it and showing the result moves.
pub(crate) fn check_adversarial(rec: &Rec) -> Result<bool> {
    let (h, d) = (8usize, 32usize);
    let dims1 = KdaDims {
        hidden: 0,
        heads: h,
        head_dim: d,
        tokens: 1,
    };
    let mut rng = Lcg(0xD00D);
    let n = h * d;
    let l2 = |x: &[f32]| -> Vec<f32> {
        x.chunks_exact(d)
            .flat_map(|r| {
                let inv = 1.0 / (r.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
                r.iter().map(move |a| a * inv)
            })
            .collect()
    };
    let q = l2(&rng.vec(n));
    let k = l2(&rng.vec(n));
    let vv = rng.vec(n);
    let gate: Vec<f32> = rng
        .vec(n)
        .iter()
        .map(|x| -5.0 * (1.0 / (1.0 + (-(x * 3.0)).exp())))
        .collect();
    let beta: Vec<f32> = rng
        .vec(h)
        .iter()
        .map(|x| 1.0 / (1.0 + (-x).exp()))
        .collect();
    let state0: Vec<f32> = rng.vec(h * d * d).iter().map(|x| x * 0.1).collect();

    let run = |gate: &[f32], beta: &[f32], st: &Vec<f32>| -> Result<(Vec<f32>, Vec<f32>)> {
        let mut s = st.clone();
        let o = rec.step(&q, &k, &vv, gate, beta, &mut s, h, d, false)?;
        Ok((o, s))
    };
    let (base_o, base_s) = run(&gate, &beta, &state0)?;
    let mut ok = true;
    let mut probe = |label: &str, o: &[f32], s: &[f32], need: bool| {
        let dv = compare(o, &base_o).max_abs.max(compare(s, &base_s).max_abs);
        let verdict = if (dv > 1e-4) == need { "ok" } else { "FAIL" };
        println!("  {label:<52} divergence={dv:.3e}  [{verdict}]");
        if (dv > 1e-4) != need {
            ok = false;
        }
    };

    // D1 — decay varies across the key channel D. Flatten it per head and the result moves.
    let flat_gate: Vec<f32> = (0..h)
        .flat_map(|hh| std::iter::repeat_n(gate[hh * d], d))
        .collect();
    let (o, s) = run(&flat_gate, &beta, &state0)?;
    probe(
        "D1 decay is per key-channel (vs per-head flat)",
        &o,
        &s,
        true,
    );

    // D2 — beta varies by head.
    let flat_beta = vec![beta[0]; h];
    let (o, s) = run(&gate, &flat_beta, &state0)?;
    probe("D2 beta is per head (vs uniform)", &o, &s, true);

    // D3 — the state update is k (x) delta, not delta (x) k. Transposing every head's state
    // block would be a no-op if the orientation were symmetric. It is not.
    let mut t_state = state0.clone();
    for hh in 0..h {
        for a in 0..d {
            for b in (a + 1)..d {
                t_state.swap(hh * d * d + a * d + b, hh * d * d + b * d + a);
            }
        }
    }
    let (o, s) = run(&gate, &beta, &t_state)?;
    probe(
        "D3 k (x) delta orientation (vs transposed state)",
        &o,
        &s,
        true,
    );

    // D4 — q is applied AFTER the state update. Compare against a reference that reads the
    // pre-update state; they must differ, and the kernel must match the post-update one.
    let mut ref_state = state0.clone();
    let ref_o = kda_recurrent_prenorm(&q, &k, &vv, &gate, &beta, dims1, &mut ref_state);
    let post = compare(&base_o, &ref_o);
    let mut pre_state = state0.clone();
    let pre_o = {
        // Same recurrence but reading the DECAYED-only state for the output.
        let mut o = vec![0.0f32; h * d];
        for hh in 0..h {
            let s = &mut pre_state[hh * d * d..(hh + 1) * d * d];
            for kk in 0..d {
                let dec = gate[hh * d + kk].exp();
                for vi in 0..d {
                    s[kk * d + vi] *= dec;
                }
            }
            let scale = 1.0f32 / (d as f32).sqrt();
            for vi in 0..d {
                let mut acc = 0.0f32;
                for kk in 0..d {
                    acc += s[kk * d + vi] * (q[hh * d + kk] * scale);
                }
                o[hh * d + vi] = acc;
            }
        }
        o
    };
    let pre = compare(&base_o, &pre_o);
    println!(
        "  D4 q applied AFTER update: vs post-update ref max_abs={:.3e}, vs pre-update ref max_abs={:.3e}",
        post.max_abs, pre.max_abs
    );
    if !(post.max_abs < 1e-6 && pre.max_abs > 1e-4) {
        println!(
            "    ! q ordering is not distinguishable, or does not match the post-update reference"
        );
        ok = false;
    }

    // D5 — 1/sqrt(D) placement. The kernel pre-scales q (HF's `query = query * scale`); the
    // CPU reference scales each term at the output. Algebraically identical, so the residual
    // here is pure rounding and must be tiny.
    println!(
        "  D5 scale placement (pre-scale q vs per-term at output) max_abs={:.3e}",
        post.max_abs
    );
    if post.max_abs > 1e-6 {
        println!("    ! scale placement is not numerically equivalent");
        ok = false;
    }
    Ok(ok)
}
