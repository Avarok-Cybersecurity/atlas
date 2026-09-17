// SPDX-License-Identifier: AGPL-3.0-only
//! GEMM / real-tensor / mgemm legs of the native EXL3 parity gate.

use anyhow::Result;
use half::f16;
use spark_model::layers::ops::exl3_mgemm;

use crate::truth::{cb_enum, decode_what_f64, materialized_bf16_f64, truth_dense, truth_matmul};
use crate::util::{
    Ctx, DevWeight, GEMM_MAX_Z, GEMM_REL_RMS, GEMV_MAX_Z, GEMV_REL_RMS, Lcg, as_bytes, down_f32,
    gate_leg, metrics, run_pipeline, up,
};

pub fn gen_weight(
    rng: &mut Lcg,
    k: usize,
    n: usize,
    k_bits: u32,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let trellis: Vec<u16> = (0..(k / 16) * (n / 16) * 16 * k_bits as usize)
        .map(|_| rng.u16())
        .collect();
    let suh: Vec<u16> = (0..k).map(|_| rng.scale_f16()).collect();
    let svh: Vec<u16> = (0..n).map(|_| rng.scale_f16()).collect();
    (trellis, suh, svh)
}

/// GEMM numeric oracle: K sweep, the two map shapes, forced shapes 3/4, an
/// fp16-C leg (gated loose), determinism, negative controls, calibration.
pub fn leg_gemm(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let mut ok = true;

    // K sweep at [32 x 256 x 128] (heuristic: shape 1 for K in {2,4}, else 2).
    for k_bits in [2u32, 3, 4, 5, 6, 8] {
        let (m, k, n) = (32usize, 256usize, 128usize);
        let (trellis, suh, svh) = gen_weight(rng, k, n, k_bits);
        let a: Vec<u16> = (0..m * k).map(|_| rng.act_f16()).collect();
        let what = decode_what_f64(&trellis, k, n, k_bits, cb_enum(2));
        let y64 = truth_matmul(&a, &suh, &svh, &what, m, k, n, 1.0);
        let w = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
        let out = run_pipeline(ctx, &a, &w, m, k, n, k_bits, 2, true, None, None)?;
        w.free(ctx.g);
        ok &= gate_leg(
            &format!("gemm [32x256x128] K={k_bits} cb2 f32"),
            &out.y,
            &y64,
            GEMM_REL_RMS,
            GEMM_MAX_Z,
        );
    }

    // Map shape [256 x 128 x 128] K=4 (heuristic -> shape 1) + negative
    // controls at the same shape.
    {
        let (m, k, n) = (256usize, 128usize, 128usize);
        let (mut trellis, suh, svh) = gen_weight(rng, k, n, 4);
        let a: Vec<u16> = (0..m * k).map(|_| rng.act_f16()).collect();
        let what = decode_what_f64(&trellis, k, n, 4, cb_enum(2));
        let y64 = truth_matmul(&a, &suh, &svh, &what, m, k, n, 1.0);
        let w = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
        let out = run_pipeline(ctx, &a, &w, m, k, n, 4, 2, true, None, None)?;
        ok &= gate_leg(
            "gemm [256x128x128] K=4 cb2 f32 (shape1)",
            &out.y,
            &y64,
            GEMM_REL_RMS,
            GEMM_MAX_Z,
        );

        // Determinism: same launch twice must be bit-identical.
        let out2 = run_pipeline(ctx, &a, &w, m, k, n, 4, 2, true, None, None)?;
        let det = out.c_bytes == out2.c_bytes;
        println!("gemm determinism (two launches bit-identical) = {det}");
        ok &= det;

        // CONTROL 1: wrong codebook (decode cb2 data with the cb1 kernel)
        // must blow the gate.
        let bad = run_pipeline(ctx, &a, &w, m, k, n, 4, 1, true, None, None)?;
        let (rr, _) = metrics(&bad.y, &y64);
        let moved = rr > GEMM_REL_RMS;
        println!("gemm CONTROL wrong-codebook rel_rms={rr:.3e} exceeds gate = {moved}");
        w.free(ctx.g);

        // CONTROL 2: corrupt one whole tile (16*K words) must blow the gate.
        for word in trellis.iter_mut().take(16 * 4) {
            *word = 0xFFFF;
        }
        let wbad = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
        let bad2 = run_pipeline(ctx, &a, &wbad, m, k, n, 4, 2, true, None, None)?;
        wbad.free(ctx.g);
        let (rr2, _) = metrics(&bad2.y, &y64);
        let moved2 = rr2 > GEMM_REL_RMS;
        println!("gemm CONTROL corrupted-tile rel_rms={rr2:.3e} exceeds gate = {moved2}");
        if !(moved && moved2) {
            println!("FAIL — a negative control stayed under the gate; harness is VACUOUS.");
            return Ok(false);
        }
    }

    // Map shape [512 x 2560 x 640] K=4 (heuristic -> shape 2), with the
    // materialized-path calibration backstop, plus an fp16-C arm (loose
    // gate: fp16 split-k handoffs).
    {
        let (m, k, n) = (512usize, 2560usize, 640usize);
        let (trellis, suh, svh) = gen_weight(rng, k, n, 4);
        let a: Vec<u16> = (0..m * k).map(|_| rng.act_f16()).collect();
        let what = decode_what_f64(&trellis, k, n, 4, cb_enum(2));
        let y64 = truth_matmul(&a, &suh, &svh, &what, m, k, n, 1.0);
        let w = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
        let out = run_pipeline(ctx, &a, &w, m, k, n, 4, 2, true, None, None)?;
        ok &= gate_leg(
            "gemm [512x2560x640] K=4 cb2 f32 (shape2)",
            &out.y,
            &y64,
            GEMM_REL_RMS,
            GEMM_MAX_Z,
        );

        let out16 = run_pipeline(ctx, &a, &w, m, k, n, 4, 2, false, None, None)?;
        ok &= gate_leg(
            "gemm [512x2560x640] K=4 cb2 f16-C (loose gate)",
            &out16.y,
            &y64,
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
        w.free(ctx.g);

        let w_mat = materialized_bf16_f64(&trellis, &suh, &svh, k, n, 4, cb_enum(2));
        let y_mat = truth_dense(&a, &w_mat, m, k, n);
        let (err_native, _) = metrics(&out.y, &y64);
        let (err_mat, _) = metrics(&y_mat, &y64);
        let cal = err_native <= 2.0 * err_mat + 1e-3;
        println!(
            "calibration: err_native={err_native:.3e} vs err_bf16_materialized={err_mat:.3e}  (native <= 2*mat + 1e-3) = {cal}"
        );
        ok &= cal;
    }

    // Forced shapes 3 and 4 (need n % 256 / n % 512).
    for (shape, n) in [(3usize, 768usize), (4, 1024)] {
        let (m, k) = (64usize, 2560usize);
        let (trellis, suh, svh) = gen_weight(rng, k, n, 4);
        let a: Vec<u16> = (0..m * k).map(|_| rng.act_f16()).collect();
        let what = decode_what_f64(&trellis, k, n, 4, cb_enum(2));
        let y64 = truth_matmul(&a, &suh, &svh, &what, m, k, n, 1.0);
        let w = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
        let out = run_pipeline(ctx, &a, &w, m, k, n, 4, 2, true, None, Some(shape))?;
        w.free(ctx.g);
        ok &= gate_leg(
            &format!("gemm [64x2560x{n}] K=4 cb2 f32 (forced shape{shape})"),
            &out.y,
            &y64,
            GEMM_REL_RMS,
            GEMM_MAX_Z,
        );
    }

    Ok(ok)
}

/// REAL-tensor leg: qwen4_exp expert [2560 x 640] K=4 MUL1 through gemv
/// (m=1, both cfgs) and gemm (m=64), vs the f64 truth, plus calibration.
pub fn leg_real(ctx: &Ctx, rng: &mut Lcg) -> Result<Option<bool>> {
    let dir = std::env::var("EXL3_REAL_DIR").unwrap_or_else(|_| ".research/real_tensor".into());
    let read_u16 = |name: &str| -> Result<Vec<u16>> {
        let b = std::fs::read(format!("{dir}/{name}"))?;
        Ok(b.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    };
    let Ok(meta) = std::fs::read_to_string(format!("{dir}/meta.txt")) else {
        println!("(real tensor dir {dir} absent — REAL leg skipped)");
        return Ok(None);
    };
    let mut k_dim = 0usize;
    let mut n_dim = 0usize;
    let mut k_bits = 0u32;
    let mut cb = 2u32;
    for line in meta.lines() {
        let (key, val) = line.split_once('=').unwrap_or(("", ""));
        match key {
            "in" => k_dim = val.parse()?,
            "out" => n_dim = val.parse()?,
            "k" => k_bits = val.parse()?,
            "cb" => cb = val.parse()?,
            _ => {}
        }
    }
    let trellis = read_u16("trellis.bin")?;
    let suh = read_u16("suh.bin")?;
    let svh = read_u16("svh.bin")?;
    println!("REAL tensor {dir}: [{k_dim} x {n_dim}] K={k_bits} cb={cb}");

    let what = decode_what_f64(&trellis, k_dim, n_dim, k_bits, cb_enum(cb));
    let w = DevWeight::upload(ctx.g, &trellis, &suh, &svh)?;
    let mut ok = true;

    // gemv m=1, both configs.
    let a1: Vec<u16> = (0..k_dim).map(|_| rng.act_f16()).collect();
    let y64_1 = truth_matmul(&a1, &suh, &svh, &what, 1, k_dim, n_dim, 1.0);
    for cfg in [0u32, 1] {
        let out = run_pipeline(
            ctx,
            &a1,
            &w,
            1,
            k_dim,
            n_dim,
            k_bits,
            cb,
            true,
            Some(cfg),
            None,
        )?;
        ok &= gate_leg(
            &format!("REAL gemv m=1 cfg{cfg} f32"),
            &out.y,
            &y64_1,
            GEMV_REL_RMS,
            GEMV_MAX_Z,
        );
    }

    // gemm m=64 + calibration backstop.
    let m = 64usize;
    let a: Vec<u16> = (0..m * k_dim).map(|_| rng.act_f16()).collect();
    let y64 = truth_matmul(&a, &suh, &svh, &what, m, k_dim, n_dim, 1.0);
    let out = run_pipeline(ctx, &a, &w, m, k_dim, n_dim, k_bits, cb, true, None, None)?;
    ok &= gate_leg("REAL gemm m=64 f32", &out.y, &y64, GEMM_REL_RMS, GEMM_MAX_Z);
    w.free(ctx.g);

    let w_mat = materialized_bf16_f64(&trellis, &suh, &svh, k_dim, n_dim, k_bits, cb_enum(cb));
    let y_mat = truth_dense(&a, &w_mat, m, k_dim, n_dim);
    let (err_native, _) = metrics(&out.y, &y64);
    let (err_mat, _) = metrics(&y_mat, &y64);
    let cal = err_native <= 2.0 * err_mat + 1e-3;
    println!(
        "REAL calibration: err_native={err_native:.3e} vs err_bf16_materialized={err_mat:.3e} = {cal}"
    );
    ok &= cal;
    Ok(Some(ok))
}

/// mgemm smoke: 4 synthetic experts [2560 -> 640] K=4 cb2, weighted routing
/// for num_tokens in {1, 2}, fp32 C, vs f64 weighted per-expert truth.
pub fn leg_mgemm(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    leg_mgemm_k(ctx, rng, 4)
}

/// [`leg_mgemm`] at an arbitrary K (the widened-envelope legs run 3/5/6).
pub fn leg_mgemm_k(ctx: &Ctx, rng: &mut Lcg, k_bits: u32) -> Result<bool> {
    let g = ctx.g;
    let (k, n) = (2560usize, 640usize);
    let cb = 2u32;
    let num_experts = 4usize;

    let mut experts = Vec::new();
    for _ in 0..num_experts {
        experts.push(gen_weight(rng, k, n, k_bits));
    }
    let whats: Vec<Vec<f64>> = experts
        .iter()
        .map(|(t, _, _)| decode_what_f64(t, k, n, k_bits, cb_enum(cb)))
        .collect();
    let dev: Vec<DevWeight> = experts
        .iter()
        .map(|(t, su, sv)| DevWeight::upload(g, t, su, sv))
        .collect::<Result<_>>()?;
    let ptr_bytes = |f: &dyn Fn(&DevWeight) -> u64| -> Vec<u8> {
        dev.iter().flat_map(|d| f(d).to_le_bytes()).collect()
    };
    let b_list = up(g, &ptr_bytes(&|d| d.trellis.0))?;
    let suh_list = up(g, &ptr_bytes(&|d| d.suh.0))?;
    let svh_list = up(g, &ptr_bytes(&|d| d.svh.0))?;
    let stream = g.default_stream();
    let mut ok = true;

    for num_tokens in [1usize, 2] {
        let bszm = 4usize; // 4 slots; stride = bszm / num_tokens experts per token
        let stride = bszm / num_tokens;
        let m = 1usize;
        let indices: Vec<i64> = vec![0, 2, 1, 3];
        let weights_f: Vec<f32> = vec![0.6, 0.4, 0.3, 0.7];
        let weights_h: Vec<u16> = weights_f
            .iter()
            .map(|&x| f16::from_f32(x).to_bits())
            .collect();

        // bszm_in = num_tokens slabs? The kernel broadcasts only when
        // bszm_in == 1; otherwise slot j reads A + j*m*k. Duplicate each
        // token's activation across its slot run.
        let (bszm_in, a_bits): (usize, Vec<u16>) = if num_tokens == 1 {
            let x: Vec<u16> = (0..m * k).map(|_| rng.act_f16()).collect();
            (1, x)
        } else {
            let toks: Vec<Vec<u16>> = (0..num_tokens)
                .map(|_| (0..m * k).map(|_| rng.act_f16()).collect())
                .collect();
            let mut all = Vec::with_capacity(bszm * m * k);
            for slot in 0..bszm {
                all.extend_from_slice(&toks[slot / stride]);
            }
            (bszm, all)
        };

        // f64 truth: token t = sum over its slots of w_j * pipeline(x_t, e_j).
        let mut y64 = vec![0f64; num_tokens * m * n];
        for slot in 0..bszm {
            let t = slot / stride;
            let e = indices[slot] as usize;
            let x = if bszm_in == 1 {
                &a_bits[0..m * k]
            } else {
                &a_bits[slot * m * k..(slot + 1) * m * k]
            };
            let (_, su, sv) = &experts[e];
            let w_eff = f16::from_bits(weights_h[slot]).to_f64();
            let y = truth_matmul(x, su, sv, &whats[e], m, k, n, w_eff);
            for (acc, v) in y64[t * m * n..(t + 1) * m * n].iter_mut().zip(y.iter()) {
                *acc += v;
            }
        }

        let a_d = up(g, &as_bytes(&a_bits))?;
        let a_had_elems = bszm * m * k; // HARD requirement: bszm*m*k halves
        let a_had_d = g.alloc(a_had_elems * 2)?;
        let c_d = g.alloc(bszm * m * n * 4)?;
        let idx_d = up(
            g,
            &indices
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        let wts_d = up(g, &as_bytes(&weights_h))?;

        exl3_mgemm(
            g,
            a_d,
            b_list,
            c_d,
            m,
            k,
            n,
            k_bits,
            cb,
            true,
            ctx.locks,
            suh_list,
            a_had_d,
            a_had_elems,
            svh_list,
            Some(idx_d),
            Some(wts_d),
            bszm_in,
            bszm, // bszm_out: C carries one scratch slab per slot
            -1,
            -1,
            num_tokens,
            None,
            None,
            None,
            None,
            ctx.sms,
            stream,
        )?;
        g.synchronize(stream)?;
        let y_gpu: Vec<f64> = down_f32(g, c_d, num_tokens * m * n)?
            .iter()
            .map(|&v| v as f64)
            .collect();
        for p in [a_d, a_had_d, c_d, idx_d, wts_d] {
            g.free(p).ok();
        }
        ok &= gate_leg(
            &format!("mgemm 4 experts [2560x640] K={k_bits} cb2 f32 num_tokens={num_tokens}"),
            &y_gpu,
            &y64,
            GEMM_REL_RMS,
            GEMM_MAX_Z,
        );
    }

    for d in &dev {
        d.free(g);
    }
    for p in [b_list, suh_list, svh_list] {
        g.free(p).ok();
    }
    Ok(ok)
}
