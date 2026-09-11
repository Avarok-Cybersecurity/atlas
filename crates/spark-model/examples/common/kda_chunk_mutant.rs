// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `kda_chunk_microtest.rs` to keep it under the 500-LoC cap.
//! Test-only harness code; no serving path runs any of it.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Result, bail};
use serde_json::Value;
use spark_model::layers::glm5next_kda_ref::{KdaDims, kda_chunked, kda_recurrent_prenorm};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

/// Deliberately-wrong variants of the chunk formulation. Each corresponds to a hazard the
/// instruction named; the point is to show the test suite SEPARATES them from the correct
/// path rather than merely asserting the correct path passes.
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Mutation {
    None,
    /// decay collapsed to one value per head instead of per key-channel
    PerHeadDecay,
    /// `exp(gc[j]-gc[i])` instead of `exp(gc[i]-gc[j])`
    SignFlip,
    /// intra mask keeps `j >= i` (drops the diagonal) — transposed triangle orientation
    TriangleFlip,
    /// state updated BEFORE the output is read instead of after
    BoundaryOrder,
}

/// Compact chunked prefill with an injectable defect. Correct mode is checked against the
/// Slice-2 reference `kda_chunked`, so a drift here cannot silently weaken the sensitivity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mutant_chunk(
    i: &Inputs,
    t: usize,
    h: usize,
    d: usize,
    c: usize,
    state: &mut [f32],
    m: Mutation,
) -> Vec<f32> {
    let nchunks = t.div_ceil(c);
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut out = vec![0.0f32; t * h * d];
    let at = |buf: &[f32], tok: usize, hh: usize, dd: usize| -> f32 {
        if tok >= t {
            0.0
        } else {
            buf[(tok * h + hh) * d + dd]
        }
    };
    let bat = |tok: usize, hh: usize| -> f32 { if tok >= t { 0.0 } else { i.beta[tok * h + hh] } };

    for hh in 0..h {
        let s = &mut state[hh * d * d..(hh + 1) * d * d];
        for ch in 0..nchunks {
            let off = ch * c;
            let mut gc = vec![0.0f32; c * d];
            for dd in 0..d {
                let mut acc = 0.0f32;
                for p in 0..c {
                    acc += at(&i.gate, off + p, hh, dd);
                    gc[p * d + dd] = acc;
                }
            }
            if m == Mutation::PerHeadDecay {
                for p in 0..c {
                    let v0 = gc[p * d];
                    for dd in 0..d {
                        gc[p * d + dd] = v0;
                    }
                }
            }
            let dm = |a: usize, b: usize, dd: usize| -> f32 {
                if m == Mutation::SignFlip {
                    (gc[b * d + dd] - gc[a * d + dd]).exp()
                } else {
                    (gc[a * d + dd] - gc[b * d + dd]).exp()
                }
            };

            let mut aa = vec![0.0f32; c * c];
            for p in 0..c {
                for j in 0..p {
                    let mut acc = 0.0f32;
                    for dd in 0..d {
                        acc += at(&i.k, off + p, hh, dd)
                            * bat(off + p, hh)
                            * at(&i.k, off + j, hh, dd)
                            * dm(p, j, dd);
                    }
                    aa[p * c + j] = -acc;
                }
            }
            for p in 1..c {
                let row: Vec<f32> = (0..p).map(|j| aa[p * c + j]).collect();
                for j in 0..p {
                    let mut acc = 0.0f32;
                    for (mm, r) in row.iter().enumerate() {
                        acc += r * aa[mm * c + j];
                    }
                    aa[p * c + j] = row[j] + acc;
                }
            }
            for p in 0..c {
                aa[p * c + p] = 1.0;
            }

            let mut u = vec![0.0f32; c * d];
            let mut w = vec![0.0f32; c * d];
            for p in 0..c {
                for dd in 0..d {
                    let (mut au, mut aw) = (0.0f32, 0.0f32);
                    for j in 0..=p {
                        let a = aa[p * c + j];
                        au += a * at(&i.v, off + j, hh, dd) * bat(off + j, hh);
                        aw +=
                            a * at(&i.k, off + j, hh, dd) * bat(off + j, hh) * gc[j * d + dd].exp();
                    }
                    u[p * d + dd] = au;
                    w[p * d + dd] = aw;
                }
            }

            let mut vnew = vec![0.0f32; c * d];
            for p in 0..c {
                for vi in 0..d {
                    let mut acc = 0.0f32;
                    for kk in 0..d {
                        acc += w[p * d + kk] * s[kk * d + vi];
                    }
                    vnew[p * d + vi] = u[p * d + vi] - acc;
                }
            }

            let update_state = |s: &mut [f32], gc: &[f32], vnew: &[f32]| {
                for kk in 0..d {
                    let gl = gc[(c - 1) * d + kk];
                    for vi in 0..d {
                        let mut acc = s[kk * d + vi] * gl.exp();
                        for p in 0..c {
                            acc += at(&i.k, off + p, hh, kk)
                                * (gl - gc[p * d + kk]).exp()
                                * vnew[p * d + vi];
                        }
                        s[kk * d + vi] = acc;
                    }
                }
            };
            if m == Mutation::BoundaryOrder {
                update_state(s, &gc, &vnew);
            }

            for p in 0..c {
                if off + p >= t {
                    continue;
                }
                for vi in 0..d {
                    let mut acc = 0.0f32;
                    for kk in 0..d {
                        acc += at(&i.q, off + p, hh, kk)
                            * scale
                            * gc[p * d + kk].exp()
                            * s[kk * d + vi];
                    }
                    let keep_hi = m == Mutation::TriangleFlip;
                    for j in 0..c {
                        let keep = if keep_hi { j >= p } else { j <= p };
                        if !keep {
                            continue;
                        }
                        let mut intra = 0.0f32;
                        for dd in 0..d {
                            intra += at(&i.q, off + p, hh, dd)
                                * scale
                                * at(&i.k, off + j, hh, dd)
                                * dm(p, j, dd);
                        }
                        acc += intra * vnew[j * d + vi];
                    }
                    out[((off + p) * h + hh) * d + vi] = acc;
                }
            }

            if m != Mutation::BoundaryOrder {
                update_state(s, &gc, &vnew);
            }
        }
    }
    out
}

/// D — adversarial. Each mutation must move the answer; the correct mode must not.
pub(crate) fn test_d(gpu: &Gpu) -> Result<bool> {
    let (h, d, t, c) = (8usize, 32usize, 10usize, 4usize);
    let mut rng = Lcg(0xADDE_5EED);
    let n = t * h * d;
    let ins = Inputs {
        q: l2_rows(&rng.vec(n), d),
        k: l2_rows(&rng.vec(n), d),
        v: rng.vec(n),
        gate: rng
            .vec(n)
            .iter()
            .map(|x| -5.0 * (1.0 / (1.0 + (-(x * 3.0)).exp())))
            .collect(),
        beta: rng
            .vec(t * h)
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect(),
    };
    let s0: Vec<f32> = rng.vec(h * d * d).iter().map(|x| x * 0.05).collect();
    let mut sg = s0.clone();
    let gpu_o = gpu.chunk(&ins, t, h, d, c, &mut sg, 0.0)?;
    let mut ok = true;

    for (label, m, must_move) in [
        ("D0 correct mutant mode matches GPU", Mutation::None, false),
        (
            "D1 decay collapsed to per-head",
            Mutation::PerHeadDecay,
            true,
        ),
        (
            "D2 exp(gc[j]-gc[i]) sign reversal",
            Mutation::SignFlip,
            true,
        ),
        (
            "D3 triangle orientation flipped",
            Mutation::TriangleFlip,
            true,
        ),
        (
            "D4 state updated before output",
            Mutation::BoundaryOrder,
            true,
        ),
    ] {
        let mut st = s0.clone();
        let mo = mutant_chunk(&ins, t, h, d, c, &mut st, m);
        let e = compare(&gpu_o, &mo);
        let moved = e.max_abs > 1e-4;
        let verdict = if moved == must_move { "ok" } else { "FAIL" };
        println!("  {label:<44} max_abs={:.3e}  [{verdict}]", e.max_abs);
        if moved != must_move {
            ok = false;
        }
    }

    // D5 — padded tail cannot reach a valid output or the final state. T=10, chunk=4 leaves
    // 2 pad positions; fill them with large garbage instead of zeros.
    let mut s_clean = s0.clone();
    let o_clean = gpu.chunk(&ins, t, h, d, c, &mut s_clean, 0.0)?;
    let mut s_dirty = s0.clone();
    let o_dirty = gpu.chunk(&ins, t, h, d, c, &mut s_dirty, 7.5)?;
    let eo = compare(&o_dirty, &o_clean);
    let es = compare(&s_dirty, &s_clean);
    println!(
        "  D5 poisoned pad tail (fill=7.5) out max_abs={:.3e} state max_abs={:.3e}  [{}]",
        eo.max_abs,
        es.max_abs,
        if eo.max_abs == 0.0 && es.max_abs == 0.0 {
            "ok"
        } else {
            "FAIL"
        }
    );
    if eo.max_abs != 0.0 || es.max_abs != 0.0 {
        ok = false;
    }

    // D6 — off-by-one on the final chunk: T and T-1 must agree on the first T-1 outputs.
    let mut s_full = s0.clone();
    let o_full = gpu.chunk(&ins, t, h, d, c, &mut s_full, 0.0)?;
    let mut s_short = s0.clone();
    let o_short = gpu.chunk(&ins, t - 1, h, d, c, &mut s_short, 0.0)?;
    let e = compare(&o_short, &o_full[..(t - 1) * h * d]);
    println!(
        "  D6 T-1 prefix matches T prefix                max_abs={:.3e}  [{}]",
        e.max_abs,
        if within(&e) { "ok" } else { "FAIL" }
    );
    ok &= within(&e);
    Ok(ok)
}

/// Isolated latency, correctness having already passed. Baseline only, no optimisation.
pub(crate) fn latency(gpu: &Gpu) -> Result<()> {
    let (h, d) = (PROD_H, PROD_D);
    let mut rng = Lcg(0x1A7E);
    for &(t, c) in &[(512usize, 32usize), (1024, 32), (2048, 32)] {
        let n = t * h * d;
        let ins = Inputs {
            q: l2_rows(&rng.vec(n), d),
            k: l2_rows(&rng.vec(n), d),
            v: rng.vec(n),
            gate: rng
                .vec(n)
                .iter()
                .map(|x| -5.0 * (1.0 / (1.0 + (-(x * 3.0)).exp())))
                .collect(),
            beta: rng
                .vec(t * h)
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect(),
        };
        let s0 = vec![0.0f32; h * d * d];
        let mut st = s0.clone();
        gpu.chunk(&ins, t, h, d, c, &mut st, 0.0)?; // warm + end-to-end sanity

        // Isolated KERNEL time: upload once, then time only the two launches. The end-to-end
        // figure above is dominated by ~0.6 GB of host<->device traffic per call, which a real
        // layer would never pay -- q/k/v/gate arrive already resident from the conv and gate.
        let g = gpu.g;
        let nchunks = t.div_ceil(c);
        let tp = nchunks * c;
        let pad = |src: &[f32], per: usize| -> Vec<f32> {
            let mut o = vec![0.0f32; tp * per];
            o[..t * per].copy_from_slice(&src[..t * per]);
            o
        };
        let (dq, dk, dv, dg) = (
            up_f32(g, &pad(&ins.q, h * d))?,
            up_f32(g, &pad(&ins.k, h * d))?,
            up_f32(g, &pad(&ins.v, h * d))?,
            up_f32(g, &pad(&ins.gate, h * d))?,
        );
        let db = up_f32(g, &pad(&ins.beta, h))?;
        let n = tp * h * d;
        let (dgc, du, dw) = (g.alloc(n * 4)?, g.alloc(n * 4)?, g.alloc(n * 4)?);
        let dout = g.alloc(n * 4)?;
        let dstate = up_f32(g, &s0)?;
        let (sp, ss) = (smem_prepare(c, d), smem_scan(c, d));

        let launch = |which: u8| -> Result<()> {
            if which == 0 {
                KernelLaunch::new(g, gpu.prepare)
                    .grid([nchunks as u32, h as u32, 1])
                    .block([BLOCK, 1, 1])
                    .shared_mem(sp as u32)
                    .arg_ptr(dk)
                    .arg_ptr(dv)
                    .arg_ptr(dg)
                    .arg_ptr(db)
                    .arg_ptr(dgc)
                    .arg_ptr(du)
                    .arg_ptr(dw)
                    .arg_u32(h as u32)
                    .arg_u32(d as u32)
                    .arg_u32(c as u32)
                    .arg_u32(t as u32)
                    .launch(0)?;
            } else {
                KernelLaunch::new(g, gpu.scan)
                    .grid([h as u32, 1, 1])
                    .block([BLOCK, 1, 1])
                    .shared_mem(ss as u32)
                    .arg_ptr(dq)
                    .arg_ptr(dk)
                    .arg_ptr(dgc)
                    .arg_ptr(du)
                    .arg_ptr(dw)
                    .arg_ptr(dstate)
                    .arg_ptr(dout)
                    .arg_u32(h as u32)
                    .arg_u32(d as u32)
                    .arg_u32(c as u32)
                    .arg_u32(nchunks as u32)
                    .arg_u32(t as u32)
                    .arg_f32(1.0 / (d as f32).sqrt())
                    .launch(0)?;
            }
            g.synchronize(0)
        };
        launch(0)?;
        launch(1)?;
        let reps = 5;
        let t0 = Instant::now();
        for _ in 0..reps {
            launch(0)?;
        }
        let ms_prep = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let t1 = Instant::now();
        for _ in 0..reps {
            launch(1)?;
        }
        let ms_scan = t1.elapsed().as_secs_f64() * 1000.0 / reps as f64;
        let ms = ms_prep + ms_scan;
        println!(
            "  T={t:<5} chunk={c}  prepare {ms_prep:7.2} ms + scan {ms_scan:7.2} ms = {ms:7.2} ms  ({:.3} ms/token, ONE KDA layer, kernels only)",
            ms / t as f64
        );
    }
    Ok(())
}
