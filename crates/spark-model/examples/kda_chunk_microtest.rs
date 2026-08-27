// SPDX-License-Identifier: AGPL-3.0-only
//! Numeric gate for the GLM-5.3-Flash KDA CHUNKED PREFILL kernels
//! (`kda_chunk::kda_chunk_prepare` + `kda_chunk_scan`) against HuggingFace 5.16.1 and against
//! the already-proven Slice-4 decode kernel.
//!
//! ## Why two kernels
//! The verified formulation separates into a per-chunk phase that is fully parallel and a
//! cross-chunk phase that is inherently sequential (it carries the recurrent state). Forcing
//! them into one launch would serialise the parallel half for nothing.
//!
//! ## The load-bearing check
//! `kda_chunk` over `T` tokens must equal `T` sequential `kda_recurrent` decode steps, for the
//! same normalised q/k, v, gate, beta and initial state. That invariant is independent of any
//! golden and it is what actually pins the WY reformulation: a decay-sign flip, a transposed
//! triangle, or a chunk-boundary ordering error all break it while still looking plausible.
//!
//! ## Shared memory is a correctness blocker here
//! Atlas's CUDA backend has no `cuFuncSetAttribute` opt-in, so a block cannot exceed the
//! DEFAULT 48 KiB (49152 B); GB10's 101376 B ceiling is unreachable. Requirements are asserted
//! against 49152 before every launch — see `smem_*`.
//!
//!   cargo run -p spark-model --release --example kda_chunk_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use serde_json::Value;
use spark_model::layers::glm5next_kda_ref::{KdaDims, kda_chunked, kda_recurrent_prenorm};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

const FIXTURE: &str = include_str!("../src/layers/glm5next_kda_ref/kda_golden.json");
#[path = "common/golden.rs"]
mod golden;

static PROD: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| golden::load("crates/spark-model/src/layers/glm5next_kda_ref/kda_chunk_prod_golden.json", "gen_kda_chunk_prod_golden.py"));

const PROD_H: usize = 64;
const PROD_D: usize = 128;
const BLOCK: u32 = 128;

/// Hard ceiling: default dynamic shared memory per block, with no opt-in path in this backend.
const SMEM_CEILING: usize = 49152;

const MAX_ABS: f64 = 2.0e-6;
const MAX_REL: f64 = 1.0e-4;
const MAX_FLOOR_RATIO: f64 = 2.0;

fn smem_prepare(c: usize, d: usize) -> usize {
    (c * d + c * c + c) * 4
}
fn smem_scan(c: usize, d: usize) -> usize {
    (2 * c * d + c * c) * 4
}

// ───────────────────────────────────────────────────────────────── plumbing

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn arr(v: &Value, section: &str, name: &str) -> Vec<f32> {
    v[section][name]["data"]
        .as_array()
        .unwrap_or_else(|| panic!("missing {section}.{name}"))
        .iter()
        .map(|x| x.as_f64().expect("numeric") as f32)
        .collect()
}

#[derive(Default, Clone)]
struct Err2 {
    max_abs: f64,
    max_rel: f64,
    exact: usize,
    total: usize,
}

const REL_GUARD_FRACTION: f64 = 1e-3;

fn compare(got: &[f32], want: &[f32]) -> Err2 {
    assert_eq!(
        got.len(),
        want.len(),
        "length {} vs {}",
        got.len(),
        want.len()
    );
    let mut e = Err2 {
        total: got.len(),
        ..Default::default()
    };
    let peak = want.iter().fold(0.0f64, |m, w| m.max((*w as f64).abs()));
    let guard = (peak * REL_GUARD_FRACTION).max(1e-30);
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64 - *w as f64).abs();
        e.max_abs = e.max_abs.max(d);
        if (*w as f64).abs() > guard {
            e.max_rel = e.max_rel.max(d / (*w as f64).abs());
        }
        if g.to_bits() == w.to_bits() {
            e.exact += 1;
        }
    }
    e
}
fn report(label: &str, e: &Err2) {
    println!(
        "  {label:<52} max_abs={:.3e} max_rel={:.3e} exact={}/{}",
        e.max_abs, e.max_rel, e.exact, e.total
    );
}
fn within(e: &Err2) -> bool {
    e.max_abs <= MAX_ABS && e.max_rel <= MAX_REL
}

/// Chunked prefill and the sequential recurrence are ALGEBRAICALLY equal but numerically
/// distinct algorithms: the WY reformulation accumulates through A, u, w and a per-chunk state
/// jump, the recurrence accumulates token by token. Their difference is a property of the two
/// formulations, not of the GPU — so it is measured on the CPU reference over the identical
/// fixture and the GPU is required to sit inside that, rather than inside a tolerance picked
/// to make the run go green.
fn within_floor(e: &Err2, floor: &Err2) -> bool {
    e.max_abs <= MAX_ABS.max(floor.max_abs * MAX_FLOOR_RATIO)
        && e.max_rel <= MAX_REL.max(floor.max_rel * MAX_FLOOR_RATIO)
}
fn checksum(s: &[f32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}

struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit()).collect()
    }
}

// ───────────────────────────────────────────────────────────────── GPU driver

struct Gpu<'a> {
    g: &'a dyn GpuBackend,
    prepare: KernelHandle,
    scan: KernelHandle,
    recurrent: KernelHandle,
}

/// Inputs laid out `[T, H, D]` (token-major), matching the goldens and the decode kernel.
struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
}

impl Gpu<'_> {
    /// Chunked prefill. `pad_fill` lets a test poison the padded tail to prove it cannot
    /// reach a valid output or the final state.
    fn chunk(
        &self,
        i: &Inputs,
        t: usize,
        h: usize,
        d: usize,
        c: usize,
        state: &mut Vec<f32>,
        pad_fill: f32,
    ) -> Result<Vec<f32>> {
        let g = self.g;
        let nchunks = t.div_ceil(c);
        let tp = nchunks * c;
        let (sp, ss) = (smem_prepare(c, d), smem_scan(c, d));
        assert!(
            sp <= SMEM_CEILING && ss <= SMEM_CEILING,
            "chunk={c} needs {sp}/{ss} B shared, ceiling {SMEM_CEILING}"
        );

        let pad_vec = |src: &[f32], per: usize| -> Vec<f32> {
            let mut o = vec![pad_fill; tp * per];
            o[..t * per].copy_from_slice(&src[..t * per]);
            o
        };
        let (pq, pk, pv, pg) = (
            pad_vec(&i.q, h * d),
            pad_vec(&i.k, h * d),
            pad_vec(&i.v, h * d),
            pad_vec(&i.gate, h * d),
        );
        let pb = pad_vec(&i.beta, h);

        let (dq, dk, dv, dg, db) = (
            up_f32(g, &pq)?,
            up_f32(g, &pk)?,
            up_f32(g, &pv)?,
            up_f32(g, &pg)?,
            up_f32(g, &pb)?,
        );
        let n = tp * h * d;
        let (dgc, du, dw) = (g.alloc(n * 4)?, g.alloc(n * 4)?, g.alloc(n * 4)?);
        let dout = up_f32(g, &vec![0.0f32; n])?;
        let dstate = up_f32(g, state)?;

        KernelLaunch::new(g, self.prepare)
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

        KernelLaunch::new(g, self.scan)
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
        g.synchronize(0)?;

        *state = down_f32(g, dstate, h * d * d)?;
        let full = down_f32(g, dout, n)?;
        Ok(full[..t * h * d].to_vec())
    }

    /// `T` sequential decode steps through the Slice-4 kernel.
    fn recurrent(
        &self,
        i: &Inputs,
        t: usize,
        h: usize,
        d: usize,
        state: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        let g = self.g;
        let per = h * d;
        let mut out = Vec::with_capacity(t * per);
        for tok in 0..t {
            let (a, b) = (tok * per, (tok + 1) * per);
            let (dq, dk, dv) = (
                up_f32(g, &i.q[a..b])?,
                up_f32(g, &i.k[a..b])?,
                up_f32(g, &i.v[a..b])?,
            );
            let dg = up_f32(g, &i.gate[a..b])?;
            let db = up_f32(g, &i.beta[tok * h..(tok + 1) * h])?;
            let dstate = up_f32(g, state)?;
            let dout = g.alloc(per * 4)?;
            KernelLaunch::new(g, self.recurrent)
                .grid([h as u32, 1, 1])
                .block([BLOCK.min(d as u32), 1, 1])
                .shared_mem((3 * d * 4) as u32)
                .arg_ptr(dq)
                .arg_ptr(dk)
                .arg_ptr(dv)
                .arg_ptr(dg)
                .arg_ptr(db)
                .arg_ptr(dstate)
                .arg_ptr(dout)
                .arg_u32(h as u32)
                .arg_u32(d as u32)
                .arg_f32(1.0 / (d as f32).sqrt())
                .launch(0)?;
            g.synchronize(0)?;
            *state = down_f32(g, dstate, h * d * d)?;
            out.extend_from_slice(&down_f32(g, dout, per)?);
        }
        Ok(out)
    }
}

// ───────────────────────────────────────────────── adversarial mutant reference

/// Deliberately-wrong variants of the chunk formulation. Each corresponds to a hazard the
/// instruction named; the point is to show the test suite SEPARATES them from the correct
/// path rather than merely asserting the correct path passes.
#[derive(Clone, Copy, PartialEq)]
enum Mutation {
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
fn mutant_chunk(
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

// ───────────────────────────────────────────────────────────────── tests

/// A — HF golden, small fixture, every output token and the final state, at chunk 2 and the
/// padded chunk 4 (T=6, pad 2).
fn test_a(gpu: &Gpu) -> Result<bool> {
    let v: Value = serde_json::from_str(FIXTURE)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    let ins = Inputs {
        q: arr(&v, "outputs", "q_l2"),
        k: arr(&v, "outputs", "k_l2"),
        v: arr(&v, "inputs", "v_in"),
        gate: arr(&v, "outputs", "gate"),
        beta: arr(&v, "outputs", "beta"),
    };
    let mut ok = true;
    for (c, oname, sname) in [
        (2usize, "core_chunked", "state_chunked"),
        (4, "core_chunked_c4_padded", "state_chunked_c4_padded"),
    ] {
        let mut st = vec![0.0f32; h * d * d];
        let o = gpu.chunk(&ins, t, h, d, c, &mut st, 0.0)?;
        let eo = compare(&o, &arr(&v, "outputs", oname));
        let es = compare(&st, &arr(&v, "outputs", sname));
        report(&format!("A chunk={c} T={t} out vs HF"), &eo);
        report(&format!("A chunk={c} T={t} final state vs HF"), &es);
        ok &= within(&eo) && within(&es);
    }
    Ok(ok)
}

/// B — chunk == T sequential decode steps. Independent of any golden.
fn test_b(gpu: &Gpu) -> Result<bool> {
    let (h, d) = (PROD_H, PROD_D);
    let mut rng = Lcg(0xB10C_5EED);
    let mut ok = true;
    println!(
        "  {:<11}{:>6}{:>12}{:>12}{:>12}{:>12}{:>11}",
        "shape", "chunk", "out max_abs", "out max_rel", "st max_abs", "st max_rel", "GPU/floor"
    );
    for &(t, c) in &[
        (5usize, 8usize), // T < chunk
        (16, 16),         // T == chunk
        (17, 16),         // T == chunk + 1
        (64, 16),         // multiple full chunks
        (70, 16),         // ragged tail
        (129, 32),        // ragged, larger chunk
        (256, 32),        // multiple full chunks, production-ish
    ] {
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
        let (mut sc, mut sr) = (s0.clone(), s0.clone());
        let oc = gpu.chunk(&ins, t, h, d, c, &mut sc, 0.0)?;
        let or = gpu.recurrent(&ins, t, h, d, &mut sr)?;

        // Same two formulations on the CPU reference: the algorithmic floor, no GPU involved.
        let dims = KdaDims {
            hidden: 0,
            heads: h,
            head_dim: d,
            tokens: t,
        };
        let mut fc = s0.clone();
        let cpu_chunk = kda_chunked(
            &ins.q, &ins.k, &ins.v, &ins.gate, &ins.beta, dims, c, &mut fc,
        );
        let mut fr = s0.clone();
        let cpu_rec =
            kda_recurrent_prenorm(&ins.q, &ins.k, &ins.v, &ins.gate, &ins.beta, dims, &mut fr);
        let floor_o = compare(&cpu_chunk, &cpu_rec);
        let floor_s = compare(&fc, &fr);

        let eo = compare(&oc, &or);
        let es = compare(&sc, &sr);
        let ratio = if floor_o.max_abs > 0.0 {
            eo.max_abs / floor_o.max_abs
        } else {
            1.0
        };
        let good = within_floor(&eo, &floor_o) && within_floor(&es, &floor_s);
        println!(
            "  T={t:<9}{c:>6}{:>12.3e}{:>12.3e}{:>12.3e}{:>12.3e}{:>11.3}  {}",
            eo.max_abs,
            eo.max_rel,
            es.max_abs,
            es.max_rel,
            ratio,
            if good { "ok" } else { "FAIL" }
        );
        ok &= good;
    }
    Ok(ok)
}

fn l2_rows(x: &[f32], d: usize) -> Vec<f32> {
    x.chunks_exact(d)
        .flat_map(|r| {
            let inv = 1.0 / (r.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
            r.iter().map(move |a| a * inv)
        })
        .collect()
}

/// C — production geometry against the HF golden, at several chunk sizes.
fn test_c(gpu: &Gpu) -> Result<bool> {
    let v: Value = serde_json::from_str(PROD)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    let stride = f["sample_stride"].as_u64().unwrap() as usize;
    let lb = f["lower_bound"].as_f64().unwrap() as f32;
    assert_eq!((h, d), (PROD_H, PROD_D));

    let probe: Vec<f32> = v["lcg_probe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let mut pr = Lcg(0x5EED_C400);
    if pr
        .vec(probe.len())
        .iter()
        .zip(&probe)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        println!("    ! LCG mismatch with the generator");
        return Ok(false);
    }
    let hf_self = v["hf_self_checks"]["chunk_vs_recurrent_max_abs"]
        .as_f64()
        .unwrap();
    println!("  oracle self-consistency: HF chunk vs HF recurrent = {hf_self:.3e}");

    let mut rng = Lcg(0x5EED_C400);
    let s0: Vec<f32> = rng.vec(h * d * d).iter().map(|x| x * 0.05).collect();
    let n = t * h * d;
    let ins = Inputs {
        q: l2_rows(&rng.vec(n), d),
        k: l2_rows(&rng.vec(n), d),
        v: rng.vec(n),
        gate: rng
            .vec(n)
            .iter()
            .map(|x| lb * (1.0 / (1.0 + (-(x * 3.0)).exp())))
            .collect(),
        beta: rng
            .vec(t * h)
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect(),
    };
    let want_o = arr(&v, "outputs", "out");
    let want_s = arr(&v, "outputs", "state_sample");
    let want_ck = v["state_checksum"].as_f64().unwrap();

    // CPU/HF floor, no GPU involved.
    let mut cpu_state = s0.clone();
    let cpu_o = kda_chunked(
        &ins.q,
        &ins.k,
        &ins.v,
        &ins.gate,
        &ins.beta,
        KdaDims {
            hidden: 0,
            heads: h,
            head_dim: d,
            tokens: t,
        },
        2,
        &mut cpu_state,
    );
    let floor = compare(&cpu_o, &want_o);
    report("C floor: CPU-ref vs HF (no GPU)", &floor);

    let mut ok = true;
    for &c in &[2usize, 4, 8, 16, 32] {
        let mut st = s0.clone();
        let o = gpu.chunk(&ins, t, h, d, c, &mut st, 0.0)?;
        let eo = compare(&o, &want_o);
        let samp: Vec<f32> = st.iter().step_by(stride).copied().collect();
        let es = compare(&samp, &want_s);
        let ck = checksum(&st);
        let ck_rel = (ck - want_ck).abs() / want_ck.abs().max(1.0);
        let r = if floor.max_abs > 0.0 {
            eo.max_abs / floor.max_abs
        } else {
            1.0
        };
        report(&format!("C chunk={c} out vs HF"), &eo);
        println!(
            "     state sample max_abs={:.3e}  full-state checksum rel={ck_rel:.3e}  GPU/floor={r:.3}  smem prep/scan={}/{} B",
            es.max_abs,
            smem_prepare(c, d),
            smem_scan(c, d)
        );
        ok &= within(&eo) && within(&es) && ck_rel < 1e-6 && r <= MAX_FLOOR_RATIO;

        // GPU chunk vs GPU recurrent on the same production fixture.
        let mut sr = s0.clone();
        let or = gpu.recurrent(&ins, t, h, d, &mut sr)?;
        let ec = compare(&o, &or);
        if c == 2 {
            let dims = KdaDims {
                hidden: 0,
                heads: h,
                head_dim: d,
                tokens: t,
            };
            let mut fr = s0.clone();
            let cpu_rec =
                kda_recurrent_prenorm(&ins.q, &ins.k, &ins.v, &ins.gate, &ins.beta, dims, &mut fr);
            let floor_cr = compare(&cpu_o, &cpu_rec);
            report("C floor: CPU chunk vs CPU recurrent", &floor_cr);
            report("C chunk vs recurrent-kernel path", &ec);
            ok &= within_floor(&ec, &floor_cr);
        }
    }
    Ok(ok)
}

/// D — adversarial. Each mutation must move the answer; the correct mode must not.
fn test_d(gpu: &Gpu) -> Result<bool> {
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
fn latency(gpu: &Gpu) -> Result<()> {
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

// ───────────────────────────────────────────────────────────────── main

fn main() -> Result<()> {
    let g = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &g;
    let d = Gpu {
        g: gpu,
        prepare: gpu.kernel("kda_chunk", "kda_chunk_prepare")?,
        scan: gpu.kernel("kda_chunk", "kda_chunk_scan")?,
        recurrent: gpu.kernel("kda_recurrent", "kda_recurrent_decode_f32")?,
    };
    println!("kda_chunk: kda_chunk_prepare + kda_chunk_scan resolved from PTX (no fallback)");
    println!(
        "shared-memory ceiling {SMEM_CEILING} B (no cuFuncSetAttribute opt-in in this backend)"
    );
    for c in [8usize, 16, 32, 64] {
        let (p, s) = (smem_prepare(c, PROD_D), smem_scan(c, PROD_D));
        println!(
            "  chunk={c:<3} D=128  prepare={p:>6} B  scan={s:>6} B  {}",
            if p.max(s) <= SMEM_CEILING {
                "ok"
            } else {
                "EXCEEDS -> unusable"
            }
        );
    }

    println!("\nA — HF golden, fixture H=2 D=4 T=6");
    let a = test_a(&d)?;
    println!("\nB — chunk == T sequential decode steps (production H=64 D=128)");
    let b = test_b(&d)?;
    println!("\nC — production geometry vs HF golden");
    let c = test_c(&d)?;
    println!("\nD — adversarial");
    let dd = test_d(&d)?;

    if a && b && c && dd {
        println!("\nisolated latency (correctness passed first)");
        latency(&d)?;
        println!("\nPASS");
        Ok(())
    } else {
        bail!("FAIL — see the lines marked FAIL above");
    }
}
