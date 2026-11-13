// SPDX-License-Identifier: AGPL-3.0-only
// The symmetric round-to-nearest rule follows vLLM (https://github.com/vllm-project/vllm, commit
// 8a36f08), vllm/model_executor/layers/quantization/utils/quant_utils.py:716-812 (`quantize_weights`
// without zero points for `uint4b8`: `s = max(|max / 7|, |min / -8|)`, `q = clamp(round(w / s), -8, 7)
// + 8`, round half to even). Copyright contributors to the vLLM project, Apache-2.0.
// Changes: the scale is rounded to BF16 before the codes are chosen (so the codes fit the stored
// scale), an optional per-group clip search over shrink ratios picks the scale with the least squared
// error, all-zero groups get scale 0, non-finite inputs are refused, and the work is split over host
// threads.

//! Responsibility: the opt-in RTN int4 g128 quantizer (`kU4B8`, symmetric) from a BF16 `[n, k]`
//! weight to a [`LogicalInt4`]. Host only; deterministic (the result does not depend on the thread
//! count).

use anyhow::{Result, bail, ensure};

use super::reference::{LogicalInt4, bf16_to_f32, dequant, f32_to_bf16};

/// Largest clip-search candidate count: shrink ratios 0.95 down to 0.50 in steps of 0.05.
pub const MAX_CLIP_SEARCH: u8 = 10;
/// The step between clip-search shrink ratios.
pub const CLIP_STEP: f32 = 0.05;
/// The largest positive code offset of `uint4b8` (`quant_type.max()`).
pub const MAX_Q: f32 = 7.0;
/// The magnitude of the most negative code offset (`-quant_type.min()`).
pub const MIN_Q_MAG: f32 = 8.0;

/// Options of the RTN quantizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtnOpts {
    pub group: usize,
    /// Extra shrink ratios to try per group (0 = plain RTN).
    pub clip_search: u8,
}

/// Quantize one group of values at one BF16 scale: codes into `q`, squared error returned.
fn codes_at(vals: &[f32], s: u16, q: &mut [u8]) -> f64 {
    let sf = bf16_to_f32(s);
    let mut err = 0f64;
    for (v, out) in vals.iter().zip(q.iter_mut()) {
        let code = if sf == 0.0 {
            8u8
        } else {
            let r = (v / sf).round_ties_even().clamp(-MIN_Q_MAG, MAX_Q);
            (r as i32 + 8) as u8
        };
        *out = code;
        let d = f64::from(*v) - f64::from(bf16_to_f32(dequant(code, 8, s)));
        err += d * d;
    }
    err
}

/// Quantize one group: the BF16 scale and the codes (`q.len() == vals.len()`). With `clip_search`
/// candidates, the scale `bf16(base * r)` with the least squared error wins; ties keep the larger `r`.
pub fn quantize_group(vals: &[f32], clip_search: u8, q: &mut [u8]) -> u16 {
    let (mut mx, mut mn) = (f32::NEG_INFINITY, f32::INFINITY);
    for &v in vals {
        mx = mx.max(v);
        mn = mn.min(v);
    }
    let base = (mx / MAX_Q).abs().max((mn / -MIN_Q_MAG).abs());
    let mut best_s = f32_to_bf16(base);
    let mut best_err = codes_at(vals, best_s, q);
    if clip_search > 0 && base > 0.0 {
        let mut trial = vec![0u8; vals.len()];
        for j in 1..=clip_search.min(MAX_CLIP_SEARCH) {
            let r = 1.0 - CLIP_STEP * f32::from(j);
            let s = f32_to_bf16(base * r);
            let err = codes_at(vals, s, &mut trial);
            if err < best_err {
                best_err = err;
                best_s = s;
                q.copy_from_slice(&trial);
            }
        }
    }
    best_s
}

/// Quantize rows `n0..n0 + rows` of `w` (`[n, k]` BF16): codes `[rows][k]` and scales `[rows][groups]`.
fn quantize_rows(
    w: &[u16],
    k: usize,
    n0: usize,
    rows: usize,
    opts: &RtnOpts,
) -> (Vec<u8>, Vec<u16>) {
    let groups = k / opts.group;
    let mut q = vec![0u8; rows * k];
    let mut s = vec![0u16; rows * groups];
    let mut vals = vec![0f32; opts.group];
    for r in 0..rows {
        let row = &w[(n0 + r) * k..(n0 + r + 1) * k];
        for g in 0..groups {
            for (dst, &b) in vals
                .iter_mut()
                .zip(&row[g * opts.group..(g + 1) * opts.group])
            {
                *dst = bf16_to_f32(b);
            }
            let qo = &mut q[r * k + g * opts.group..r * k + (g + 1) * opts.group];
            s[r * groups + g] = quantize_group(&vals, opts.clip_search, qo);
        }
    }
    (q, s)
}

/// Symmetric RTN int4 of a BF16 `[n, k]` weight into the Marlin orientation (`q [k][n]`,
/// `scales [k / group][n]`, `kU4B8`).
pub fn rtn_int4_sym(w: &[u16], n: usize, k: usize, opts: &RtnOpts) -> Result<LogicalInt4> {
    ensure!(
        w.len() == n * k,
        "marlin rtn: {} values for a [{n}x{k}] weight",
        w.len()
    );
    ensure!(
        opts.group > 0 && k.is_multiple_of(opts.group),
        "marlin rtn: K={k} is not a multiple of the group {}",
        opts.group
    );
    if let Some(i) = w.iter().position(|&b| !bf16_to_f32(b).is_finite()) {
        bail!(
            "marlin rtn: weight element [{}, {}] is not finite",
            i / k,
            i % k
        );
    }
    let groups = k / opts.group;
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .clamp(1, 32);
    let per = n.div_ceil(threads).max(1);
    let parts: Vec<(usize, Vec<u8>, Vec<u16>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..n)
            .step_by(per)
            .map(|n0| {
                let rows = per.min(n - n0);
                sc.spawn(move || {
                    let (q, s) = quantize_rows(w, k, n0, rows, opts);
                    (n0, q, s)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| anyhow::anyhow!("marlin rtn: a quantizer thread panicked"))
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let mut q = vec![0u8; k * n];
    let mut scales = vec![0u16; groups * n];
    for (n0, pq, ps) in parts {
        let rows = ps.len() / groups;
        for r in 0..rows {
            let nn = n0 + r;
            for kk in 0..k {
                q[kk * n + nn] = pq[r * k + kk];
            }
            for g in 0..groups {
                scales[g * n + nn] = ps[r * groups + g];
            }
        }
    }
    Ok(LogicalInt4 {
        n,
        k,
        group: opts.group,
        q,
        scales,
        zeros: None,
    })
}
