// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself
// derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0.
// Source: src/ops/kernel/sampling_device.cuh:79-114 (splitmix64 uniform, candidate order) and 401-438
// (`sampling_pick_from_support`), src/ops/kernel/speculative_round.cuh:415-450 (one-hot draft
// acceptance). Changes: translated from CUDA to Rust as the CPU model of
// `kernels/ada/common/spec_accept_sample_ada.cu`, with Atlas `SamplingParams` semantics and 64-bit
// positions, by Atlas contributors.
//! Responsibility: the CPU model of the device acceptance kernel, job for job, from the same blob.
//!
//! Greedy jobs are bit-identical to the kernel and to the host verify path (the edit arithmetic is the
//! host's, the argmax is [`spark_runtime::sampler::argmax_first_wins_f32`]). Sampled jobs follow the
//! kernel's algorithm in sequential f32: the kernel's parallel sums and its `expf` differ only by
//! rounding, so a GPU comparison sees the same pick except at a rounding boundary.

use super::blob::{self, DESC_AT, DESC_WORDS, DRAFTS_AT, FLAG_SAMPLE, h};
use super::{AcceptRequest, CAND_CAP, EditDesc, JobResult, NO_DRAFT, SampleParams, status};

/// RNG purpose of the draft acceptance test.
pub const PURPOSE_ACCEPT: u32 = 1;
/// RNG purpose of the residual or direct sample.
pub const PURPOSE_SAMPLE: u32 = 2;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Uniform in [0, 1) with 24 random bits, a pure function of (key, position, purpose): the kernel's
/// `sa_uniform`.
pub fn uniform(key: u64, position: u64, purpose: u32) -> f32 {
    let mut k = splitmix64(key ^ position.wrapping_mul(0xD1B5_4A32_D192_ED03));
    k = splitmix64(k ^ (u64::from(purpose) << 21));
    ((k >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
}

/// Monotone u32 key of a float (the kernel's `sa_order_key`).
pub fn order_key(v: f32) -> u32 {
    let b = v.to_bits();
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

/// The host pipeline's effect on one logit at verify position `pos`: masks, the repetition penalty
/// once per occurrence, the presence/frequency delta, then each bias in order.
pub fn apply_edit(mut x: f32, e: &EditDesc, pos: u32, eos_until: u32, rep: f32) -> f32 {
    if e.mask || (e.eos && pos < eos_until) {
        x = f32::NEG_INFINITY;
    } else if e.sub12 {
        x -= 12.0;
    }
    for _ in 0..e.rep_count {
        if x > 0.0 {
            x /= rep;
        } else {
            x *= rep;
        }
    }
    if let Some(d) = e.delta {
        x -= d;
    }
    for &b in e.bias.iter().take(e.n_bias as usize) {
        x += b;
    }
    x
}

/// A sampled job's support: ids and unnormalized weights, in the order the pick walks them.
#[derive(Clone, Debug, PartialEq)]
pub struct Support {
    /// Token ids.
    pub ids: Vec<u32>,
    /// Weights `exp(v / T - s_max)`, same order.
    pub w: Vec<f32>,
}

/// The support of one processed row under `p`, or the fallback [`status`] bit.
///
/// No truncation: every finite token in index order. Top-k (at most [`CAND_CAP`]), min-p and top-p:
/// the candidates in (value desc, index asc) order, filtered exactly as `spark_runtime::sampler` does
/// (top-k, then `w >= min_p * w_max`, then the inclusive cumulative `w / sum >= top_p`). An uncapped
/// min-p or top-p support that does not close within [`CAND_CAP`] candidates is [`status::SUPPORT`].
pub fn support(row: &[f32], p: &SampleParams) -> Result<Support, u32> {
    let finite: Vec<(u32, f32)> = row
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (i as u32, v))
        .collect();
    if finite.is_empty() {
        return Err(status::EMPTY);
    }
    let n = finite.len();
    let kk = if p.top_k > 0 && (p.top_k as usize) < n {
        p.top_k as usize
    } else {
        0
    };
    if kk > CAND_CAP {
        return Err(status::SUPPORT);
    }
    let t = p.temperature;
    let vmax = finite
        .iter()
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let smax = vmax / t;
    let truncated = kk > 0 || p.min_p > 0.0 || p.top_p < 1.0;
    if !truncated {
        let (ids, w) = finite
            .iter()
            .map(|&(i, v)| (i, (v / t - smax).exp()))
            .unzip();
        return Ok(Support { ids, w });
    }
    let mut sorted = finite;
    sorted.sort_by(|a, b| order_key(b.1).cmp(&order_key(a.1)).then(a.0.cmp(&b.0)));
    let kc = if kk > 0 { kk } else { n.min(CAND_CAP) };
    let cand = &sorted[..kc];
    let s0 = cand[0].1 / t;
    let w: Vec<f32> = cand.iter().map(|&(_, v)| (v / t - s0).exp()).collect();
    let cutoff = if kk > 0 {
        let thr = if p.min_p > 0.0 { p.min_p * w[0] } else { -1.0 };
        let n_ret = w.iter().take_while(|&&x| x >= thr).count();
        let sum_ret = w[..n_ret].iter().fold(0.0f32, |a, &x| a + x);
        let mut cutoff = n_ret;
        if p.top_p < 1.0 {
            let mut cum = 0.0f32;
            for (j, &x) in w[..n_ret].iter().enumerate() {
                cum += x / sum_ret;
                if cum >= p.top_p {
                    cutoff = j + 1;
                    break;
                }
            }
        }
        cutoff
    } else {
        // The host threshold is `min_p * max_prob`, and the full set's max weight is exp(0) = 1.
        let thr = if p.min_p > 0.0 { p.min_p } else { -1.0 };
        let mut sum_full = 0.0f32;
        let mut cnt_full = 0usize;
        for &(_, v) in &sorted {
            let x = (v / t - smax).exp();
            if x >= thr {
                sum_full += x;
                cnt_full += 1;
            }
        }
        let mut cutoff: Option<usize> = None;
        if p.top_p < 1.0 {
            let mut cum = 0.0f32;
            for (j, &x) in w.iter().enumerate() {
                if j >= cnt_full {
                    cutoff = Some(cnt_full);
                    break;
                }
                cum += x / sum_full;
                if cum >= p.top_p {
                    cutoff = Some(j + 1);
                    break;
                }
            }
        }
        match cutoff {
            Some(c) => c,
            None if cnt_full <= kc => cnt_full,
            None => return Err(status::SUPPORT),
        }
    };
    if cutoff == 0 {
        return Err(status::SUPPORT);
    }
    Ok(Support {
        ids: cand[..cutoff].iter().map(|&(i, _)| i).collect(),
        w: w[..cutoff].to_vec(),
    })
}

/// Inverse-CDF pick over a sorted support with `exclude` skipped: the first entry whose running weight
/// exceeds `u * mass` (the kernel's `sa_pick_support`).
pub fn pick_from_support(s: &Support, exclude: u32, u: f32) -> u32 {
    let mass = s
        .ids
        .iter()
        .zip(&s.w)
        .filter(|(i, _)| **i != exclude)
        .fold(0.0f32, |a, (_, &w)| a + w);
    if mass.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return s.ids[0];
    }
    let goal = u * mass;
    let mut acc = 0.0f32;
    let mut picked = s.ids[0];
    for (&i, &w) in s.ids.iter().zip(&s.w) {
        if i == exclude {
            continue;
        }
        acc += w;
        picked = i;
        if goal < acc {
            return i;
        }
    }
    picked
}

/// The FULL-mode pick (no truncation): weights in index order, the residual mass `total - w(d)` the
/// kernel uses, the first positive-weight id whose running weight exceeds the goal, else the last one.
fn pick_full(s: &Support, draft: u32, u_acc: f32, u_res: f32) -> (u32, bool) {
    let total = s.w.iter().fold(0.0f32, |a, &w| a + w);
    let wd = if draft == NO_DRAFT {
        0.0
    } else {
        s.ids
            .iter()
            .position(|&i| i == draft)
            .map_or(0.0, |j| s.w[j])
    };
    if draft != NO_DRAFT && wd > 0.0 && u_acc < wd / total {
        return (draft, true);
    }
    let mass = if draft != NO_DRAFT { total - wd } else { total };
    let goal = u_res * mass;
    let mut acc = 0.0f32;
    let mut last: Option<u32> = None;
    for (&i, &w) in s.ids.iter().zip(&s.w) {
        if i == draft || w <= 0.0 {
            continue;
        }
        acc += w;
        last = Some(i);
        if goal < acc {
            return (i, false);
        }
    }
    let fallback = if draft != NO_DRAFT { draft } else { 0 };
    (last.unwrap_or(fallback), false)
}

/// The truncated-support pick: accept the draft with probability `w(d) / total`, else the residual.
fn pick_truncated(s: &Support, draft: u32, u_acc: f32, u_res: f32) -> (u32, bool) {
    if draft != NO_DRAFT {
        let total = s.w.iter().fold(0.0f32, |a, &w| a + w);
        let wd = s
            .ids
            .iter()
            .position(|&i| i == draft)
            .map_or(0.0, |j| s.w[j]);
        if wd > 0.0 && u_acc < wd / total {
            return (draft, true);
        }
        return (pick_from_support(s, draft, u_res), false);
    }
    (pick_from_support(s, u32::MAX, u_res), false)
}

/// Whether `p` truncates the support (decides the kernel's FULL or TRUNC branch).
fn truncates(row: &[f32], p: &SampleParams) -> bool {
    let n = row.iter().filter(|v| v.is_finite()).count();
    (p.top_k > 0 && (p.top_k as usize) < n) || p.min_p > 0.0 || p.top_p < 1.0
}

/// Sample (or accept the draft from) one processed row: the kernel's sampled branch.
pub fn sample_row(row: &[f32], p: &SampleParams, draft: u32, key: u64, abs_pos: u64) -> JobResult {
    let mut st = nonfinite_bit(row);
    let s = match support(row, p) {
        Ok(s) => s,
        Err(bit) => {
            return JobResult {
                pick: 0,
                status: st | bit,
            };
        }
    };
    let u_acc = uniform(key, abs_pos, PURPOSE_ACCEPT);
    let u_res = uniform(key, abs_pos, PURPOSE_SAMPLE);
    let (pick, accepted) = if truncates(row, p) {
        pick_truncated(&s, draft, u_acc, u_res)
    } else {
        pick_full(&s, draft, u_acc, u_res)
    };
    if accepted {
        st |= status::ACCEPTED;
    }
    JobResult { pick, status: st }
}

fn nonfinite_bit(row: &[f32]) -> u32 {
    if row.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
        status::NONFINITE
    } else {
        0
    }
}

/// One processed row: the BF16 logits widened, then this regime's edits at `pos`.
pub fn processed_row(
    bf16_row: &[u8],
    vocab: usize,
    edits: &[EditDesc],
    pos: u32,
    eos_until: u32,
    rep: f32,
) -> Vec<f32> {
    let mut v: Vec<f32> = (0..vocab)
        .map(|i| {
            f32::from_bits(
                u32::from(u16::from_le_bytes([bf16_row[2 * i], bf16_row[2 * i + 1]])) << 16,
            )
        })
        .collect();
    for e in edits {
        let t = e.token as usize;
        v[t] = apply_edit(v[t], e, pos, eos_until, rep);
    }
    v
}

/// Run every job of an encoded request over host BF16 rows (`[n_pos, vocab]`, little-endian): the
/// kernel's output, job `regime * n_pos + pos`.
pub fn run_blob(rows: &[u8], w: &[u32]) -> Vec<JobResult> {
    let vocab = w[h::VOCAB] as usize;
    let n_pos = w[h::NPOS] as usize;
    let n_reg = w[h::NREG] as usize;
    let sample = w[h::FLAGS] & FLAG_SAMPLE != 0;
    let params = SampleParams {
        temperature: f32::from_bits(w[h::TEMP]),
        top_k: w[h::TOPK],
        top_p: f32::from_bits(w[h::TOPP]),
        min_p: f32::from_bits(w[h::MINP]),
    };
    let rep = f32::from_bits(w[h::REP]);
    let eos_until = w[h::EOS_UNTIL];
    let key = u64::from(w[h::KEY_LO]) | (u64::from(w[h::KEY_HI]) << 32);
    let pos_base = u64::from(w[h::POS_LO]) | (u64::from(w[h::POS_HI]) << 32);
    let counts = [w[h::NEDIT0] as usize, w[h::NEDIT1] as usize];
    let mut regimes: Vec<Vec<EditDesc>> = Vec::with_capacity(n_reg);
    let mut at = DESC_AT;
    for &c in counts.iter().take(n_reg) {
        let edits = (0..c)
            .map(|j| blob::desc_from_words(&w[at + j * DESC_WORDS..at + (j + 1) * DESC_WORDS]))
            .collect();
        at += c * DESC_WORDS;
        regimes.push(edits);
    }
    let mut out = Vec::with_capacity(n_reg * n_pos);
    for edits in &regimes {
        for pos in 0..n_pos {
            let r = &rows[pos * vocab * 2..(pos + 1) * vocab * 2];
            let v = processed_row(r, vocab, edits, pos as u32, eos_until, rep);
            out.push(if sample {
                sample_row(&v, &params, w[DRAFTS_AT + pos], key, pos_base + pos as u64)
            } else {
                JobResult {
                    pick: spark_runtime::sampler::argmax_first_wins_f32(&v),
                    status: nonfinite_bit(&v),
                }
            });
        }
    }
    out
}

/// [`run_blob`] over a request (encodes it first, so the encoding is exercised too).
pub fn run(rows: &[u8], req: &AcceptRequest<'_>) -> Vec<JobResult> {
    run_blob(rows, &blob::encode(req))
}
