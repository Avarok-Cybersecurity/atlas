// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 **attention, ratio-0 layers** (`Attention.forward` with `compress_ratio == 0`):
//! low-rank Q with RMSNorm, a single latent KV row per token (`wkv` + `kv_norm`), RoPE on the
//! last `rope_head_dim` of both, the KV row quantised to fp8 in place (block 32, ue8m0 scale),
//! a sliding-window ring cache, `sparse_attn` with the attention sink, the inverse rotation on
//! the output, then the grouped low-rank output projection (`wo_a` block-diagonal over groups,
//! `wo_b`). CPU reference against the golden.
//!
//! Precision follows the reference: every `Linear` is a bf16 GEMM (f32 accumulate, bf16 result),
//! RMSNorm is f32 back to bf16, RoPE is a complex multiply in f32 written back into the bf16
//! tensor, the fp8 round trip is exact (e4m3 times a power of two is bf16-exact), and
//! `sparse_attn` runs in f32 and returns bf16. Accumulation order inside a GEMM and inside the
//! attention sums is the only thing that differs from torch, so tensors are compared at bf16
//! tolerance; the window indices are compared exactly.
//!
//! The ring cache is the part that only a sequential run can check: prefill of 12 tokens into 8
//! slots seeds `slots[4..8] = tokens 4..8` and `slots[0..4] = tokens 8..12`; decode step 12 writes
//! slot 4 and reads the whole ring oldest-first; step 13 writes slot 5. The tests run the three
//! regimes in that order with one cache and check `sa_kv` (the ring) at each decode step.

use super::engram::{e4m3_to_f32, f32_to_e4m3_rne};
use super::hc::rms_norm;
use super::moe::linear_bf16;
use super::to_bf16_rne;

const FP8_MAX: f32 = 448.0;
/// `fp8_block_size` in the reference (model.py:27).
pub const FP8_BLOCK: usize = 32;

/// `2 ** ceil(log2(v))` for positive normal f32 via the IEEE fields, as `fast_log2_ceil` /
/// `fast_pow2` in kernel.py:22-35 (and `_pow2_ceil_log2` in the shims).
pub fn pow2_ceil(v: f32) -> f32 {
    let bits = v.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    let man = bits & 0x7F_FFFF;
    let n = e + if man != 0 { 1 } else { 0 };
    f32::from_bits(((n + 127) as u32) << 23)
}

/// `act_quant(x, 32, "ue8m0", e8m0, inplace=True)`: per block of `FP8_BLOCK`, scale = the
/// power-of-two ceiling of `amax / 448` (amax floored at 1e-4), values to e4m3 (RNE, clamped),
/// then dequantised and stored back as bf16.
pub fn act_quant_inplace(x: &mut [f32]) {
    for blk in x.chunks_mut(FP8_BLOCK) {
        let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-4);
        let s = pow2_ceil(amax / FP8_MAX);
        for v in blk.iter_mut() {
            let q = f32_to_e4m3_rne((*v / s).clamp(-FP8_MAX, FP8_MAX));
            *v = to_bf16_rne(e4m3_to_f32(q) * s);
        }
    }
}

/// `precompute_freqs_cis(dim, seqlen, original_seq_len=0, base, ...)`, the no-YaRN form the
/// ratio-0 layers use: `[seqlen][dim/2]` of (cos, sin).
pub fn freqs_cis(dim: usize, seqlen: usize, base: f32) -> Vec<(f32, f32)> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|k| 1.0 / base.powf((2 * k) as f32 / dim as f32))
        .collect();
    let mut out = Vec::with_capacity(seqlen * half);
    for p in 0..seqlen {
        for &f in &freqs {
            let a = p as f32 * f;
            out.push((a.cos(), a.sin()));
        }
    }
    out
}

/// `apply_rotary_emb` on the last `rope_dim` elements of each `row_len`-wide row of `x`, row `r`
/// at position `pos[r]`; adjacent pairs are complex numbers, multiplied in f32, written back bf16.
pub fn apply_rotary(
    x: &mut [f32],
    row_len: usize,
    rope_dim: usize,
    pos: &[usize],
    fc: &[(f32, f32)],
    inverse: bool,
) {
    let half = rope_dim / 2;
    for (r, &p) in pos.iter().enumerate() {
        let row = &mut x[r * row_len + row_len - rope_dim..(r + 1) * row_len];
        for k in 0..half {
            let (c, s) = fc[p * half + k];
            let s = if inverse { -s } else { s };
            let (a, b) = (row[2 * k], row[2 * k + 1]);
            row[2 * k] = to_bf16_rne(a * c - b * s);
            row[2 * k + 1] = to_bf16_rne(a * s + b * c);
        }
    }
}

/// `get_window_topk_idxs`: `[queries][cols]` of slot indices, -1 for a slot holding nothing.
/// Prefill: one causal window per query into the chunk itself. Decode: the whole ring, oldest
/// first.
pub fn window_topk_idxs(win: usize, seqlen: usize, start_pos: usize) -> (Vec<i32>, usize) {
    if start_pos == 0 {
        let cols = seqlen.min(win);
        let mut idx = Vec::with_capacity(seqlen * cols);
        for t in 0..seqlen {
            let base = t.saturating_sub(win - 1);
            for j in 0..cols {
                let i = base + j;
                idx.push(if i > t { -1 } else { i as i32 });
            }
        }
        (idx, cols)
    } else {
        let oldest = start_pos % win + 1;
        let order = (oldest..win).chain(0..oldest);
        (
            order
                .map(|i| if i > start_pos { -1 } else { i as i32 })
                .collect(),
            win,
        )
    }
}

/// `sparse_attn` (kernel.py:310-405, shim form): per query and head, softmax over the gathered
/// rows plus the sink (denominator only), all in f32; output bf16. `q`: `[s][h][d]`,
/// `kv`: `[rows][d]`, `idx`: `[s][topk]`.
pub fn sparse_attn(
    q: &[f32],
    kv: &[f32],
    sink: &[f32],
    idx: &[i32],
    s: usize,
    h: usize,
    d: usize,
    topk: usize,
    scale: f32,
) -> Vec<f32> {
    let mut o = vec![0f32; s * h * d];
    for t in 0..s {
        let ids = &idx[t * topk..(t + 1) * topk];
        for hh in 0..h {
            let qv = &q[(t * h + hh) * d..(t * h + hh + 1) * d];
            let mut scores: Vec<f32> = ids
                .iter()
                .map(|&i| {
                    if i < 0 {
                        f32::NEG_INFINITY
                    } else {
                        let kr = &kv[i as usize * d..(i as usize + 1) * d];
                        qv.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() * scale
                    }
                })
                .collect();
            let m = scores.iter().cloned().fold(sink[hh], f32::max);
            let mut denom = (sink[hh] - m).exp();
            for v in scores.iter_mut() {
                *v = (*v - m).exp();
                denom += *v;
            }
            let ov = &mut o[(t * h + hh) * d..(t * h + hh + 1) * d];
            for (j, &i) in ids.iter().enumerate() {
                if i < 0 {
                    continue;
                }
                let p = scores[j] / denom;
                let kr = &kv[i as usize * d..(i as usize + 1) * d];
                for dd in 0..d {
                    ov[dd] += p * kr[dd];
                }
            }
            for v in ov.iter_mut() {
                *v = to_bf16_rne(*v);
            }
        }
    }
    o
}

/// The layer's parameters, all bf16 values except the f32 sink. Shapes as `nn.Linear` stores
/// them (`[out, in]`).
pub struct AttnWeights<'a> {
    pub sink: &'a [f32],
    pub wq_a: &'a [f32],
    pub q_norm: &'a [f32],
    pub wq_b: &'a [f32],
    pub wkv: &'a [f32],
    pub kv_norm: &'a [f32],
    pub wo_a: &'a [f32],
    pub wo_b: &'a [f32],
}

pub struct AttnCfg {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_rank: usize,
    pub o_rank: usize,
    pub groups: usize,
    pub window: usize,
    pub eps: f32,
}

/// The sliding-window ring cache, `[window][head_dim]`.
pub struct WindowCache {
    pub slots: Vec<f32>,
}

impl WindowCache {
    pub fn new(c: &AttnCfg) -> Self {
        WindowCache {
            slots: vec![0f32; c.window * c.head_dim],
        }
    }
}

/// Every intermediate the golden captured, plus the output. `o` is `sparse_attn`'s result
/// BEFORE the inverse rotation, which is where the golden samples it.
pub struct AttnRun {
    pub q: Vec<f32>,
    pub kv_rows: Vec<f32>,
    pub idx: Vec<i32>,
    pub topk: usize,
    pub o: Vec<f32>,
    pub out: Vec<f32>,
}

/// `Attention.forward` for a ratio-0 layer. `x`: `[seqlen][dim]` bf16 values; `fc` from
/// [`freqs_cis`] over `max_seq_len`. Mutates `cache` exactly as the reference's ring buffer.
pub fn attention(
    x: &[f32],
    seqlen: usize,
    start_pos: usize,
    w: &AttnWeights,
    c: &AttnCfg,
    fc: &[(f32, f32)],
    cache: &mut WindowCache,
) -> AttnRun {
    let (hd, rd, nh) = (c.head_dim, c.rope_dim, c.n_heads);
    let pos: Vec<usize> = (0..seqlen).map(|t| start_pos + t).collect();

    // q: low-rank, normed, rotated
    let qr = rms_norm(
        &linear_bf16(x, w.wq_a, seqlen, c.dim, c.q_rank),
        w.q_norm,
        seqlen,
        c.q_rank,
        c.eps,
    );
    let mut q = linear_bf16(&qr, w.wq_b, seqlen, c.q_rank, nh * hd);
    let head_pos: Vec<usize> = pos
        .iter()
        .flat_map(|&p| std::iter::repeat_n(p, nh))
        .collect();
    apply_rotary(&mut q, hd, rd, &head_pos, fc, false);

    // kv: one latent row per token, normed, rotated, fp8 round trip
    let mut kv = rms_norm(
        &linear_bf16(x, w.wkv, seqlen, c.dim, hd),
        w.kv_norm,
        seqlen,
        hd,
        c.eps,
    );
    apply_rotary(&mut kv, hd, rd, &pos, fc, false);
    act_quant_inplace(&mut kv);

    // ring cache and the rows attention reads
    let win = c.window;
    let kv_rows = if start_pos == 0 {
        if seqlen <= win {
            cache.slots[..seqlen * hd].copy_from_slice(&kv);
        } else {
            let cutoff = seqlen % win;
            let tail = &kv[(seqlen - win) * hd..];
            cache.slots[cutoff * hd..win * hd].copy_from_slice(&tail[..(win - cutoff) * hd]);
            cache.slots[..cutoff * hd].copy_from_slice(&tail[(win - cutoff) * hd..]);
        }
        kv.clone()
    } else {
        let slot = start_pos % win;
        cache.slots[slot * hd..(slot + 1) * hd].copy_from_slice(&kv);
        cache.slots.clone()
    };
    let (idx, topk) = window_topk_idxs(win, seqlen, start_pos);

    let scale = (hd as f32).powf(-0.5);
    // `sa_o` is captured as `sparse_attn` returns it; the inverse rotation is applied in place
    // after, so keep the pre-rotation tensor for the comparison
    let o_sa = sparse_attn(&q, &kv_rows, w.sink, &idx, seqlen, nh, hd, topk, scale);
    let mut o = o_sa.clone();
    apply_rotary(&mut o, hd, rd, &head_pos, fc, true);

    // grouped low-rank output: wo_a is [groups][o_rank][heads_per_group * hd], then wo_b
    let gw = nh * hd / c.groups;
    let mut og = vec![0f32; seqlen * c.groups * c.o_rank];
    for t in 0..seqlen {
        for g in 0..c.groups {
            let ov = &o[t * nh * hd + g * gw..t * nh * hd + (g + 1) * gw];
            for r in 0..c.o_rank {
                let wr = &w.wo_a[(g * c.o_rank + r) * gw..(g * c.o_rank + r + 1) * gw];
                og[(t * c.groups + g) * c.o_rank + r] =
                    to_bf16_rne(ov.iter().zip(wr).map(|(a, b)| a * b).sum());
            }
        }
    }
    let out = linear_bf16(&og, w.wo_b, seqlen, c.groups * c.o_rank, c.dim);
    AttnRun {
        q,
        kv_rows,
        idx,
        topk,
        o: o_sa,
        out,
    }
}

#[cfg(test)]
#[path = "attn_tests.rs"]
mod tests;
