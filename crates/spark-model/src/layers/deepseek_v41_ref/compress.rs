// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 **compressed attention** (`compress_ratio > 0`): the `Compressor` (softmax
//! pooling of `ratio` tokens into one latent, or a plain projection at ratio 1), the `Indexer`
//! (fp4 query heads against one shared fp4 key per compressed position, rectified scores combined
//! by `weights_proj`), `select_candidate_blocks` (level one of the two-level top-k), the
//! `SharedAttentionRuntime` slots layers hand down the stack, YaRN RoPE with
//! `compress_rope_theta`, the two fp4 quantisers, and the full `Attention.forward` for any ratio.
//! CPU reference against the golden; the ratio-0 pieces are shared with `attn.rs`.
//!
//! Precision follows the reference: the ratio-2 compressor is f32 end to end until its RMSNorm
//! casts to bf16; the indexer's q and k go through the fp4 e2m1 round trip (e8m0 scales, block
//! 32) and its scores are bf16 tensors; the compressed KV latent goes through e2m1 with e4m3
//! scales (block 16). Index selections are compared exactly, tensors at bf16 tolerance.

use super::attn::{act_quant_inplace, apply_rotary, pow2_ceil, sparse_attn, window_topk_idxs};
use super::engram::{e4m3_to_f32, f32_to_e4m3_rne};
use super::hc::rms_norm;
use super::moe::linear_bf16;
use super::to_bf16_rne;

/// `fp4_block_size` (model.py:28).
pub const FP4_BLOCK: usize = 32;
/// Block size the compressed KV latent is quantised with (`fp4_act_quant(latent, 16, ...)`).
pub const LATENT_BLOCK: usize = 16;
const FP4_MAX: f32 = 6.0;
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Round onto the e2m1 grid with ties to the even code, keeping the sign (`_to_e2m1_rne`).
pub fn to_e2m1_rne(y: f32) -> f32 {
    let a = y.abs();
    let mut lo = 0usize;
    for (i, &v) in E2M1.iter().enumerate() {
        if v <= a {
            lo = i;
        }
    }
    let hi = (lo + 1).min(7);
    let (dlo, dhi) = (a - E2M1[lo], E2M1[hi] - a);
    let pick_hi = dhi < dlo || (dhi == dlo && hi % 2 == 0 && lo % 2 == 1);
    let v = if pick_hi { E2M1[hi] } else { E2M1[lo] };
    v.copysign(y)
}

/// `fp4_act_quant(x, block, inplace=True)` with e8m0 scales: scale = pow2 ceiling of
/// `amax / 6` (amax floored at `6 * 2^-126`), values to e2m1, dequantised back to bf16.
pub fn fp4_quant_e8m0_inplace(x: &mut [f32], block: usize) {
    for blk in x.chunks_mut(block) {
        let amax = blk
            .iter()
            .fold(0f32, |a, v| a.max(v.abs()))
            .max(6.0 * 2f32.powi(-126));
        let s = pow2_ceil(amax / FP4_MAX);
        for v in blk.iter_mut() {
            *v = to_bf16_rne(to_e2m1_rne((*v / s).clamp(-FP4_MAX, FP4_MAX)) * s);
        }
    }
}

/// `fp4_act_quant(x, 16, inplace=True, scale_dtype=e4m3)`: amax floored at `6 * 2^-9`, scale =
/// `amax / 6` rounded through e4m3, values to e2m1, dequantised back to bf16.
pub fn fp4_quant_e4m3_inplace(x: &mut [f32], block: usize) {
    for blk in x.chunks_mut(block) {
        let amax = blk
            .iter()
            .fold(0f32, |a, v| a.max(v.abs()))
            .max(6.0 * 2f32.powi(-9));
        let s = e4m3_to_f32(f32_to_e4m3_rne(amax / FP4_MAX));
        for v in blk.iter_mut() {
            *v = to_bf16_rne(to_e2m1_rne((*v / s).clamp(-FP4_MAX, FP4_MAX)) * s);
        }
    }
}

/// `precompute_freqs_cis` WITH YaRN (`original_seq_len > 0`), as the compress layers build it
/// from `compress_rope_theta`. `[seqlen][dim/2]` of (cos, sin).
pub fn yarn_freqs_cis(
    dim: usize,
    seqlen: usize,
    original_seq_len: usize,
    base: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Vec<(f32, f32)> {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half)
        .map(|k| 1.0 / base.powf((2 * k) as f32 / dim as f32))
        .collect();
    let corrected = |rot: f64| -> f64 {
        dim as f64 * (original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * (base as f64).ln())
    };
    let low = corrected(beta_fast as f64).floor().max(0.0) as f32;
    let high = (corrected(beta_slow as f64).ceil()).min((dim - 1) as f64) as f32;
    for (i, f) in freqs.iter_mut().enumerate() {
        let ramp = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
        let smooth = 1.0 - ramp;
        *f = *f / factor * (1.0 - smooth) + *f * smooth;
    }
    let mut out = Vec::with_capacity(seqlen * half);
    for p in 0..seqlen {
        for &f in &freqs {
            let a = p as f32 * f;
            out.push((a.cos(), a.sin()));
        }
    }
    out
}

/// f32 `F.linear` (the ratio-2 compressor's `wkv` / `wgate` are fp32 weights on an fp32 input).
pub fn linear_f32(x: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * out_dim];
    for i in 0..rows {
        for o in 0..out_dim {
            y[i * out_dim + o] = x[i * in_dim..(i + 1) * in_dim]
                .iter()
                .zip(&w[o * in_dim..(o + 1) * in_dim])
                .map(|(a, b)| a * b)
                .sum();
        }
    }
    y
}

pub struct CompressorWeights<'a> {
    /// f32 at ratio > 1, bf16 values at ratio 1
    pub wkv: &'a [f32],
    /// ratio > 1 only
    pub wgate: Option<&'a [f32]>,
    pub norm: &'a [f32],
}

pub struct CompressorState {
    pub kv_state: Vec<f32>,
    pub score_state: Vec<f32>,
}

impl CompressorState {
    pub fn new(ratio: usize, hd: usize) -> Self {
        CompressorState {
            kv_state: vec![0f32; ratio * hd],
            score_state: vec![f32::NEG_INFINITY; ratio * hd],
        }
    }
}

/// `Compressor.forward`: the pre-RoPE latent `[groups][hd]` (bf16 values), or `None` while a
/// group is still filling up in decode.
pub fn compressor(
    x: &[f32],
    seqlen: usize,
    start_pos: usize,
    ratio: usize,
    dim: usize,
    hd: usize,
    w: &CompressorWeights,
    st: &mut CompressorState,
    eps: f32,
) -> Option<Vec<f32>> {
    if ratio == 1 {
        return Some(rms_norm(
            &linear_bf16(x, w.wkv, seqlen, dim, hd),
            w.norm,
            seqlen,
            hd,
            eps,
        ));
    }
    let kv = linear_f32(x, w.wkv, seqlen, dim, hd);
    let score = linear_f32(x, w.wgate.expect("wgate at ratio > 1"), seqlen, dim, hd);
    let pool = |kvg: &[f32], scg: &[f32]| -> Vec<f32> {
        // softmax over the `ratio` members per dimension, then the weighted sum
        let mut out = vec![0f32; hd];
        for d in 0..hd {
            let m = (0..ratio)
                .map(|r| scg[r * hd + d])
                .fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = (0..ratio).map(|r| (scg[r * hd + d] - m).exp()).collect();
            let sum: f32 = ex.iter().sum();
            out[d] = (0..ratio).map(|r| kvg[r * hd + d] * (ex[r] / sum)).sum();
        }
        out
    };
    let pooled: Vec<f32> = if start_pos == 0 {
        let remainder = seqlen % ratio;
        let cutoff = seqlen - remainder;
        if remainder > 0 {
            st.kv_state[..remainder * hd].copy_from_slice(&kv[cutoff * hd..]);
            st.score_state[..remainder * hd].copy_from_slice(&score[cutoff * hd..]);
        }
        if seqlen < ratio {
            return None;
        }
        (0..cutoff / ratio)
            .flat_map(|g| {
                pool(
                    &kv[g * ratio * hd..(g + 1) * ratio * hd],
                    &score[g * ratio * hd..(g + 1) * ratio * hd],
                )
            })
            .collect()
    } else {
        let slot = start_pos % ratio;
        st.kv_state[slot * hd..(slot + 1) * hd].copy_from_slice(&kv);
        st.score_state[slot * hd..(slot + 1) * hd].copy_from_slice(&score);
        if (start_pos + 1) % ratio != 0 {
            return None;
        }
        pool(&st.kv_state, &st.score_state)
    };
    let groups = pooled.len() / hd;
    let as_bf16: Vec<f32> = pooled.into_iter().map(to_bf16_rne).collect();
    Some(rms_norm(&as_bf16, w.norm, groups, hd, eps))
}

pub struct IndexerWeights<'a> {
    pub wq_b: &'a [f32],
    pub weights_proj: &'a [f32],
    /// index-key owners (kv sources) only
    pub wk: Option<&'a [f32]>,
    pub k_norm: Option<&'a [f32]>,
}

/// `SharedAttentionRuntime`: what source layers publish and later layers read.
#[derive(Default)]
pub struct SharedRuntime {
    /// the source's full cache `[max_groups][hd]`
    pub compress_kv: Vec<f32>,
    /// the source's full index-key cache `[max_groups][index_hd]`
    pub index_k: Vec<f32>,
    /// `[queries][topk]`, already offset for the window rows
    pub topk_idxs: Vec<i32>,
    pub topk: usize,
    /// `[queries][width]`
    pub candidates: Vec<bool>,
    pub cand_width: usize,
}

pub struct IndexerCfg {
    pub n_heads: usize,
    pub index_hd: usize,
    pub rope_dim: usize,
    pub q_rank: usize,
    pub dim: usize,
    pub hd: usize,
    pub index_topk: usize,
    pub cand_topk_blocks: usize,
    pub cand_block: usize,
    pub eps: f32,
}

/// The index SET torch's CPU `topk(k, largest=True)` returns for `vals`, ties included. For the
/// row lengths here (`k * 64 > n`) torch runs `std::nth_element` on (value, index) pairs with a
/// descending comparator and returns the first `k` pairs, so among equal values (the indexer's
/// many -inf entries) the picks are whatever libstdc++'s introselect leaves in front. That order
/// is deterministic and this is a line-for-line emulation of it (`__introselect`,
/// `__move_median_to_first`, `__unguarded_partition`, `__insertion_sort`). Pinned by
/// `topk_tie_order_matches_torch_cpu` against this box's torch.
pub fn torch_cpu_topk_set(vals: &[f32], k: usize) -> Vec<usize> {
    let n = vals.len();
    assert!(k <= n && k > 0);
    assert!(
        k * 64 > n,
        "torch takes the partial_sort branch for k*64 <= n; not emulated"
    );
    let mut q: Vec<(f32, usize)> = vals
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    // comp(x, y): x before y when x is NaN and y is not, or x.value > y.value
    let comp = |x: &(f32, usize), y: &(f32, usize)| (x.0.is_nan() && !y.0.is_nan()) || x.0 > y.0;
    let nth = k - 1;
    let (mut first, mut last) = (0usize, n);
    let mut depth = 2 * (usize::BITS - n.leading_zeros()) as usize;
    while last - first > 3 {
        if depth == 0 {
            // heap_select fallback; never reached for these sizes
            unreachable!("introselect depth limit reached");
        }
        depth -= 1;
        let mid = first + (last - first) / 2;
        // __move_median_to_first(first, first+1, mid, last-1)
        let (a, b, c) = (first + 1, mid, last - 1);
        let r = if comp(&q[a], &q[b]) {
            if comp(&q[b], &q[c]) {
                b
            } else if comp(&q[a], &q[c]) {
                c
            } else {
                a
            }
        } else if comp(&q[a], &q[c]) {
            a
        } else if comp(&q[b], &q[c]) {
            c
        } else {
            b
        };
        q.swap(first, r);
        // __unguarded_partition(first+1, last, pivot=first)
        let pivot = q[first];
        let (mut lo, mut hi) = (first + 1, last);
        let cut = loop {
            while comp(&q[lo], &pivot) {
                lo += 1;
            }
            hi -= 1;
            while comp(&pivot, &q[hi]) {
                hi -= 1;
            }
            if lo >= hi {
                break lo;
            }
            q.swap(lo, hi);
            lo += 1;
        };
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    // __insertion_sort(first, last)
    for i in (first + 1)..last {
        let val = q[i];
        if comp(&val, &q[first]) {
            for j in (first + 1..=i).rev() {
                q[j] = q[j - 1];
            }
            q[first] = val;
        } else {
            let mut j = i;
            while comp(&val, &q[j - 1]) {
                q[j] = q[j - 1];
                j -= 1;
            }
            q[j] = val;
        }
    }
    q[..k].iter().map(|&(_, i)| i).collect()
}

/// `select_candidate_blocks`: `logits[queries][width]` with unreachable positions at -inf;
/// `compress_lens[q]`. Returns the keep mask `[queries][width]`.
pub fn select_candidate_blocks(
    logits: &[f32],
    queries: usize,
    width: usize,
    compress_lens: &[usize],
    topk_blocks: usize,
    block: usize,
) -> Vec<bool> {
    let nb = (width + block - 1) / block;
    let mut keep = vec![false; queries * width];
    for q in 0..queries {
        let row = &logits[q * width..(q + 1) * width];
        let mut scores: Vec<f32> = (0..nb)
            .map(|b| {
                (b * block..((b + 1) * block).min(width))
                    .map(|i| row[i])
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        // the block holding this query's newest position is pinned in
        let last = (compress_lens[q] as i64 - 1).div_euclid(block as i64);
        if last >= 0 && (last as usize) < nb {
            scores[last as usize] = f32::INFINITY;
        }
        for b in torch_cpu_topk_set(&scores, topk_blocks.min(nb)) {
            if scores[b] > f32::NEG_INFINITY {
                for i in b * block..((b + 1) * block).min(width) {
                    keep[q * width + i] = true;
                }
            }
        }
    }
    keep
}

/// `Indexer.forward`. Returns `[queries][topk]` compressed-position indices, offset by
/// `offset`, -1 where unreachable; also publishes index keys when this layer owns them.
#[allow(clippy::too_many_arguments)]
pub fn indexer(
    x: &[f32],
    qr: &[f32],
    latent: Option<&[f32]>,
    seqlen: usize,
    start_pos: usize,
    offset: usize,
    ratio: usize,
    c: &IndexerCfg,
    w: &IndexerWeights,
    fc: &[(f32, f32)],
    k_cache: Option<&mut Vec<f32>>,
    shared: &mut SharedRuntime,
    is_candidate_source: bool,
    uses_candidates: bool,
) -> (Vec<i32>, usize) {
    let (nh, ihd, rd) = (c.n_heads, c.index_hd, c.rope_dim);
    let end_pos = start_pos + seqlen;

    if let (Some(lat), Some(cache)) = (latent, k_cache) {
        let groups = lat.len() / c.hd;
        let mut k = rms_norm(
            &linear_bf16(lat, w.wk.expect("wk"), groups, c.hd, ihd),
            w.k_norm.expect("k_norm"),
            groups,
            ihd,
            c.eps,
        );
        let pos: Vec<usize> = if start_pos == 0 {
            (0..groups).map(|g| g * ratio).collect()
        } else {
            vec![start_pos + 1 - ratio]
        };
        apply_rotary(&mut k, ihd, rd, &pos, fc, false);
        fp4_quant_e8m0_inplace(&mut k, FP4_BLOCK);
        let at = start_pos / ratio;
        cache[at * ihd..(at + groups) * ihd].copy_from_slice(&k);
        shared.index_k = cache.clone();
    }

    let mut q = linear_bf16(qr, w.wq_b, seqlen, c.q_rank, nh * ihd);
    let head_pos: Vec<usize> = (0..seqlen)
        .flat_map(|t| std::iter::repeat(start_pos + t).take(nh))
        .collect();
    apply_rotary(&mut q, ihd, rd, &head_pos, fc, false);
    fp4_quant_e8m0_inplace(&mut q, FP4_BLOCK);

    let width = end_pos / ratio;
    let index_k = &shared.index_k[..width * ihd];
    let wscale = (ihd as f32).powf(-0.5) * (nh as f32).powf(-0.5);
    let weights: Vec<f32> = linear_bf16(x, w.weights_proj, seqlen, c.dim, nh)
        .into_iter()
        .map(|v| to_bf16_rne(v * wscale))
        .collect();

    // index_score[q][t] = sum_h bf16(relu(bf16(q_h . k_t)) * weights[h]), each stage bf16
    let mut score = vec![0f32; seqlen * width];
    for t in 0..seqlen {
        for p in 0..width {
            let kr = &index_k[p * ihd..(p + 1) * ihd];
            let mut acc = 0f32;
            for h in 0..nh {
                let qv = &q[(t * nh + h) * ihd..(t * nh + h + 1) * ihd];
                let dot = to_bf16_rne(qv.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>());
                acc += to_bf16_rne(dot.max(0.0) * weights[t * nh + h]);
            }
            score[t * width + p] = to_bf16_rne(acc);
        }
    }
    let compress_lens: Vec<usize> = if start_pos == 0 {
        (0..seqlen).map(|t| (t + 1) / ratio).collect()
    } else {
        vec![end_pos / ratio; seqlen]
    };
    if start_pos == 0 {
        for t in 0..seqlen {
            for p in compress_lens[t]..width {
                score[t * width + p] = f32::NEG_INFINITY;
            }
        }
    }
    if is_candidate_source {
        shared.candidates = select_candidate_blocks(
            &score,
            seqlen,
            width,
            &compress_lens,
            c.cand_topk_blocks,
            c.cand_block,
        );
        shared.cand_width = width;
    } else if uses_candidates {
        for i in 0..seqlen * width {
            if !shared.candidates[i] {
                score[i] = f32::NEG_INFINITY;
            }
        }
    }
    let topk = c.index_topk.min(end_pos / ratio);
    let mut out = Vec::with_capacity(seqlen * topk);
    for t in 0..seqlen {
        let row = &score[t * width..(t + 1) * width];
        let mut picked = torch_cpu_topk_set(row, topk);
        picked.sort_unstable();
        for i in picked {
            out.push(if i < compress_lens[t] {
                (i + offset) as i32
            } else {
                -1
            });
        }
    }
    (out, topk)
}

pub struct CompAttnCfg {
    pub dim: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_rank: usize,
    pub o_rank: usize,
    pub groups: usize,
    pub window: usize,
    pub eps: f32,
    pub ratio: usize,
    pub is_kv_source: bool,
    pub is_index_source: bool,
    pub is_candidate_source: bool,
    pub uses_candidates: bool,
}

pub struct LayerAttnState {
    pub window: Vec<f32>,
    pub compressor: Option<CompressorState>,
    pub compress_kv_cache: Option<Vec<f32>>,
    pub k_cache: Option<Vec<f32>>,
}

impl LayerAttnState {
    pub fn new(c: &CompAttnCfg, max_seq: usize, index_hd: usize) -> Self {
        let groups = if c.ratio > 0 { max_seq / c.ratio } else { 0 };
        LayerAttnState {
            window: vec![0f32; c.window * c.head_dim],
            compressor: if c.is_kv_source {
                Some(CompressorState::new(c.ratio.max(1), c.head_dim))
            } else {
                None
            },
            compress_kv_cache: if c.is_kv_source {
                Some(vec![0f32; groups * c.head_dim])
            } else {
                None
            },
            k_cache: if c.is_kv_source {
                Some(vec![0f32; groups * index_hd])
            } else {
                None
            },
        }
    }
}

pub struct CompAttnRun {
    pub q: Vec<f32>,
    pub kv_rows: Vec<f32>,
    pub idx: Vec<i32>,
    pub topk: usize,
    pub o: Vec<f32>,
    pub out: Vec<f32>,
}

/// `Attention.forward` for any `compress_ratio`, mutating the layer state and the shared slots.
#[allow(clippy::too_many_arguments)]
pub fn attention_any(
    x: &[f32],
    seqlen: usize,
    start_pos: usize,
    w: &super::attn::AttnWeights,
    comp: Option<&CompressorWeights>,
    idxw: Option<&IndexerWeights>,
    icfg: &IndexerCfg,
    c: &CompAttnCfg,
    fc: &[(f32, f32)],
    st: &mut LayerAttnState,
    shared: &mut SharedRuntime,
) -> CompAttnRun {
    let (hd, rd, nh) = (c.head_dim, c.rope_dim, c.n_heads);
    let pos: Vec<usize> = (0..seqlen).map(|t| start_pos + t).collect();
    let head_pos: Vec<usize> = pos
        .iter()
        .flat_map(|&p| std::iter::repeat(p).take(nh))
        .collect();

    let qr = rms_norm(
        &linear_bf16(x, w.wq_a, seqlen, c.dim, c.q_rank),
        w.q_norm,
        seqlen,
        c.q_rank,
        c.eps,
    );
    let mut q = linear_bf16(&qr, w.wq_b, seqlen, c.q_rank, nh * hd);
    apply_rotary(&mut q, hd, rd, &head_pos, fc, false);

    let mut kv = rms_norm(
        &linear_bf16(x, w.wkv, seqlen, c.dim, hd),
        w.kv_norm,
        seqlen,
        hd,
        c.eps,
    );
    apply_rotary(&mut kv, hd, rd, &pos, fc, false);
    act_quant_inplace(&mut kv);
    let win = c.window;
    let mut kv_rows = if start_pos == 0 {
        if seqlen <= win {
            st.window[..seqlen * hd].copy_from_slice(&kv);
        } else {
            let cutoff = seqlen % win;
            let tail = &kv[(seqlen - win) * hd..];
            st.window[cutoff * hd..win * hd].copy_from_slice(&tail[..(win - cutoff) * hd]);
            st.window[..cutoff * hd].copy_from_slice(&tail[(win - cutoff) * hd..]);
        }
        kv.clone()
    } else {
        let slot = start_pos % win;
        st.window[slot * hd..(slot + 1) * hd].copy_from_slice(&kv);
        st.window.clone()
    };
    let (mut idx, mut topk) = window_topk_idxs(win, seqlen, start_pos);

    if c.ratio > 0 {
        let ratio = c.ratio;
        let offset = kv_rows.len() / hd;
        let compress_len = (start_pos + seqlen) / ratio;
        let latent = if c.is_kv_source {
            let r = compressor(
                x,
                seqlen,
                start_pos,
                ratio,
                c.dim,
                hd,
                comp.expect("compressor weights"),
                st.compressor.as_mut().expect("compressor state"),
                c.eps,
            );
            shared.compress_kv = st.compress_kv_cache.as_ref().expect("cache").clone();
            r
        } else {
            None
        };
        // the indexer needs the latent before RoPE, so it runs before the cache is written
        let (cidx, ctopk) = if !c.is_index_source {
            (shared.topk_idxs.clone(), shared.topk)
        } else if compress_len == 0 {
            (Vec::new(), 0)
        } else {
            let r = indexer(
                x,
                &qr,
                latent.as_deref(),
                seqlen,
                start_pos,
                offset,
                ratio,
                icfg,
                idxw.expect("indexer weights"),
                fc,
                st.k_cache.as_mut(),
                shared,
                c.is_candidate_source,
                c.uses_candidates,
            );
            shared.topk_idxs = r.0.clone();
            shared.topk = r.1;
            r
        };
        if let Some(mut lat) = latent {
            let groups = lat.len() / hd;
            let lpos: Vec<usize> = if start_pos == 0 {
                (0..groups).map(|g| g * ratio).collect()
            } else {
                vec![start_pos + 1 - ratio]
            };
            apply_rotary(&mut lat, hd, rd, &lpos, fc, false);
            fp4_quant_e4m3_inplace(&mut lat, LATENT_BLOCK);
            let cache = st.compress_kv_cache.as_mut().expect("cache");
            let at = start_pos / ratio;
            cache[at * hd..(at + groups) * hd].copy_from_slice(&lat);
            shared.compress_kv = cache.clone();
        }
        // read after the write
        kv_rows.extend_from_slice(&shared.compress_kv[..compress_len * hd]);
        let mut merged = Vec::with_capacity(seqlen * (topk + ctopk));
        for t in 0..seqlen {
            merged.extend_from_slice(&idx[t * topk..(t + 1) * topk]);
            merged.extend_from_slice(&cidx[t * ctopk..(t + 1) * ctopk]);
        }
        idx = merged;
        topk += ctopk;
    }

    let scale = (hd as f32).powf(-0.5);
    let o_sa = sparse_attn(&q, &kv_rows, w.sink, &idx, seqlen, nh, hd, topk, scale);
    let mut o = o_sa.clone();
    apply_rotary(&mut o, hd, rd, &head_pos, fc, true);
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
    CompAttnRun {
        q,
        kv_rows,
        idx,
        topk,
        o: o_sa,
        out,
    }
}

#[cfg(test)]
#[path = "compress_tests.rs"]
mod tests;
