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

mod compressor;
mod indexer;

pub use compressor::{CompressorState, CompressorWeights, compressor};
pub use indexer::{
    IndexerCfg, IndexerWeights, SharedRuntime, indexer, select_candidate_blocks, torch_cpu_topk_set,
};

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
    let pick_hi = dhi < dlo || (dhi == dlo && hi.is_multiple_of(2) && lo % 2 == 1);
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
        .flat_map(|&p| std::iter::repeat_n(p, nh))
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
