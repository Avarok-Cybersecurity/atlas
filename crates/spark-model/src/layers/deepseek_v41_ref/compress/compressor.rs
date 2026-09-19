// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The `Compressor` of DeepSeek-V4.1 compressed attention: `Compressor.forward` (softmax pooling
//! of `ratio` tokens into one latent, or a plain projection at ratio 1) with its weights and the
//! decode-time fill state. Split out of `compress.rs` for the file-size cap; the quantisers and
//! `linear_f32` it needs stay there.

use super::super::hc::rms_norm;
use super::super::moe::linear_bf16;
use super::super::to_bf16_rne;
use super::linear_f32;

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
        if !(start_pos + 1).is_multiple_of(ratio) {
            return None;
        }
        pool(&st.kv_state, &st.score_state)
    };
    let groups = pooled.len() / hd;
    let as_bf16: Vec<f32> = pooled.into_iter().map(to_bf16_rne).collect();
    Some(rms_norm(&as_bf16, w.norm, groups, hd, eps))
}
