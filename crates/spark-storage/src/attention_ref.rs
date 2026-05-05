// SPDX-License-Identifier: AGPL-3.0-only
//
// Pure-Rust reference for FlashAttention-2 online-softmax decode attention,
// matching the math the tiled CUDA kernel runs. Used to validate that
// kernel-side tile fragmentation produces equivalent output to a single-shot
// computation within float reordering tolerance.

use half::bf16;

#[inline]
fn b2f(x: bf16) -> f32 {
    x.to_f32()
}

#[derive(Clone)]
pub struct AttnState {
    pub m: Vec<f32>, // [num_seqs, num_q_heads]
    pub l: Vec<f32>, // [num_seqs, num_q_heads]
    pub o: Vec<f32>, // [num_seqs, num_q_heads, head_dim]
}

impl AttnState {
    pub fn new(num_seqs: usize, num_q_heads: usize, head_dim: usize) -> Self {
        let n_q = num_seqs * num_q_heads;
        Self {
            m: vec![f32::NEG_INFINITY; n_q],
            l: vec![0.0; n_q],
            o: vec![0.0; n_q * head_dim],
        }
    }
}

/// One tile update. Mirrors the CUDA kernel's per-token online-softmax
/// recurrence exactly (same accumulation order, fp32 throughout).
#[allow(clippy::too_many_arguments)]
pub fn step_tile_ref(
    state: &mut AttnState,
    q: &[bf16],
    k_pool: &[bf16],
    v_pool: &[bf16],
    tile_blocks: &[i32],       // [num_seqs, tile_capacity]
    tile_block_counts: &[i32], // [num_seqs]
    num_seqs: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    tile_capacity: usize,
    gqa_ratio: usize,
) {
    let inv_sqrt_d = 1.0_f32 / (head_dim as f32).sqrt();
    let kv_token_stride = num_kv_heads * head_dim;
    for seq in 0..num_seqs {
        let n_blocks = tile_block_counts[seq] as usize;
        for qh in 0..num_q_heads {
            let kh = qh / gqa_ratio;
            let q_off = (seq * num_q_heads + qh) * head_dim;
            let m_idx = seq * num_q_heads + qh;
            let o_off = m_idx * head_dim;
            let mut m_run = state.m[m_idx];
            let mut l_run = state.l[m_idx];
            let mut o_run: Vec<f32> = state.o[o_off..o_off + head_dim].to_vec();
            for b in 0..n_blocks {
                let blk_id = tile_blocks[seq * tile_capacity + b] as usize;
                let blk_base = blk_id * block_size * kv_token_stride;
                for t in 0..block_size {
                    let kv_base = blk_base + t * kv_token_stride + kh * head_dim;
                    // Compute logit = Q · K_t / sqrt(d)
                    let mut dot = 0.0_f32;
                    for i in 0..head_dim {
                        dot += b2f(q[q_off + i]) * b2f(k_pool[kv_base + i]);
                    }
                    let logit = dot * inv_sqrt_d;
                    let m_new = m_run.max(logit);
                    let scale_old = (m_run - m_new).exp();
                    let scale_new = (logit - m_new).exp();
                    let l_new = l_run * scale_old + scale_new;
                    for i in 0..head_dim {
                        let v = b2f(v_pool[kv_base + i]);
                        o_run[i] = o_run[i] * scale_old + v * scale_new;
                    }
                    m_run = m_new;
                    l_run = l_new;
                }
            }
            state.m[m_idx] = m_run;
            state.l[m_idx] = l_run;
            state.o[o_off..o_off + head_dim].copy_from_slice(&o_run);
        }
    }
}

pub fn finalize_ref(
    state: &AttnState,
    num_seqs: usize,
    num_q_heads: usize,
    head_dim: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); num_seqs * num_q_heads * head_dim];
    for seq in 0..num_seqs {
        for qh in 0..num_q_heads {
            let idx = seq * num_q_heads + qh;
            let l = state.l[idx];
            let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
            for i in 0..head_dim {
                let val = state.o[idx * head_dim + i] * inv_l;
                out[idx * head_dim + i] = bf16::from_f32(val);
            }
        }
    }
    out
}
