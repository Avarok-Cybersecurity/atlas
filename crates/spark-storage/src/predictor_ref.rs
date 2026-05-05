// SPDX-License-Identifier: AGPL-3.0-only
//
// Pure-Rust reference implementation of the predictor math. Used to validate
// the CUDA kernels (BF16 numerics + reduction order parity) and to compute
// ground-truth scores for the recall@K test.
//
// Tensor layout matches the GPU kernels exactly:
//   Q       : [num_q_heads, head_dim]
//   K_block : [block_size,  num_kv_heads, head_dim]
//   P       : [head_dim,    r]
//   q_proj  : [num_q_heads, r]
//   A_g     : [num_blocks,  num_kv_heads, r]
//   scores  : [num_blocks]   (max over q_heads of dot product, lossless mode)

use half::bf16;

#[inline]
fn b2f(x: bf16) -> f32 {
    x.to_f32()
}
#[inline]
fn f2b(x: f32) -> bf16 {
    bf16::from_f32(x)
}

pub fn project_q_ref(
    q: &[bf16],
    p: &[bf16],
    num_q_heads: usize,
    head_dim: usize,
    r: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); num_q_heads * r];
    for h in 0..num_q_heads {
        for o in 0..r {
            let mut acc = 0.0_f32;
            for i in 0..head_dim {
                acc += b2f(q[h * head_dim + i]) * b2f(p[i * r + o]);
            }
            out[h * r + o] = f2b(acc);
        }
    }
    out
}

/// Per-token projection. Output layout `[num_kv_heads, block_size, r]`.
pub fn project_kv_block_ref(
    k_block: &[bf16],
    p: &[bf16],
    block_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
    r: usize,
) -> Vec<bf16> {
    let mut out = vec![bf16::from_f32(0.0); num_kv_heads * block_size * r];
    for kh in 0..num_kv_heads {
        for tok in 0..block_size {
            let k_base = (tok * num_kv_heads + kh) * head_dim;
            for o in 0..r {
                let mut acc = 0.0_f32;
                for i in 0..head_dim {
                    acc += b2f(k_block[k_base + i]) * b2f(p[i * r + o]);
                }
                out[(kh * block_size + tok) * r + o] = f2b(acc);
            }
        }
    }
    out
}

pub fn predictor_score_ref(
    q_proj: &[bf16],
    k_lr_seq: &[bf16],
    num_q_heads: usize,
    num_kv_heads: usize,
    block_size: usize,
    r: usize,
    num_active_blocks: usize,
) -> Vec<f32> {
    assert!(num_q_heads.is_multiple_of(num_kv_heads));
    let gqa = num_q_heads / num_kv_heads;
    let per_block = num_kv_heads * block_size * r;
    let mut scores = vec![f32::NEG_INFINITY; num_active_blocks];
    for blk in 0..num_active_blocks {
        let mut best = f32::NEG_INFINITY;
        for qh in 0..num_q_heads {
            let kh = qh / gqa;
            for tok in 0..block_size {
                let mut dot = 0.0_f32;
                for i in 0..r {
                    let q = b2f(q_proj[qh * r + i]);
                    let k = b2f(k_lr_seq[blk * per_block + (kh * block_size + tok) * r + i]);
                    dot += q * k;
                }
                if dot > best {
                    best = dot;
                }
            }
        }
        scores[blk] = best;
    }
    scores
}

/// Ground-truth attention-weight per block: max over q_heads of the
/// softmax-normalized attention to the block's tokens. Used by the recall
/// test as the "oracle" the predictor's scores are compared against.
pub fn ground_truth_block_weights(
    q: &[bf16],
    k: &[bf16],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    block_size: usize,
    num_blocks: usize,
) -> Vec<f32> {
    let gqa = num_q_heads / num_kv_heads;
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let total_tokens = num_blocks * block_size;
    let mut block_weights = vec![0.0_f32; num_blocks];

    // Per-query-head softmax over all tokens, then per-block sum, then max.
    #[allow(clippy::needless_range_loop)]
    for qh in 0..num_q_heads {
        let kh = qh / gqa;
        // Logits = q_h · k_t  for each t.
        let mut logits = vec![0.0_f32; total_tokens];
        for t in 0..total_tokens {
            let blk = t / block_size;
            let off = t % block_size;
            let k_base = (off * num_kv_heads + kh) * head_dim;
            let mut dot = 0.0_f32;
            for i in 0..head_dim {
                dot += b2f(q[qh * head_dim + i])
                    * b2f(k[blk * block_size * num_kv_heads * head_dim + k_base + i]);
            }
            logits[t] = dot * scale;
        }
        let lmax = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|l| (l - lmax).exp()).sum();
        for blk in 0..num_blocks {
            let mut bw = 0.0_f32;
            for off in 0..block_size {
                let t = blk * block_size + off;
                bw += (logits[t] - lmax).exp() / sum;
            }
            if bw > block_weights[blk] {
                block_weights[blk] = bw;
            }
        }
    }
    block_weights
}
