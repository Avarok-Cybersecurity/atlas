// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after XAttention (antidiagonal block scoring for block-sparse prefill), described in its paper;
// independent implementation, no source copied (the reference repository has no detected license).

//! Responsibility: the FP64 CPU reference of block-sparse prefill: antidiagonal block scoring, block
//! masses, the selection masks, and causal attention restricted to a block mask (the dense reference is
//! the all-blocks case). It restates `prefill_block_sparse_ada.cu`'s rules in the same order, on
//! contiguous K/V (the paged layout is the kernel's addressing, which the GPU microtest exercises).

use super::BLOCK;
use super::select;

/// One chunked-prefill attention problem. Rows are FP32 values (the GPU's BF16 inputs widened; FP8
/// keys and values already multiplied by their scale).
#[derive(Clone, Copy, Debug)]
pub struct Problem<'a> {
    /// `[q_len][nq][hd]`: the chunk's queries, absolute positions `q_offset..q_offset + q_len`.
    pub q: &'a [f32],
    /// `[kv_len][nkv][hd]`.
    pub k: &'a [f32],
    /// `[kv_len][nkv][hd]`.
    pub v: &'a [f32],
    pub q_len: usize,
    pub kv_len: usize,
    pub q_offset: usize,
    pub nq: usize,
    pub nkv: usize,
    pub hd: usize,
    pub inv_sqrt_d: f64,
}

impl Problem<'_> {
    /// The query row of absolute position `pos`, head `h`, if it is in this chunk.
    pub fn q_row(&self, pos: usize, h: usize) -> Option<&[f32]> {
        if pos < self.q_offset || pos >= self.q_offset + self.q_len {
            return None;
        }
        let at = ((pos - self.q_offset) * self.nq + h) * self.hd;
        Some(&self.q[at..at + self.hd])
    }

    /// The key row of position `pos`, kv head `g`, if `pos < kv_len`.
    pub fn k_row(&self, pos: usize, g: usize) -> Option<&[f32]> {
        if pos >= self.kv_len {
            return None;
        }
        let at = (pos * self.nkv + g) * self.hd;
        Some(&self.k[at..at + self.hd])
    }

    fn v_row(&self, pos: usize, g: usize) -> &[f32] {
        let at = (pos * self.nkv + g) * self.hd;
        &self.v[at..at + self.hd]
    }

    /// The kv head query head `h` reads (GQA).
    pub fn kv_head(&self, h: usize) -> usize {
        h / (self.nq / self.nkv)
    }

    /// First and last query block of the chunk (absolute block indices).
    pub fn query_blocks(&self) -> (usize, usize) {
        let b = BLOCK as usize;
        (self.q_offset / b, (self.q_offset + self.q_len - 1) / b)
    }

    /// Candidate key blocks of query block `i`: `0..=min(i, last key block)`.
    pub fn candidates(&self, i: usize) -> usize {
        i.min((self.kv_len - 1) / BLOCK as usize) + 1
    }
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// Strided row `a` of query block `i` counts only when all `stride` of its rows are in the chunk.
pub fn row_counted(p: &Problem, i: usize, a: usize, stride: usize) -> bool {
    let first = i * BLOCK as usize + stride * a;
    first >= p.q_offset && first + stride <= p.q_offset + p.q_len
}

/// The strided logit of (query block `i`, strided row `a`) and (key block `j`, strided row `b`) for head
/// `h`: `(1/sqrt(d))/S * sum_r q[S*a + S-1-r] . k[S*b + r]`, rows outside the chunk or past `kv_len`
/// reading as zero. `None` when the kernel masks the pair (key group past `kv_len`, or `b > a` on the
/// diagonal).
pub fn strided_logit(
    p: &Problem,
    h: usize,
    i: usize,
    a: usize,
    j: usize,
    b: usize,
    stride: usize,
) -> Option<f64> {
    let bl = BLOCK as usize;
    if j * bl + stride * b >= p.kv_len || (j == i && b > a) {
        return None;
    }
    let g = p.kv_head(h);
    let mut acc = 0.0;
    for r in 0..stride {
        let qp = i * bl + stride * a + (stride - 1 - r);
        let kp = j * bl + stride * b + r;
        if let (Some(q), Some(k)) = (p.q_row(qp, h), p.k_row(kp, g)) {
            acc += dot(q, k);
        }
    }
    Some(acc * p.inv_sqrt_d / stride as f64)
}

/// Per strided row of query block `i`: its log-sum-exp over each candidate key block (`None` for a row
/// that does not count; `-inf` for a block with no unmasked strided key).
pub fn block_lse(p: &Problem, h: usize, i: usize, stride: usize) -> Vec<Option<Vec<f64>>> {
    let r_rows = BLOCK as usize / stride;
    let n_j = p.candidates(i);
    (0..r_rows)
        .map(|a| {
            if !row_counted(p, i, a, stride) {
                return None;
            }
            Some(
                (0..n_j)
                    .map(|j| {
                        let ls: Vec<f64> = (0..r_rows)
                            .filter_map(|b| strided_logit(p, h, i, a, j, b, stride))
                            .collect();
                        log_sum_exp(&ls)
                    })
                    .collect(),
            )
        })
        .collect()
}

/// `log(sum(exp(x)))`, `-inf` for an empty slice.
pub fn log_sum_exp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m == f64::NEG_INFINITY {
        return m;
    }
    m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

/// The block masses of query block `i` (mean over counted strided rows of the softmax probability in
/// each candidate key block; they sum to 1). `None` when no strided row counts: the kernel then keeps
/// every block.
pub fn block_mass(p: &Problem, h: usize, i: usize, stride: usize) -> Option<Vec<f64>> {
    let rows: Vec<Vec<f64>> = block_lse(p, h, i, stride).into_iter().flatten().collect();
    if rows.is_empty() {
        return None;
    }
    let n_j = p.candidates(i);
    let mut mass = vec![0.0; n_j];
    for row in &rows {
        let norm = log_sum_exp(row);
        for (m, x) in mass.iter_mut().zip(row) {
            if x.is_finite() {
                *m += (x - norm).exp();
            }
        }
    }
    let inv = 1.0 / rows.len() as f64;
    mass.iter_mut().for_each(|m| *m *= inv);
    Some(mass)
}

/// The selection of every (head, query block) of the chunk, indexed `h * n_qb + qb` (`qb` relative to
/// the chunk's first query block), each over its candidate blocks.
pub fn masks(p: &Problem, stride: usize, threshold: f64) -> Vec<Vec<bool>> {
    let (lo, hi) = p.query_blocks();
    let mut out = Vec::with_capacity(p.nq * (hi - lo + 1));
    for h in 0..p.nq {
        for i in lo..=hi {
            out.push(match block_mass(p, h, i, stride) {
                Some(m) => select::select_greedy(&m, threshold),
                None => vec![true; p.candidates(i)],
            });
        }
    }
    out
}

/// Causal attention of every chunk row over the keys whose block `keep(h, qb, j)` admits, FP64.
/// Output `[q_len][nq][hd]`; a row with no admitted key is zero.
pub fn attention(p: &Problem, keep: &dyn Fn(usize, usize, usize) -> bool) -> Vec<f64> {
    let bl = BLOCK as usize;
    let (lo, _) = p.query_blocks();
    let mut out = vec![0.0; p.q_len * p.nq * p.hd];
    let mut logits = Vec::with_capacity(p.kv_len);
    for row in 0..p.q_len {
        let pos = p.q_offset + row;
        let qb = pos / bl - lo;
        for h in 0..p.nq {
            let g = p.kv_head(h);
            let q = p.q_row(pos, h).expect("row in chunk");
            logits.clear();
            for kp in 0..=pos.min(p.kv_len - 1) {
                if keep(h, qb, kp / bl) {
                    let k = p.k_row(kp, g).expect("key in range");
                    logits.push((kp, dot(q, k) * p.inv_sqrt_d));
                }
            }
            let m = logits.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
            if m == f64::NEG_INFINITY {
                continue;
            }
            let o = &mut out[(row * p.nq + h) * p.hd..(row * p.nq + h + 1) * p.hd];
            let mut l = 0.0;
            for &(kp, s) in &logits {
                let w = (s - m).exp();
                l += w;
                for (od, vd) in o.iter_mut().zip(p.v_row(kp, g)) {
                    *od += w * *vd as f64;
                }
            }
            o.iter_mut().for_each(|x| *x /= l);
        }
    }
    out
}

/// Plain causal attention over every key (the threshold-1.0 case).
pub fn dense(p: &Problem) -> Vec<f64> {
    attention(p, &|_, _, _| true)
}

/// Attention through the masks `masks(p, ..)` returns (or the GPU's, decoded to the same shape).
pub fn sparse(p: &Problem, masks: &[Vec<bool>]) -> Vec<f64> {
    let (lo, hi) = p.query_blocks();
    let n_qb = hi - lo + 1;
    attention(p, &|h, qb, j| {
        masks[h * n_qb + qb].get(j).copied().unwrap_or(false)
    })
}
