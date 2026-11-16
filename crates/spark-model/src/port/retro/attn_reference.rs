// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026), Section 4.2, Eq 2-5.

//! Responsibility: the f64 CPU reference of the estimation attention on the device layouts
//! (exact buffer, zone lists, BF16 index sections), the target the GPU microtest compares
//! `retro_attn_ada.cu` against. The unit tests check it against L33's oracle
//! (`oracle_eval::retro_attention`).

use super::HEAD_DIM;
use crate::port::retro::gather::HostExact;

/// BF16 bits to f32.
pub fn bf16_f32(bits: u16) -> f32 {
    half::bf16::from_bits(bits).to_f32()
}

/// One attention launch on the host. BF16 tensors are raw bits.
pub struct HostAttnIn<'a> {
    /// `[rows][q_row_stride]`.
    pub q: &'a [u16],
    pub q_row_stride: usize,
    pub rows: usize,
    /// Entries the last row drops; row `r` drops `row_trim + rows - 1 - r`.
    pub row_trim: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub scale: f32,
    pub exact: &'a HostExact,
    /// `[kv_heads][es_cap]` estimation ids, `[kv_heads][2]` counts.
    pub est_ids: &'a [i32],
    pub counts: &'a [i32],
    pub es_cap: usize,
    /// `[kv_heads][cap_centroids][256]` BF16, the same for value sums, `[kv_heads][cap]` sizes.
    pub centroids: &'a [u16],
    pub value_sum: &'a [u16],
    pub cluster_size: &'a [i32],
    pub cap_centroids: usize,
}

fn row_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter().map(|&b| f64::from(bf16_f32(b))).collect()
}

fn dot(q: &[f64], k: &[u16]) -> f64 {
    q.iter()
        .zip(k)
        .map(|(x, &y)| x * f64::from(bf16_f32(y)))
        .sum()
}

/// The (score, weight, value) terms of query head `j` of row `r`.
fn terms(a: &HostAttnIn, r: usize, j: usize) -> Vec<(f64, f64, Vec<f64>)> {
    let hd = HEAD_DIM as usize;
    let group = (a.q_heads / a.kv_heads.max(1)).max(1);
    let h = j / group;
    let cap = a.exact.cap as usize;
    let scale = f64::from(a.scale);
    let q = row_f64(&a.q[r * a.q_row_stride + j * hd..][..hd]);
    let trim = (a.row_trim + a.rows - 1 - r) as i64;
    let n = (i64::from(a.exact.valid[h].clamp(0, cap as i32)) - trim).max(0) as usize;
    let mut out = Vec::new();
    for e in 0..n {
        if a.exact.pos[h * cap + e] < 0 {
            continue;
        }
        let at = (h * cap + e) * hd;
        let s = dot(&q, &a.exact.k[at..at + hd]) * scale;
        out.push((s, 1.0, row_f64(&a.exact.v[at..at + hd])));
    }
    let nc = a.counts[h * 2 + 1].clamp(0, a.es_cap as i32) as usize;
    for &id in &a.est_ids[h * a.es_cap..][..nc] {
        if id < 0 || id as usize >= a.cap_centroids {
            continue;
        }
        let row = h * a.cap_centroids + id as usize;
        let size = a.cluster_size[row];
        if size <= 0 {
            continue;
        }
        let s = dot(&q, &a.centroids[row * hd..][..hd]) * scale;
        out.push((s, f64::from(size), row_f64(&a.value_sum[row * hd..][..hd])));
    }
    out
}

/// The combined formula in f64. Returns `out [rows][q_heads][256]` and `lse [rows][q_heads]`
/// (zeros and negative infinity where nothing is attended).
pub fn reference_attention(a: &HostAttnIn) -> (Vec<f32>, Vec<f32>) {
    let hd = HEAD_DIM as usize;
    let mut out = vec![0f32; a.rows * a.q_heads * hd];
    let mut lse = vec![f32::NEG_INFINITY; a.rows * a.q_heads];
    for r in 0..a.rows {
        for j in 0..a.q_heads {
            let t = terms(a, r, j);
            if t.is_empty() {
                continue;
            }
            let m = t.iter().map(|x| x.0).fold(f64::NEG_INFINITY, f64::max);
            let mut num = vec![0f64; hd];
            let mut den = 0f64;
            for (s, w, v) in &t {
                let e = (s - m).exp();
                den += w * e;
                for (n, x) in num.iter_mut().zip(v) {
                    *n += e * x;
                }
            }
            let at = (r * a.q_heads + j) * hd;
            for (o, n) in out[at..at + hd].iter_mut().zip(&num) {
                *o = (n / den) as f32;
            }
            lse[r * a.q_heads + j] = (m + den.ln()) as f32;
        }
    }
    (out, lse)
}
