// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the f64 CPU references of the bounded view (lane L31): RoPE forward and
//! inverse with the partial-rotary NeoX pairing the Atlas K write uses
//! (`fused_k_norm_rope_cache.cu`: pairs `(d, d + rotary_dim / 2)`, frequency
//! `theta^(-2 d / rotary_dim)`), the pre-RoPE page mean, the count-weighted block merge, the
//! page-softmax scorer in both GQA modes, and an independent brute-force picker. The unit
//! tests and the GPU microtest (`spark-model/examples/kv_view_microtest.rs`) compare the
//! device kernels against these. Public because the example is another crate.

use super::{GqaAgg, PickCfg};

/// RoPE frequency of rotary pair `pair`.
pub fn rope_freq(pair: usize, rotary_dim: usize, theta: f64) -> f64 {
    1.0 / theta.powf((2 * pair) as f64 / rotary_dim as f64)
}

/// Rotate the first `rotary_dim` dims of one head vector by position `pos` in place.
pub fn rope_forward(x: &mut [f64], pos: u32, rotary_dim: usize, theta: f64) {
    let half = rotary_dim / 2;
    for i in 0..half {
        let a = f64::from(pos) * rope_freq(i, rotary_dim, theta);
        let (s, c) = a.sin_cos();
        let (x0, x1) = (x[i], x[i + half]);
        x[i] = x0 * c - x1 * s;
        x[i + half] = x1 * c + x0 * s;
    }
}

/// Undo [`rope_forward`] at the same position.
pub fn rope_inverse(x: &mut [f64], pos: u32, rotary_dim: usize, theta: f64) {
    let half = rotary_dim / 2;
    for i in 0..half {
        let a = f64::from(pos) * rope_freq(i, rotary_dim, theta);
        let (s, c) = a.sin_cos();
        let (y0, y1) = (x[i], x[i + half]);
        x[i] = y0 * c + y1 * s;
        x[i + half] = y1 * c - y0 * s;
    }
}

/// RMSNorm with the Qwen3 `(1 + w)` weight at `w = 0`: `x / sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len().max(1) as f64;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * r).collect()
}

/// Mean of `n` head vectors of `head_dim` stored back to back.
pub fn mean_rows(rows: &[f64], n: usize, head_dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; head_dim];
    if n == 0 {
        return out;
    }
    for t in 0..n {
        for (o, v) in out.iter_mut().zip(&rows[t * head_dim..(t + 1) * head_dim]) {
            *o += v;
        }
    }
    out.iter_mut().for_each(|o| *o /= n as f64);
    out
}

/// The pre-RoPE mean of one page from its stored (post-RoPE) keys: `post` holds `fill`
/// head vectors at positions `pos0, pos0 + 1, ...`.
pub fn page_mean_pre_rope(
    post: &[f64],
    fill: usize,
    head_dim: usize,
    pos0: u32,
    rotary_dim: usize,
    theta: f64,
) -> Vec<f64> {
    let mut pre = post[..fill * head_dim].to_vec();
    for t in 0..fill {
        rope_inverse(
            &mut pre[t * head_dim..(t + 1) * head_dim],
            pos0 + t as u32,
            rotary_dim,
            theta,
        );
    }
    mean_rows(&pre, fill, head_dim)
}

/// Count-weighted mean of page means; `None` when every count is 0.
pub fn merge_means(means: &[&[f64]], counts: &[u32]) -> Option<Vec<f64>> {
    let total: u64 = counts.iter().map(|&c| u64::from(c)).sum();
    if total == 0 || means.is_empty() {
        return None;
    }
    let mut out = vec![0.0; means[0].len()];
    for (m, &c) in means.iter().zip(counts) {
        for (o, v) in out.iter_mut().zip(m.iter()) {
            *o += f64::from(c) * v;
        }
    }
    out.iter_mut().for_each(|o| *o /= total as f64);
    Some(out)
}

/// One sequence's sketch in f64, indexed by the page's position in the sequence.
#[derive(Clone, Debug, Default)]
pub struct SketchF64 {
    pub n_pages: usize,
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `[n_pages, layers, kv_heads, head_dim]`.
    pub means: Vec<f64>,
    /// `[n_pages]` tokens behind each mean.
    pub counts: Vec<u32>,
}

impl SketchF64 {
    pub fn mean(&self, page: usize, layer: usize, kv_head: usize) -> &[f64] {
        let at = ((page * self.layers + layer) * self.kv_heads + kv_head) * self.head_dim;
        &self.means[at..at + self.head_dim]
    }
}

/// Captured query rows in f64.
#[derive(Clone, Debug, Default)]
pub struct QueryF64 {
    pub rows: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub head_dim: usize,
    /// `[layers, rows, q_heads, head_dim]`.
    pub data: Vec<f64>,
}

impl QueryF64 {
    pub fn row(&self, layer: usize, row: usize, head: usize) -> &[f64] {
        let at = ((layer * self.rows + row) * self.q_heads + head) * self.head_dim;
        &self.data[at..at + self.head_dim]
    }
}

/// The page-softmax scores of every view block (see the module docs of `kv_select`).
pub fn score_blocks_f64(
    s: &SketchF64,
    q: &QueryF64,
    pages_per_block: usize,
    gqa: GqaAgg,
) -> Vec<f64> {
    let nb = super::n_blocks(s.n_pages, pages_per_block);
    let group = q.q_heads / s.kv_heads;
    let scale = 1.0 / (s.head_dim as f64).sqrt();
    let mut scores = vec![0.0f64; nb];
    for l in 0..s.layers {
        let merged: Vec<Vec<Option<Vec<f64>>>> = (0..nb)
            .map(|b| {
                let pages = b * pages_per_block..((b + 1) * pages_per_block).min(s.n_pages);
                (0..s.kv_heads)
                    .map(|g| {
                        let ms: Vec<&[f64]> = pages.clone().map(|p| s.mean(p, l, g)).collect();
                        merge_means(&ms, &s.counts[pages.clone()])
                    })
                    .collect()
            })
            .collect();
        for r in 0..q.rows {
            let probs: Vec<Vec<f64>> = (0..q.q_heads)
                .map(|h| {
                    let qv = q.row(l, r, h);
                    let logits: Vec<Option<f64>> = merged
                        .iter()
                        .map(|mb| {
                            mb[h / group]
                                .as_ref()
                                .map(|m| qv.iter().zip(m).map(|(a, b)| a * b).sum::<f64>() * scale)
                        })
                        .collect();
                    softmax(&logits)
                })
                .collect();
            match gqa {
                GqaAgg::Mean => {
                    let w = 1.0 / (s.layers * q.q_heads) as f64;
                    for p in &probs {
                        for (sc, v) in scores.iter_mut().zip(p) {
                            *sc += v * w;
                        }
                    }
                }
                GqaAgg::Max => {
                    let w = 1.0 / (s.layers * s.kv_heads) as f64;
                    for g in 0..s.kv_heads {
                        let mx: Vec<f64> = (0..nb)
                            .map(|b| {
                                (0..group)
                                    .map(|i| probs[g * group + i][b])
                                    .fold(0.0, f64::max)
                            })
                            .collect();
                        let z: f64 = mx.iter().sum();
                        if z > 0.0 {
                            for (sc, v) in scores.iter_mut().zip(&mx) {
                                *sc += v / z * w;
                            }
                        }
                    }
                }
            }
        }
    }
    scores
}

/// Softmax over the eligible (`Some`) entries; ineligible entries get 0. All zero when
/// nothing is eligible.
pub fn softmax(logits: &[Option<f64>]) -> Vec<f64> {
    let mx = logits
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if mx == f64::NEG_INFINITY {
        return vec![0.0; logits.len()];
    }
    let e: Vec<f64> = logits
        .iter()
        .map(|l| l.map_or(0.0, |v| (v - mx).exp()))
        .collect();
    let z: f64 = e.iter().sum();
    e.iter().map(|v| v / z).collect()
}

/// An independent picker: rank every block by (class, key) at once and take the first
/// `budget_blocks`. Class 0 tail, 1 recent (newest first), 2 sink (oldest first), 3
/// mandatory (newest first), 4 scored (higher score first, then newer; NaN last).
pub fn pick_bruteforce(scores: &[f32], n_blocks: usize, cfg: &PickCfg) -> Vec<u32> {
    if n_blocks == 0 || cfg.budget_blocks == 0 {
        return Vec::new();
    }
    let recent_from = n_blocks.saturating_sub(cfg.recent_blocks);
    let mut keyed: Vec<(u8, f64, usize)> = (0..n_blocks)
        .map(|b| {
            let class = if b == n_blocks - 1 {
                0
            } else if b >= recent_from {
                1
            } else if b < cfg.sink_blocks {
                2
            } else if cfg.mandatory.contains(&(b as u32)) {
                3
            } else {
                4
            };
            let key = match class {
                1 | 3 => -(b as f64),
                2 => b as f64,
                4 => {
                    let s = scores.get(b).copied().unwrap_or(f32::NAN);
                    if s.is_nan() {
                        f64::INFINITY
                    } else {
                        -f64::from(s)
                    }
                }
                _ => 0.0,
            };
            (class, key, b)
        })
        .collect();
    keyed.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.total_cmp(&y.1)).then(y.2.cmp(&x.2)));
    let mut out: Vec<u32> = keyed
        .iter()
        .take(cfg.budget_blocks)
        .map(|k| k.2 as u32)
        .collect();
    out.sort_unstable();
    out
}

/// FP8 E4M3 (OCP, saturating, no infinities) to f32.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = (b >> 3) & 0x0f;
    let man = b & 0x07;
    if exp == 0x0f && man == 0x07 {
        return f32::NAN;
    }
    let v = if exp == 0 {
        f32::from(man) / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + f32::from(man) / 8.0) * 2f32.powi(i32::from(exp) - 7)
    };
    sign * v
}

/// Round to BF16 and back (round to nearest even).
pub fn bf16_round(x: f64) -> f64 {
    half::bf16::from_f64(x).to_f64()
}
