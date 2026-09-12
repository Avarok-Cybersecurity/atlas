// SPDX-License-Identifier: AGPL-3.0-only

//! Stable LatentMoE CPU reference.
//!
//! ```text
//! latent = down_proj(h)                 // hidden → moe_latent
//! latent = RMSNorm(latent)              // if latent_moe_use_norm
//! scores = sigmoid(router(h) + bias)    // noaux_tc
//! top-k, optional renormalize
//! y = Σ w_i expert_i(latent)            // SiTU-GLU experts
//! y = RMSNorm(y); y = up_proj(y)
//! out = y + shared_experts(h)           // shared stay full-width
//! ```

#![allow(clippy::too_many_arguments)]

use super::attnres::rms_norm;
use super::situ::{sigmoid, situ_glu_vec};

#[derive(Clone, Copy, Debug)]
pub struct LatentMoeConfig {
    pub hidden: usize,
    pub latent: usize,
    pub expert_hidden: usize,
    pub n_routed: usize,
    pub top_k: usize,
    pub n_shared: usize,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub use_norm: bool,
    pub renormalize: bool,
}

impl LatentMoeConfig {
    pub fn production() -> Self {
        Self {
            hidden: 7168,
            latent: 3584,
            expert_hidden: 3072,
            n_routed: 896,
            top_k: 16,
            n_shared: 2,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            use_norm: true,
            renormalize: true,
        }
    }
}

/// Sigmoid + correction-bias scores, then top-k (stable: higher score, then lower id).
pub fn sigmoid_topk(logits: &[f32], bias: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(logits.len(), bias.len());
    let n = logits.len();
    let k = k.min(n);
    let scores: Vec<f32> = logits
        .iter()
        .zip(bias)
        .map(|(l, b)| sigmoid(l + b))
        .collect();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    let mut w: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
    let z: f32 = w.iter().sum();
    if z > 0.0 {
        for ww in &mut w {
            *ww /= z;
        }
    }
    (idx, w)
}

/// One SiTU-GLU expert: `down( situ(w1 x, w3 x) )`. Weights are `[out, in]`.
pub fn expert_situ(
    x: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    in_dim: usize,
    hidden: usize,
    beta: f32,
    beta_lin: f32,
) -> Vec<f32> {
    let gate = matvec(w1, x, hidden, in_dim);
    let up = matvec(w3, x, hidden, in_dim);
    let mid = situ_glu_vec(&gate, &up, beta, beta_lin);
    matvec(w2, &mid, in_dim, hidden)
}

fn matvec(w: &[f32], x: &[f32], out: usize, inn: usize) -> Vec<f32> {
    assert_eq!(w.len(), out * inn);
    assert_eq!(x.len(), inn);
    let mut y = vec![0.0f32; out];
    for o in 0..out {
        let mut acc = 0.0f32;
        let row = &w[o * inn..(o + 1) * inn];
        for i in 0..inn {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
    y
}

/// Routed latent path + optional shared expert (identity-scale for tests).
pub fn latent_moe_forward(
    h: &[f32],
    down: &[f32],
    up: &[f32],
    norm_w: &[f32],
    logits: &[f32],
    bias: &[f32],
    experts: &[(Vec<f32>, Vec<f32>, Vec<f32>)],
    shared: Option<&[f32]>,
    cfg: &LatentMoeConfig,
    eps: f32,
) -> (Vec<f32>, Vec<usize>) {
    let mut latent = matvec(down, h, cfg.latent, cfg.hidden);
    if cfg.use_norm {
        latent = rms_norm(&latent, norm_w, eps);
    }
    let (ids, weights) = sigmoid_topk(logits, bias, cfg.top_k);
    let mut mixed = vec![0.0f32; cfg.latent];
    for (&id, &w) in ids.iter().zip(&weights) {
        let (w1, w2, w3) = &experts[id];
        let y = expert_situ(
            &latent,
            w1,
            w2,
            w3,
            cfg.latent,
            cfg.expert_hidden,
            cfg.situ_beta,
            cfg.situ_linear_beta,
        );
        for (m, yy) in mixed.iter_mut().zip(y) {
            *m += w * yy;
        }
    }
    if cfg.use_norm {
        mixed = rms_norm(&mixed, norm_w, eps);
    }
    let mut out = matvec(up, &mixed, cfg.hidden, cfg.latent);
    if let Some(s) = shared {
        for (o, ss) in out.iter_mut().zip(s) {
            *o += ss;
        }
    }
    (out, ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_topk_picks_highest() {
        let logits = [0.1, 5.0, 0.2, 4.0];
        let bias = [0.0, 0.0, 0.0, 0.0];
        let (ids, w) = sigmoid_topk(&logits, &bias, 2);
        assert_eq!(ids, vec![1, 3]);
        assert!((w[0] + w[1] - 1.0).abs() < 1e-6);
        assert!(w[0] > w[1]);
    }

    #[test]
    fn force_expert_zero_mutant_diverges() {
        // C6 known-bad: forcing expert 0 must change the mix vs true top-k.
        let cfg = LatentMoeConfig {
            hidden: 2,
            latent: 2,
            expert_hidden: 2,
            n_routed: 2,
            top_k: 1,
            n_shared: 0,
            situ_beta: 4.0,
            situ_linear_beta: 25.0,
            use_norm: false,
            renormalize: true,
        };
        let h = vec![1.0, 0.0];
        let down = vec![1.0, 0.0, 0.0, 1.0];
        let up = down.clone();
        let ident = |out: usize, inn: usize| -> Vec<f32> {
            let mut w = vec![0.0; out * inn];
            for i in 0..out.min(inn) {
                w[i * inn + i] = 1.0;
            }
            w
        };
        let e0 = (ident(2, 2), ident(2, 2), ident(2, 2));
        // Expert 1 scales up-branch so SiTU output differs.
        let mut w3 = ident(2, 2);
        w3[0] = 3.0;
        let e1 = (ident(2, 2), ident(2, 2), w3);
        let experts = [e0, e1];
        let logits = [0.0, 4.0];
        let bias = [0.0, 0.0];
        let (y, ids) = latent_moe_forward(
            &h,
            &down,
            &up,
            &[1.0, 1.0],
            &logits,
            &bias,
            &experts,
            None,
            &cfg,
            1e-5,
        );
        assert_eq!(ids, vec![1]);
        let (y0, _) = latent_moe_forward(
            &h,
            &down,
            &up,
            &[1.0, 1.0],
            &[4.0, 0.0],
            &bias,
            &experts,
            None,
            &cfg,
            1e-5,
        );
        let err: f32 = y.iter().zip(&y0).map(|(a, b)| (a - b).abs()).sum();
        assert!(err > 1e-4, "forcing expert 0 must diverge, err={err}");
    }
}
