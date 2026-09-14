// SPDX-License-Identifier: AGPL-3.0-only

//! Block AttnRes CPU reference.
//!
//! Each sublayer mixes a learned softmax over completed block residuals plus
//! the current intra-block partial sum (Moonshot Block AttnRes):
//!
//! ```text
//! K_i = RMSNorm(V_i)
//! α   = softmax_i( q · K_i )
//! h   = Σ α_i V_i
//! ```
//!
//! `q` is the per-layer `*_res_proj` row. Mix=0 is the identity skip (return
//! the partial / skip source). Mix=1 is the full softmax mixture.

/// Vanilla RMSNorm: `x * w / sqrt(mean(x^2) + eps)`.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len());
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w).map(|(v, s)| v * s * inv).collect()
}

/// Softmax mixture over residual sources. `sources[0]` is conventionally the
/// skip (current partial block). `query` is `[hidden]` (`*_res_proj`).
pub fn attnres_softmax_mix(
    sources: &[Vec<f32>],
    query: &[f32],
    norm_w: &[f32],
    eps: f32,
) -> Vec<f32> {
    assert!(
        !sources.is_empty(),
        "AttnRes needs at least the skip source"
    );
    let hidden = query.len();
    let mut logits = Vec::with_capacity(sources.len());
    for src in sources {
        assert_eq!(src.len(), hidden);
        let k = rms_norm(src, norm_w, eps);
        let dot = query.iter().zip(&k).map(|(q, kk)| q * kk).sum::<f32>();
        logits.push(dot);
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let z: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= z;
    }
    let mut out = vec![0.0f32; hidden];
    for (src, a) in sources.iter().zip(weights) {
        for (o, v) in out.iter_mut().zip(src) {
            *o += a * v;
        }
    }
    out
}

/// Test / ablation lever: `mix=0` returns `skip`; `mix=1` returns `mixed`.
pub fn attnres_blend(skip: &[f32], mixed: &[f32], mix: f32) -> Vec<f32> {
    assert_eq!(skip.len(), mixed.len());
    skip.iter()
        .zip(mixed)
        .map(|(s, m)| (1.0 - mix) * s + mix * m)
        .collect()
}

/// Apply AttnRes with an explicit mix lever. `mix=0` is identity skip.
pub fn attnres_mix(
    sources: &[Vec<f32>],
    query: &[f32],
    norm_w: &[f32],
    eps: f32,
    mix: f32,
) -> Vec<f32> {
    let skip = &sources[0];
    let mixed = attnres_softmax_mix(sources, query, norm_w, eps);
    attnres_blend(skip, &mixed, mix)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn mix_zero_is_identity_skip() {
        let skip = vec![1.0, 0.0, -0.5, 2.0];
        let other = vec![0.0, 1.0, 4.0, -3.0];
        let sources = [skip.clone(), other];
        let query = vec![0.2, -0.1, 0.4, 0.3];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let id = attnres_mix(&sources, &query, &w, 1e-5, 0.0);
        assert_eq!(id, skip, "mix=0 must return the skip source");
    }

    #[test]
    fn mix_zero_vs_mix_one_diverges() {
        // Known-bad: a graph that zeros mix weights and still claims mix=1.
        // The instrument must see mix=0 ≠ mix=1 before a green AttnRes is trusted.
        let skip = vec![1.0, 0.0, -0.5, 2.0];
        let other = vec![0.0, 1.0, 4.0, -3.0];
        let sources = [skip.clone(), other];
        let query = vec![0.2, -0.1, 0.4, 0.3];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let m0 = attnres_mix(&sources, &query, &w, 1e-5, 0.0);
        let m1 = attnres_mix(&sources, &query, &w, 1e-5, 1.0);
        assert!(
            max_abs(&m0, &m1) > 0.5,
            "mix=0 ({m0:?}) must diverge from mix=1 ({m1:?})"
        );
        assert_eq!(m0, skip);
        assert_ne!(m1, skip);
    }
}
