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

/// Stream map keyed by hidden/residual pointer bits (`DevicePtr.0`).
/// Layer 0 inserts; last layer `remove`s on success. Any `Err` drops the
/// entry (umbrella `940bd4eeb`).
#[derive(Debug)]
pub struct AttnResHub<T> {
    map: std::collections::HashMap<u64, T>,
}

impl<T> Default for AttnResHub<T> {
    fn default() -> Self {
        Self {
            map: std::collections::HashMap::new(),
        }
    }
}

impl<T> AttnResHub<T> {
    pub fn insert(&mut self, key: u64, v: T) {
        self.map.insert(key, v);
    }

    pub fn get(&self, key: u64) -> Option<&T> {
        self.map.get(&key)
    }

    pub fn remove(&mut self, key: u64) -> Option<T> {
        self.map.remove(&key)
    }

    pub fn contains(&self, key: u64) -> bool {
        self.map.contains_key(&key)
    }

    pub fn decode<R, E>(
        &mut self,
        key: u64,
        f: impl FnOnce(&mut Self) -> Result<R, E>,
    ) -> Result<R, E> {
        let r = f(self);
        if r.is_err() {
            self.map.remove(&key);
        }
        r
    }
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
        const RECORDED_MIX1: [f32; 4] = [0.571_599_9, 0.428_400_1, 1.427_800_4, -0.142_000_4];
        assert!(
            max_abs(&m1, &RECORDED_MIX1) <= 1e-5,
            "mix=1 vs recorded fixture max_abs={}",
            max_abs(&m1, &RECORDED_MIX1)
        );
    }

    #[test]
    fn hub_drops_entry_on_decode_err() {
        let mut hub = AttnResHub::default();
        hub.insert(1, vec![1.0f32]);
        let err: Result<(), &str> = hub.decode(1, |_| Err("cuda fail"));
        assert!(err.is_err());
        assert!(
            !hub.contains(1),
            "RST: decode Err must drop the stream (940bd4eeb)"
        );
        hub.insert(2, vec![2.0]);
        hub.decode(2, |h| {
            h.remove(2);
            Ok::<(), &str>(())
        })
        .unwrap();
        assert!(!hub.contains(2), "last-layer success still removes");
    }
}
