// SPDX-License-Identifier: AGPL-3.0-only
//
// Random Gaussian projection matrix `P` for the Atlas high-speed-swap
// predictor. Generated once at predictor init from a fixed seed
// (Johnson–Lindenstrauss embedding); never re-derived per call. Stored on the
// host so we can hand it to the GPU as BF16 without an extra dtype dance.

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PredictorShape {
    pub head_dim: usize,
    pub r: usize,
}

impl PredictorShape {
    pub fn new(head_dim: usize, r: usize) -> Self {
        assert!(head_dim > 0 && r > 0, "head_dim/r must be positive");
        assert!(head_dim <= 256, "MAX_HEAD_DIM=256 in kv_lowrank_project.cu");
        assert!(r <= 128, "predictor_score block dim caps r at 128");
        Self { head_dim, r }
    }
}

/// Random Gaussian projection matrix `P` of shape `[head_dim, r]`, BF16, in
/// row-major layout. Variance 1/head_dim so that ⟨k, p⟩ has unit variance
/// for unit-norm `k`. Standard JL setting (KVSwap §2.2).
pub fn build_projection(shape: PredictorShape, seed: u64) -> Vec<bf16> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let inv_sqrt_d = 1.0_f32 / (shape.head_dim as f32).sqrt();
    let dist = StandardNormal;
    let n = shape.head_dim * shape.r;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let v: f32 = dist.sample(&mut rng);
        out.push(bf16::from_f32(v * inv_sqrt_d));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism() {
        let s = PredictorShape::new(128, 32);
        let a = build_projection(s, 0xCAFE_F00D);
        let b = build_projection(s, 0xCAFE_F00D);
        assert_eq!(a, b);
        let c = build_projection(s, 0xDEAD_BEEF);
        assert_ne!(a, c);
    }

    #[test]
    fn shape() {
        let s = PredictorShape::new(128, 32);
        let p = build_projection(s, 1);
        assert_eq!(p.len(), 128 * 32);
    }
}
