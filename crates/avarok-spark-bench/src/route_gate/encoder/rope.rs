// SPDX-License-Identifier: AGPL-3.0-only
// Portions follow the ARC-1.0 release documentation (Qwen3.5 text config: RoPE theta 1e7, YaRN factor 4, partial
// rotary factor 0.25, interleaved M-RoPE sections [11, 11, 10]); the YaRN formulas are the published ones. By Atlas
// contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: rotary position embedding of the encoder: the YaRN-corrected inverse frequencies, the interleaved
//! M-RoPE angle selection (equal position streams for text), and the cos/sin tables.

use std::f64::consts::PI;

/// Which of the undocumented RoPE choices to use (the goldens oracle searches them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeVariant {
    /// YaRN with the `factor` of the config (4.0) and its attention scaling `0.1 ln(factor) + 1`.
    Yarn,
    /// YaRN with the factor taken as `max_position_embeddings / original_max_position_embeddings` (what recent
    /// `transformers` does when both are present); with the ARC config that is 1, so this is plain RoPE.
    YarnRatio,
    /// No YaRN at all.
    Plain,
}

impl RopeVariant {
    pub fn name(self) -> &'static str {
        match self {
            Self::Yarn => "yarn",
            Self::YarnRatio => "yarn_ratio",
            Self::Plain => "plain",
        }
    }
}

/// RoPE settings of the text config.
#[derive(Clone, Debug)]
pub struct RopeConfig {
    pub theta: f64,
    pub partial_rotary_factor: f64,
    pub factor: f64,
    pub original_max_position: f64,
    pub max_position: f64,
    pub mrope_section: [usize; 3],
}

/// `base^(-2i/dim)`, the plain inverse frequencies of a rotary block of width `dim`.
pub fn plain_inv_freq(theta: f64, dim: usize) -> Vec<f64> {
    (0..dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f64 / dim as f64))
        .collect()
}

/// The YaRN attention scaling `0.1 ln(factor) + 1` (1 when there is no scaling).
pub fn yarn_mscale(factor: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        0.1 * factor.ln() + 1.0
    }
}

fn correction_dim(num_rot: f64, dim: usize, base: f64, max_pos: f64) -> f64 {
    dim as f64 * (max_pos / (num_rot * 2.0 * PI)).ln() / (2.0 * base.ln())
}

/// YaRN inverse frequencies: high-frequency dimensions keep the plain frequency, low-frequency ones are divided by
/// `factor`, with a linear ramp between the two correction dimensions (`beta_fast`, `beta_slow`).
pub fn yarn_inv_freq(
    theta: f64,
    dim: usize,
    factor: f64,
    original_max: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f64> {
    let half = dim / 2;
    let pos_freqs: Vec<f64> = (0..half)
        .map(|i| theta.powf(2.0 * i as f64 / dim as f64))
        .collect();
    let low = correction_dim(beta_fast, dim, theta, original_max)
        .floor()
        .max(0.0);
    let high = correction_dim(beta_slow, dim, theta, original_max)
        .ceil()
        .min(dim as f64 - 1.0);
    let high = if low == high { high + 0.001 } else { high };
    (0..half)
        .map(|i| {
            let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
            let extrapolation_factor = 1.0 - ramp;
            let interpolation = 1.0 / (factor * pos_freqs[i]);
            let extrapolation = 1.0 / pos_freqs[i];
            interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor
        })
        .collect()
}

/// Inverse frequencies and attention scaling for a variant.
pub fn inv_freq_for(cfg: &RopeConfig, rotary_dim: usize, variant: RopeVariant) -> (Vec<f64>, f64) {
    match variant {
        RopeVariant::Plain => (plain_inv_freq(cfg.theta, rotary_dim), 1.0),
        RopeVariant::Yarn => (
            yarn_inv_freq(
                cfg.theta,
                rotary_dim,
                cfg.factor,
                cfg.original_max_position,
                32.0,
                1.0,
            ),
            yarn_mscale(cfg.factor),
        ),
        RopeVariant::YarnRatio => {
            let factor = cfg.max_position / cfg.original_max_position;
            (
                yarn_inv_freq(
                    cfg.theta,
                    rotary_dim,
                    factor,
                    cfg.original_max_position,
                    32.0,
                    1.0,
                ),
                yarn_mscale(factor),
            )
        }
    }
}

/// Interleaved M-RoPE: angles `[t][dim/2]` from three position streams (temporal, height, width). The temporal angle is
/// used everywhere except that the height and width angles overwrite indexes `1, 4, 7, ...` and `2, 5, 8, ...` up to
/// `3 * section`.
pub fn mrope_angles(
    positions: &[Vec<f64>; 3],
    inv_freq: &[f64],
    section: [usize; 3],
) -> Vec<Vec<f64>> {
    let t = positions[0].len();
    (0..t)
        .map(|i| {
            let mut angles: Vec<f64> = inv_freq.iter().map(|f| positions[0][i] * f).collect();
            for (stream, offset) in [(1usize, 1usize), (2, 2)] {
                let length = (section[stream] * 3).min(inv_freq.len());
                let mut idx = offset;
                while idx < length {
                    angles[idx] = positions[stream][i] * inv_freq[idx];
                    idx += 3;
                }
            }
            angles
        })
        .collect()
}

/// `cos` and `sin` tables, `[t][rotary_dim]` (the half-width angles repeated twice, scaled by `mscale`).
pub struct RopeTables {
    pub rotary_dim: usize,
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}

impl RopeTables {
    /// Tables for text: the three position streams are all `0..t`.
    pub fn for_text(cfg: &RopeConfig, head_dim: usize, t: usize, variant: RopeVariant) -> Self {
        let rotary_dim = ((head_dim as f64 * cfg.partial_rotary_factor) as usize) & !1;
        let (inv, mscale) = inv_freq_for(cfg, rotary_dim, variant);
        let pos: Vec<f64> = (0..t).map(|i| i as f64).collect();
        let angles = mrope_angles(&[pos.clone(), pos.clone(), pos], &inv, cfg.mrope_section);
        Self::from_angles(&angles, rotary_dim, mscale)
    }

    /// Tables from half-width angles.
    pub fn from_angles(angles: &[Vec<f64>], rotary_dim: usize, mscale: f64) -> Self {
        let half = rotary_dim / 2;
        let mut cos = Vec::with_capacity(angles.len() * rotary_dim);
        let mut sin = Vec::with_capacity(angles.len() * rotary_dim);
        for row in angles {
            for _ in 0..2 {
                cos.extend(row.iter().take(half).map(|a| (a.cos() * mscale) as f32));
                sin.extend(row.iter().take(half).map(|a| (a.sin() * mscale) as f32));
            }
        }
        Self {
            rotary_dim,
            cos,
            sin,
        }
    }

    /// Rotates the first `rotary_dim` entries of one head vector at token `t` (`rotate_half` pairing `i` with
    /// `i + rotary_dim / 2`); the rest passes through.
    pub fn apply(&self, head: &mut [f32], t: usize) {
        let r = self.rotary_dim;
        let half = r / 2;
        let (c, s) = (&self.cos[t * r..(t + 1) * r], &self.sin[t * r..(t + 1) * r]);
        for i in 0..half {
            let (x1, x2) = (head[i], head[i + half]);
            head[i] = x1 * c[i] - x2 * s[i];
            head[i + half] = x2 * c[i + half] + x1 * s[i + half];
        }
    }
}
