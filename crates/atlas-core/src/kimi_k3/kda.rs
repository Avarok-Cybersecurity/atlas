// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 KDA CPU reference — a new backend, not GDN / Mamba-2.
//!
//! Production geometry: `head_dim=128`, `short_conv_kernel_size=4`,
//! `use_full_rank_gate=true`, `gate_lower_bound=-5`. Decay stays low-rank
//! `f_a`/`f_b`; the **output** gate is full-rank `g_proj` (unlike GLM-5.3's
//! `g_a`/`g_b`).
//!
//! Recurrence (decode, prenorm q/k):
//! ```text
//! S <- S * diag(exp(g_t))     // decay on KEY axis, per channel
//! delta <- (v_t - S^T k_t) * beta_t
//! S <- S + k_t ⊗ delta
//! o_t <- S^T q_t / sqrt(d)
//! ```
//!
//! Conv state is Atlas-width: `[channels, kernel]` (one slot wider than HF's
//! `kernel-1`). Slot 0 is shifted out.

#![allow(clippy::needless_range_loop)]

use super::situ::sigmoid;

/// KDA geometry. Tiny dims are legal for CPU tests; production is 128/4.
#[derive(Clone, Copy, Debug)]
pub struct KdaConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    pub gate_lower_bound: f32,
    pub use_full_rank_gate: bool,
}

impl KdaConfig {
    /// Official K3 KDA (and the 0.40B twin except head_dim/heads).
    pub fn production() -> Self {
        Self {
            heads: 96,
            head_dim: 128,
            conv_kernel: 4,
            gate_lower_bound: -5.0,
            use_full_rank_gate: true,
        }
    }

    pub fn qkv_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn conv_dim(&self) -> usize {
        3 * self.qkv_dim()
    }

    pub fn recurrent_elems(&self) -> usize {
        self.heads * self.head_dim * self.head_dim
    }

    pub fn conv_elems(&self) -> usize {
        self.conv_dim() * self.conv_kernel
    }
}

/// Per-sequence KDA state. Both buffers are FP32, read-modify-write.
#[derive(Clone, Debug)]
pub struct KdaState {
    /// `[conv_dim, conv_kernel]` FP32.
    pub conv: Vec<f32>,
    /// `[heads, head_dim, head_dim]` FP32, K-major.
    pub recurrent: Vec<f32>,
}

impl KdaState {
    pub fn new(cfg: &KdaConfig) -> Self {
        Self {
            conv: vec![0.0; cfg.conv_elems()],
            recurrent: vec![0.0; cfg.recurrent_elems()],
        }
    }
}

/// Causal depthwise conv + SiLU. Shifts Atlas-width state left, writes `x`
/// into the last slot, then `y[c] = silu(dot(w[c], state[c]))`.
pub fn conv_update(
    state: &mut [f32],
    x: &[f32],
    w: &[f32],
    channels: usize,
    kernel: usize,
) -> Vec<f32> {
    assert_eq!(x.len(), channels);
    assert_eq!(state.len(), channels * kernel);
    assert_eq!(w.len(), channels * kernel);
    let mut y = vec![0.0f32; channels];
    for c in 0..channels {
        let row = c * kernel;
        for k in 0..kernel - 1 {
            state[row + k] = state[row + k + 1];
        }
        state[row + kernel - 1] = x[c];
        let mut acc = 0.0f32;
        for k in 0..kernel {
            acc += w[row + k] * state[row + k];
        }
        y[c] = acc * sigmoid(acc); // SiLU
    }
    y
}

/// Bounded K3 forget gate: `lower_bound * sigmoid(exp(a_log[h]) * (z + dt_bias))`.
pub fn bounded_gate(
    z: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    heads: usize,
    head_dim: usize,
    lower_bound: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; heads * head_dim];
    for h in 0..heads {
        let decay = a_log[h].exp();
        for d in 0..head_dim {
            let ch = h * head_dim + d;
            out[ch] = lower_bound * sigmoid(decay * (z[ch] + dt_bias[ch]));
        }
    }
    out
}

fn l2norm_rows(x: &[f32], d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let inv = 1.0 / (row_in.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
        for (o, i) in row_out.iter_mut().zip(row_in) {
            *o = i * inv;
        }
    }
    out
}

/// One-token KDA core. `qkv` is post-conv `[3 * qkv_dim]` (q|k|v).
/// Updates `state.recurrent` in place. q/k are L2-normalised here.
pub fn kda_recurrent_step(
    qkv: &[f32],
    gate: &[f32],
    beta: &[f32],
    cfg: &KdaConfig,
    recurrent: &mut [f32],
) -> Vec<f32> {
    let (h_n, d) = (cfg.heads, cfg.head_dim);
    let qkv_dim = h_n * d;
    let q = l2norm_rows(&qkv[..qkv_dim], d, 1e-6);
    let k = l2norm_rows(&qkv[qkv_dim..2 * qkv_dim], d, 1e-6);
    let v = &qkv[2 * qkv_dim..3 * qkv_dim];
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; qkv_dim];
    let mut delta = vec![0.0f32; d];
    for h in 0..h_n {
        let base = h * d;
        let s = &mut recurrent[h * d * d..(h + 1) * d * d];
        for kd in 0..d {
            let decay = gate[base + kd].exp();
            for vd in 0..d {
                s[kd * d + vd] *= decay;
            }
        }
        let b = beta[h];
        for vd in 0..d {
            let mut kv = 0.0f32;
            for kd in 0..d {
                kv += s[kd * d + vd] * k[base + kd];
            }
            delta[vd] = (v[base + vd] - kv) * b;
        }
        for kd in 0..d {
            let kk = k[base + kd];
            for vd in 0..d {
                s[kd * d + vd] += kk * delta[vd];
            }
        }
        for vd in 0..d {
            let mut acc = 0.0f32;
            for kd in 0..d {
                acc += s[kd * d + vd] * q[base + kd] * scale;
            }
            out[base + vd] = acc;
        }
    }
    out
}

/// Full-rank output gate: `sigmoid(g) ⊙ RMSNorm(core)` per head.
pub fn full_rank_output_gate(core: &[f32], g: &[f32], head_dim: usize, eps: f32) -> Vec<f32> {
    assert_eq!(core.len(), g.len());
    let mut out = vec![0.0f32; core.len()];
    for (row_c, (row_g, row_o)) in core
        .chunks_exact(head_dim)
        .zip(g.chunks_exact(head_dim).zip(out.chunks_exact_mut(head_dim)))
    {
        let mean_sq = row_c.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..head_dim {
            row_o[i] = sigmoid(row_g[i]) * row_c[i] * inv;
        }
    }
    out
}

/// One decode token: conv update then recurrent step.
pub fn kda_decode_token(
    x_qkv: &[f32],
    conv_w: &[f32],
    gate: &[f32],
    beta: &[f32],
    cfg: &KdaConfig,
    state: &mut KdaState,
) -> Vec<f32> {
    let conv_out = conv_update(
        &mut state.conv,
        x_qkv,
        conv_w,
        cfg.conv_dim(),
        cfg.conv_kernel,
    );
    kda_recurrent_step(&conv_out, gate, beta, cfg, &mut state.recurrent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> KdaConfig {
        KdaConfig {
            heads: 1,
            head_dim: 2,
            conv_kernel: 4,
            gate_lower_bound: -5.0,
            use_full_rank_gate: true,
        }
    }

    #[test]
    fn production_geometry() {
        let p = KdaConfig::production();
        assert_eq!(p.head_dim, 128);
        assert_eq!(p.conv_kernel, 4);
        assert!(p.use_full_rank_gate);
        assert_eq!(p.gate_lower_bound, -5.0);
    }

    #[test]
    fn conv_kernel_4_state_advances() {
        let cfg = tiny();
        let ch = cfg.conv_dim(); // 6
        let k = cfg.conv_kernel;
        let mut state = vec![0.0f32; ch * k];
        let w = vec![1.0f32; ch * k];
        for t in 0..4u32 {
            let x = vec![(t + 1) as f32; ch];
            let _y = conv_update(&mut state, &x, &w, ch, k);
            // Last slot is the current sample.
            for c in 0..ch {
                assert_eq!(state[c * k + (k - 1)], x[c], "t={t} last slot");
            }
        }
        // After 4 distinct tokens the window holds 1,2,3,4 (oldest → newest).
        for c in 0..ch {
            let row = &state[c * k..(c + 1) * k];
            assert_eq!(row, &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn prefix_hit_wrong_slot_diverges() {
        // C4 seed: restore the conv/recurrent state from the wrong slot
        // after a prefix hit, and the next decode must move.
        let cfg = tiny();
        let ch = cfg.conv_dim();
        let mut correct = KdaState::new(&cfg);
        let conv_w = vec![0.25f32; cfg.conv_elems()];
        let gate = bounded_gate(
            &[0.1, -0.2],
            &[0.0, 0.0],
            &[-1.0],
            cfg.heads,
            cfg.head_dim,
            cfg.gate_lower_bound,
        );
        let beta = [0.5f32];
        let mut snapshots = Vec::new();
        for t in 0..2u32 {
            let x = vec![(t + 1) as f32 * 0.1; ch];
            let _ = kda_decode_token(&x, &conv_w, &gate, &beta, &cfg, &mut correct);
            snapshots.push(correct.clone());
        }
        // Sequential token 3 from the real prefix (after two tokens).
        let x3 = vec![0.4f32; ch];
        let y_seq = kda_decode_token(&x3, &conv_w, &gate, &beta, &cfg, &mut correct);
        // Prefix hit: restore snapshot after token 2, decode the same x3.
        let mut from_prefix = snapshots[1].clone();
        let y_hit = kda_decode_token(&x3, &conv_w, &gate, &beta, &cfg, &mut from_prefix);
        assert_eq!(y_hit, y_seq, "correct slot must match sequential decode");
        assert_eq!(from_prefix.conv, correct.conv);
        // Wrong slot: restore snapshot after token 1 (prefix-hit then wrong state).
        let mut wrong = snapshots[0].clone();
        let y_wrong = kda_decode_token(&x3, &conv_w, &gate, &beta, &cfg, &mut wrong);
        let err: f32 = y_hit
            .iter()
            .zip(&y_wrong)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            err > 1e-4,
            "wrong-slot restore must diverge (max abs {err}), hit={y_hit:?} wrong={y_wrong:?}"
        );
    }
}
