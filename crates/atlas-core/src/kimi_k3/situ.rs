// SPDX-License-Identifier: AGPL-3.0-only

//! SiTU-GLU CPU reference.
//!
//! Official law (K3 tech report Eq. 12), β₁ = `activation_situ_beta` (4),
//! β₂ = `activation_situ_linear_beta` (25):
//!
//! ```text
//! softcap(x, β) = β * tanh(x / β)
//! SiTU-GLU(g, u) = (softcap(g, β1) * sigmoid(g)) ⊙ softcap(u, β2)
//! ```
//!
//! Not SwiGLU. A β=0 mutant is the known-bad: it is unbounded SwiGLU and
//! must diverge.

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU: `x * sigmoid(x)`. Used by KDA short-conv. Decay `f_a` is a plain linear.
#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Smooth cap. `beta` must be finite and non-zero.
#[inline]
pub fn softcap(x: f32, beta: f32) -> f32 {
    beta * (x / beta).tanh()
}

/// Gate branch: `softcap(x, beta) * sigmoid(x)`. Bound `|·| < |beta|`.
#[inline]
pub fn situ_gate(x: f32, beta: f32) -> f32 {
    softcap(x, beta) * sigmoid(x)
}

/// Up branch: `softcap(x, beta_lin)`. Bound `|·| < |beta_lin|`.
#[inline]
pub fn situ_up(x: f32, beta_lin: f32) -> f32 {
    softcap(x, beta_lin)
}

/// One SiTU-GLU coordinate. Product is bounded by `|β1 * β2|`.
#[inline]
pub fn situ_glu(gate: f32, up: f32, beta: f32, beta_lin: f32) -> f32 {
    situ_gate(gate, beta) * situ_up(up, beta_lin)
}

/// Elementwise SiTU-GLU over paired gate/up vectors.
pub fn situ_glu_vec(gate: &[f32], up: &[f32], beta: f32, beta_lin: f32) -> Vec<f32> {
    assert_eq!(gate.len(), up.len(), "SiTU-GLU gate/up length mismatch");
    gate.iter()
        .zip(up)
        .map(|(g, u)| situ_glu(*g, *u, beta, beta_lin))
        .collect()
}

/// SwiGLU mutant used as the known-bad: `silu(g) * u` (no tanh cap).
#[inline]
pub fn swiglu_mutant(gate: f32, up: f32) -> f32 {
    gate * sigmoid(gate) * up
}

#[cfg(test)]
mod tests {
    use super::*;

    const B1: f32 = 4.0;
    const B2: f32 = 25.0;
    const TOL: f32 = 1e-6;

    #[test]
    fn situ_glu_matches_closed_form_vector() {
        // Hand-evaluated at (g, u) = (4, 25): both branches sit on tanh(1).
        let tanh1 = 1.0f32.tanh();
        let want_gate = B1 * tanh1 * sigmoid(4.0);
        let want_up = B2 * tanh1;
        let want = want_gate * want_up;
        let got = situ_glu(4.0, 25.0, B1, B2);
        assert!((got - want).abs() < TOL, "closed form {want} vs impl {got}");

        let g = [0.0, 4.0, -2.0, 1.5];
        let u = [0.0, 25.0, 3.0, -8.0];
        let out = situ_glu_vec(&g, &u, B1, B2);
        for i in 0..g.len() {
            let closed = (B1 * (g[i] / B1).tanh() * sigmoid(g[i])) * (B2 * (u[i] / B2).tanh());
            assert!(
                (out[i] - closed).abs() < TOL,
                "idx {i}: {} vs {closed}",
                out[i]
            );
        }
        // Zero input is exactly zero (gate branch vanishes).
        assert_eq!(out[0], 0.0);
        // Bound: |z| < |β1 β2| = 100.
        assert!(out.iter().all(|z| z.abs() < 100.0));
    }

    #[test]
    fn situ_beta_zero_mutant_diverges_from_situ() {
        // Known-bad: treating β=0 as "no cap" / SwiGLU. Instrument must fail
        // this comparison before a green SiTU result is trusted.
        let g = 4.0f32;
        let u = 25.0f32;
        let situ = situ_glu(g, u, B1, B2);
        let mutant = swiglu_mutant(g, u);
        assert!(
            (situ - mutant).abs() > 1.0,
            "SiTU {situ} must diverge from SwiGLU mutant {mutant}"
        );
        // Production betas are never zero; a β=0 call is the planted defect.
        assert_ne!(B1, 0.0);
        assert_ne!(B2, 0.0);
    }
}
