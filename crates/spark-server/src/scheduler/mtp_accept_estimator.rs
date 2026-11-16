// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08), vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py, Copyright contributors to the vLLM project, Apache-2.0.
// Changes: Triton kernels replaced by scalar f64 host code; per-request GPU buffers replaced by per-position accumulators; the feature is read from the drafter's top-1 log-probability (`pending_draft_conf`) instead of recomputed from draft logits; positions are capped at 16. See DCUT-V2-ATTRIBUTION.md.

//! Responsibility: the online acceptance estimator behind D-Cut v2: a
//! logistic model `p = sigmoid(w * f + b_k)` of "the verifier accepts draft
//! position `k`", fitted while serving by damped Newton-IRLS.
//!
//! # Model
//!
//! * `f` is the log-odds of the drafter's top-1 probability `q`:
//!   `f = ln(q / (1 - q))`. Atlas stores `ln q` (D-Cut's ranking key,
//!   `ActiveSeq::pending_draft_conf`), so [`feature`] maps `ln q` to `f` and
//!   clamps it to `+-40` (a drafter that puts all its mass on one token sends
//!   `f` to infinity).
//! * One slope `w` is shared by every draft position; each position `k` has its
//!   own intercept `b_k`. Cold start: `w = 0.5`, `b_k = -0.2` (the median of
//!   the coefficients the upstream fits on its own drafters; the log-loss is
//!   flat near the optimum and a real fit replaces it within a refit or two).
//! * Survival of draft `j` is the prefix product of the calibrated `p`, so it
//!   is non-increasing in `j` whatever the coefficients are. An unfitted
//!   estimator therefore degrades to uniform-depth truncation, never to
//!   anything harmful.
//!
//! # Labels
//!
//! A verify pass that checked `v` drafts and accepted `a` of them grades
//! positions `0..=min(a, v - 1)`: positions below `a` are label 1, position
//! `a` (the first rejected draft, when `a < v`) is label 0, and positions past
//! the first rejection were never reached and carry no label.
//!
//! # Fit
//!
//! Statistics accumulate for [`REFIT_INTERVAL`] ticks (one tick per verify
//! step), then one Newton step solves `(X'WX + lambda I) d = X'(y - p)` on the
//! arrowhead system (shared slope, per-position intercepts) by Schur
//! complement. The step is damped by `n / (n + 50)` (the slope by the round's
//! total count, each intercept by its own position's count) so a data-poor
//! position keeps learning slowly instead of freezing; ridge `lambda = 1e-3`;
//! a step that would drive the slope negative or non-finite is dropped.
//!
//! The estimator never reads the drafter's draft TOKEN, only its distribution
//! at that position, so trimming a draft never conditions on the token being
//! trimmed and verification stays lossless.

/// Draft positions tracked individually (position `k >= MAX_POS` shares the
/// last slot). Atlas speculates up to 16 tokens (DFlash gamma blocks).
pub const MAX_POS: usize = 16;
/// Verify steps between refits.
pub const REFIT_INTERVAL: u32 = 100;
/// Newton steps are damped by `n / (n + DAMPING_OBSERVATIONS)`.
pub const DAMPING_OBSERVATIONS: f64 = 50.0;
/// Ridge added to the diagonal of the Newton system.
pub const L2: f64 = 1e-3;
/// The feature is clamped to `[-MAX_LOG_ODDS, MAX_LOG_ODDS]`.
pub const MAX_LOG_ODDS: f64 = 40.0;
/// Cold-start shared slope.
pub const INIT_SLOPE: f64 = 0.5;
/// Cold-start per-position intercept.
pub const INIT_BIAS: f64 = -0.2;

/// Log-odds `ln(q / (1 - q))` of a top-1 probability given as `ln q`.
///
/// `1 - q` is `-expm1(ln q)`, which keeps full relative precision however close
/// `q` is to 1 (the complement mass is the whole quantity of interest; forming
/// `1 - exp(ln q)` loses it below one f64 ulp of 1.0). A non-finite-NaN input
/// carries no information and maps to 0.
pub fn feature(logprob: f32) -> f64 {
    if logprob.is_nan() {
        return 0.0;
    }
    let lp = f64::from(logprob).min(0.0);
    let one_minus_q = -lp.exp_m1();
    let f = if one_minus_q <= 0.0 {
        MAX_LOG_ODDS
    } else {
        lp - one_minus_q.ln()
    };
    f.clamp(-MAX_LOG_ODDS, MAX_LOG_ODDS)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The logistic acceptance model and its IRLS accumulators.
#[derive(Debug, Clone)]
pub struct AcceptEstimator {
    /// Shared slope `w` (kept `>= 0`).
    pub slope: f64,
    /// Per-position intercepts `b_k`.
    pub bias: [f64; MAX_POS],
    /// Per position `[sum w f^2, sum w f, sum w]` (the `X'WX` arrowhead).
    info: [[f64; 3]; MAX_POS],
    /// Per position `[sum r f, sum r]` (the `X'(y - p)` vector).
    grad: [[f64; 2]; MAX_POS],
    /// Graded drafts per position this round (the damping count).
    counts: [f64; MAX_POS],
    steps_since_refit: u32,
    /// Refits performed so far.
    pub refits: u64,
    /// Graded drafts accumulated since the process started.
    pub observed: u64,
}

impl Default for AcceptEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl AcceptEstimator {
    pub fn new() -> Self {
        Self {
            slope: INIT_SLOPE,
            bias: [INIT_BIAS; MAX_POS],
            info: [[0.0; 3]; MAX_POS],
            grad: [[0.0; 2]; MAX_POS],
            counts: [0.0; MAX_POS],
            steps_since_refit: 0,
            refits: 0,
            observed: 0,
        }
    }

    /// `p(accept)` for feature `f` at draft position `pos` (0-based).
    pub fn predict(&self, f: f64, pos: usize) -> f64 {
        sigmoid(self.slope * f + self.bias[pos.min(MAX_POS - 1)])
    }

    /// Survival (prefix product of the calibrated `p`) of each of the first `k`
    /// drafts of one sequence. A position with no measured confidence counts as
    /// certain (`p = 1`), exactly as D-Cut's `select` treats a missing
    /// log-probability, so a drafter that cannot report confidence is never
    /// pruned on a number nobody produced.
    pub fn survival(&self, conf_logprobs: &[f32], k: usize) -> Vec<f64> {
        let mut acc = 1.0;
        (0..k)
            .map(|j| {
                if let Some(&lp) = conf_logprobs.get(j) {
                    acc *= self.predict(feature(lp), j);
                }
                acc
            })
            .collect()
    }

    /// Fold one graded draft into the accumulators (a Newton step's rows of
    /// `X'WX` and `X'(y - p)` at the current coefficients).
    pub fn observe(&mut self, pos: usize, f: f64, label: bool) {
        let k = pos.min(MAX_POS - 1);
        let p = self.predict(f, k);
        let w = p * (1.0 - p);
        let y = if label { 1.0 } else { 0.0 };
        let resid = y - p;
        let (info, grad) = (&mut self.info[k], &mut self.grad[k]);
        info[0] += w * f * f;
        info[1] += w * f;
        info[2] += w;
        grad[0] += resid * f;
        grad[1] += resid;
        self.counts[k] += 1.0;
        self.observed += 1;
    }

    /// Grade one verify pass: `conf_logprobs` holds the drafter's top-1
    /// log-probability of every draft the pass checked, and `accepted` how many
    /// of them the verifier accepted (see the module docs for the labels).
    pub fn observe_pass(&mut self, conf_logprobs: &[f32], accepted: usize) {
        let verified = conf_logprobs.len();
        if verified == 0 {
            return;
        }
        let last = accepted.min(verified - 1);
        for (j, &lp) in conf_logprobs.iter().enumerate().take(last + 1) {
            self.observe(j, feature(lp), j < accepted);
        }
    }

    /// Count one verify step; refit when the interval is reached. Returns true
    /// when a refit ran.
    pub fn tick(&mut self) -> bool {
        self.steps_since_refit += 1;
        if self.steps_since_refit < REFIT_INTERVAL {
            return false;
        }
        self.refit();
        true
    }

    /// One damped Newton-IRLS step on the accumulated statistics, then clear
    /// them. A refit on empty statistics leaves the coefficients unchanged.
    pub fn refit(&mut self) {
        self.steps_since_refit = 0;
        self.refits += 1;
        let a = self.info.iter().map(|r| r[0]).sum::<f64>() + L2;
        let g0 = self.grad.iter().map(|r| r[0]).sum::<f64>();
        let (mut shrink, mut coupling) = (0.0, 0.0);
        for k in 0..MAX_POS {
            let c = self.info[k][2] + L2;
            shrink += self.info[k][1] * self.info[k][1] / c;
            coupling += self.info[k][1] * self.grad[k][1] / c;
        }
        let mut step_w = (g0 - coupling) / (a - shrink);
        let total_n: f64 = self.counts.iter().sum();
        step_w *= total_n / (total_n + DAMPING_OBSERVATIONS);
        if !step_w.is_finite() || self.slope + step_w < 0.0 {
            step_w = 0.0;
        }
        for k in 0..MAX_POS {
            let c = self.info[k][2] + L2;
            let n = self.counts[k];
            let damp = n / (n + DAMPING_OBSERVATIONS);
            let step_b = (self.grad[k][1] - self.info[k][1] * step_w) / c * damp;
            if step_b.is_finite() {
                self.bias[k] += step_b;
            }
        }
        self.slope += step_w;
        self.info = [[0.0; 3]; MAX_POS];
        self.grad = [[0.0; 2]; MAX_POS];
        self.counts = [0.0; MAX_POS];
    }
}

#[cfg(test)]
#[path = "mtp_accept_estimator_tests.rs"]
mod tests;
