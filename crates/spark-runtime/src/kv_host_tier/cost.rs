// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/multiprocess/kv_load_policy.py (commit 3aefae1, Apache-2.0; decline a load when recompute is cheaper); no code copied.
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: bench/README.md:103-175 ("Context-cost calibration": accept a cost fit only when its held-out p95 relative error passes a bound and materially different points are never predicted in reverse order). Changes: the offline acceptance rule applied online to live EWMA rate models, prequential (predict before update) observations as the held-out set, bounds as tier options, by Atlas contributors.

//! Responsibility: the restore cost gate. Two live rate models (host<->device copy bytes per
//! second, measured from the tier's own transfers; prefill tokens per second, fed by the glue)
//! estimate "restore these blocks" against "recompute these tokens". A model may gate only after
//! it proved itself on held-out data: every observation is predicted BEFORE it updates the model,
//! and over the last N such predictions the p95 relative error must be within the bound and no
//! pair of materially different observations may be predicted in reverse order (port plan
//! review C6). Until both models pass, the gate always restores: a restore replays exact bytes,
//! so restoring without proof is never wrong, only possibly slower.

use std::collections::VecDeque;

use super::config::TierOptions;

/// What the gate decided for one candidate restore.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateDecision {
    /// The validated model estimates the copy under `ratio` x the recompute.
    Restore { copy_secs: f64, prefill_secs: f64 },
    /// The model has not passed its check yet; restore.
    RestoreUnvalidated,
    /// Fewer tokens than `min_tokens`: recompute.
    DeclineShort,
    /// The validated model estimates recompute to be cheaper: recompute.
    DeclineCost { copy_secs: f64, prefill_secs: f64 },
}

impl GateDecision {
    pub fn restores(&self) -> bool {
        matches!(
            self,
            GateDecision::Restore { .. } | GateDecision::RestoreUnvalidated
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct Obs {
    actual: f64,
    predicted: f64,
}

/// `secs = work / rate`, with `rate` an EWMA of observed `work / secs`.
#[derive(Clone, Debug)]
pub(crate) struct RateModel {
    rate: Option<f64>,
    alpha: f64,
    window: VecDeque<Obs>,
    cap: usize,
    min_obs: usize,
    p95_limit: f64,
    material: f64,
    valid: bool,
}

impl RateModel {
    pub(crate) fn new(opts: &TierOptions) -> Self {
        Self {
            rate: None,
            alpha: opts.cost_ewma_alpha.clamp(1e-3, 1.0),
            window: VecDeque::with_capacity(opts.cost_window),
            cap: opts.cost_window,
            min_obs: opts.cost_min_observations.min(opts.cost_window),
            p95_limit: opts.cost_p95_limit,
            material: opts.cost_material_ratio.max(1.0),
            valid: false,
        }
    }

    pub(crate) fn predict(&self, work: f64) -> Option<f64> {
        self.rate.filter(|r| *r > 0.0).map(|r| work / r)
    }

    pub(crate) fn observe(&mut self, work: f64, secs: f64) {
        if !(work.is_finite() && secs.is_finite() && work > 0.0 && secs > 0.0) {
            return;
        }
        if let Some(p) = self.predict(work) {
            if self.window.len() == self.cap {
                self.window.pop_front();
            }
            self.window.push_back(Obs {
                actual: secs,
                predicted: p,
            });
        }
        let r = work / secs;
        self.rate = Some(match self.rate {
            Some(old) => old + self.alpha * (r - old),
            None => r,
        });
        self.valid = self.check();
    }

    pub(crate) fn valid(&self) -> bool {
        self.valid
    }

    /// The held-out p95 relative error over the window (None below the minimum count).
    pub(crate) fn p95_error(&self) -> Option<f64> {
        if self.window.len() < self.min_obs {
            return None;
        }
        let mut errs: Vec<f64> = self
            .window
            .iter()
            .map(|o| (o.predicted - o.actual).abs() / o.actual)
            .collect();
        errs.sort_by(f64::total_cmp);
        let idx = ((errs.len() as f64 * 0.95).ceil() as usize).clamp(1, errs.len()) - 1;
        Some(errs[idx])
    }

    /// A pair whose actual costs differ by at least the material ratio but whose predictions
    /// are strictly in the other order. Exact predicted ties are allowed.
    pub(crate) fn has_inversion(&self) -> bool {
        let w: Vec<Obs> = self.window.iter().copied().collect();
        for i in 0..w.len() {
            for j in (i + 1)..w.len() {
                let (a, b) = (w[i], w[j]);
                if a.actual >= self.material * b.actual && a.predicted < b.predicted {
                    return true;
                }
                if b.actual >= self.material * a.actual && b.predicted < a.predicted {
                    return true;
                }
            }
        }
        false
    }

    fn check(&self) -> bool {
        match self.p95_error() {
            Some(e) => e <= self.p95_limit && !self.has_inversion(),
            None => false,
        }
    }
}

/// The two models of the gate.
#[derive(Clone, Debug)]
pub(crate) struct CostModel {
    pub(crate) copy: RateModel,
    pub(crate) prefill: RateModel,
}

impl CostModel {
    pub(crate) fn new(opts: &TierOptions) -> Self {
        Self {
            copy: RateModel::new(opts),
            prefill: RateModel::new(opts),
        }
    }

    pub(crate) fn observe_copy(&mut self, bytes: f64, secs: f64) {
        self.copy.observe(bytes, secs);
    }

    pub(crate) fn observe_prefill(&mut self, tokens: f64, secs: f64) {
        self.prefill.observe(tokens, secs);
    }

    pub(crate) fn valid(&self) -> bool {
        self.copy.valid() && self.prefill.valid()
    }

    pub(crate) fn decide(
        &self,
        tokens: usize,
        bytes: u64,
        ratio: f32,
        min_tokens: u32,
    ) -> GateDecision {
        if tokens < min_tokens as usize || tokens == 0 {
            return GateDecision::DeclineShort;
        }
        if !self.valid() {
            return GateDecision::RestoreUnvalidated;
        }
        let (Some(copy_secs), Some(prefill_secs)) = (
            self.copy.predict(bytes as f64),
            self.prefill.predict(tokens as f64),
        ) else {
            return GateDecision::RestoreUnvalidated;
        };
        if copy_secs < f64::from(ratio) * prefill_secs {
            GateDecision::Restore {
                copy_secs,
                prefill_secs,
            }
        } else {
            GateDecision::DeclineCost {
                copy_secs,
                prefill_secs,
            }
        }
    }
}
