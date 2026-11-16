// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the per-request inputs of a decision that change from call to call: predictions (success
//! probability and token usage per candidate), caller prices, model residency with the swap history, and queue
//! depths. Invalid numbers (NaN, negative) read as absent, so they can never steer a decision.

use std::time::Instant;

use crate::hysteresis::SwapHistory;
use crate::swap::SwapPlan;
use crate::types::BackendId;

/// Predicted token counts for one candidate (the usage part of a prediction).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct TokenUsage {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

impl TokenUsage {
    pub const fn new(input: f64, output: f64) -> Self {
        Self {
            input,
            output,
            cache_read: 0.0,
            cache_write: 0.0,
        }
    }

    /// Every count finite and non-negative.
    pub fn is_valid(&self) -> bool {
        [self.input, self.output, self.cache_read, self.cache_write]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    }
}

/// One candidate's prediction: success probability and token usage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub p_success: f64,
    pub usage: TokenUsage,
}

impl Prediction {
    pub const fn new(p_success: f64, usage: TokenUsage) -> Self {
        Self { p_success, usage }
    }

    /// `p` clamped to [0, 1]; `None` when `p` is not finite or a count is invalid (treated as absent).
    pub fn sanitized(&self) -> Option<Self> {
        if !self.p_success.is_finite() || !self.usage.is_valid() {
            return None;
        }
        Some(Self {
            p_success: self.p_success.clamp(0.0, 1.0),
            usage: self.usage,
        })
    }
}

/// Predictions per backend. A backend without an entry (or with an invalid one) has no prediction.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Predictions {
    entries: Vec<(BackendId, Prediction)>,
}

impl Predictions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder form of `insert`.
    pub fn with(mut self, id: BackendId, prediction: Prediction) -> Self {
        self.insert(id, prediction);
        self
    }

    /// Sets (or replaces) the prediction for `id`.
    pub fn insert(&mut self, id: BackendId, prediction: Prediction) {
        match self.entries.iter_mut().find(|(k, _)| *k == id) {
            Some(slot) => slot.1 = prediction,
            None => self.entries.push((id, prediction)),
        }
    }

    /// The sanitized prediction for `id`.
    pub fn get(&self, id: BackendId) -> Option<Prediction> {
        self.entries
            .iter()
            .find(|(k, _)| *k == id)
            .and_then(|(_, p)| p.sanitized())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Caller prices in US dollars per million tokens. Local backends default to zero (or a price derived from
/// measured GPU-seconds, `utility::local_price`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Price {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

impl Price {
    pub const ZERO: Price = Price {
        input: 0.0,
        output: 0.0,
        cache_read: 0.0,
        cache_write: 0.0,
    };

    pub const fn new(input: f64, output: f64, cache_read: f64, cache_write: f64) -> Self {
        Self {
            input,
            output,
            cache_read,
            cache_write,
        }
    }

    /// Every price finite and non-negative.
    pub fn is_valid(&self) -> bool {
        [self.input, self.output, self.cache_read, self.cache_write]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    }
}

/// Prices per backend.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Prices {
    entries: Vec<(BackendId, Price)>,
}

impl Prices {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, id: BackendId, price: Price) -> Self {
        self.insert(id, price);
        self
    }

    pub fn insert(&mut self, id: BackendId, price: Price) {
        match self.entries.iter_mut().find(|(k, _)| *k == id) {
            Some(slot) => slot.1 = price,
            None => self.entries.push((id, price)),
        }
    }

    /// The price for `id` when it is set and valid.
    pub fn get(&self, id: BackendId) -> Option<Price> {
        self.entries
            .iter()
            .find(|(k, _)| *k == id)
            .map(|(_, p)| *p)
            .filter(Price::is_valid)
    }
}

/// Residency of one local backend's model on its device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidencyEntry {
    pub id: BackendId,
    pub resident: bool,
    /// Seconds a swap to this model costs everyone (the server's `x_atlas.swap_seconds_estimate`).
    pub swap_seconds: Option<f64>,
}

/// Which local models are loaded, and the swap history. Backends without an entry (cloud ones) are always
/// treated as resident: serving them needs no swap. The model assumes one shared device: a swap makes its
/// target the only resident entry (`apply_swap`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Residency {
    pub entries: Vec<ResidencyEntry>,
    pub history: SwapHistory,
}

impl Residency {
    pub fn entry(&self, id: BackendId) -> Option<&ResidencyEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    pub fn is_resident(&self, id: BackendId) -> bool {
        self.entry(id).is_none_or(|e| e.resident)
    }

    /// The swap estimate for `id` when it is set, finite and non-negative.
    pub fn swap_seconds(&self, id: BackendId) -> Option<f64> {
        self.entry(id)
            .and_then(|e| e.swap_seconds)
            .filter(|s| s.is_finite() && *s >= 0.0)
    }

    /// The resident model a swap to `id` would evict (the lowest id among the other resident entries).
    pub fn evicted_by(&self, id: BackendId) -> Option<BackendId> {
        self.entries
            .iter()
            .filter(|e| e.resident && e.id != id)
            .map(|e| e.id)
            .min()
    }

    /// Caller-side bookkeeping after the front performed `plan`: its target becomes the only resident entry
    /// and the swap is recorded for hysteresis.
    pub fn apply_swap(&mut self, plan: &SwapPlan, now: Instant) {
        for e in &mut self.entries {
            e.resident = e.id == plan.to;
        }
        self.history.record(now);
    }
}

/// Queue depth of one backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueueEntry {
    pub id: BackendId,
    pub depth: u32,
}

/// Queue state. `hard_queued` counts the requests (the current one included) whose best target is a model that
/// is not resident; it scales the swap rule's gain.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Queues {
    pub entries: Vec<QueueEntry>,
    pub hard_queued: u32,
}

impl Queues {
    pub fn depth(&self, id: BackendId) -> u32 {
        self.entries
            .iter()
            .find(|e| e.id == id)
            .map_or(0, |e| e.depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_numbers_read_as_absent() {
        let ok = TokenUsage::new(1.0, 2.0);
        let preds = Predictions::new()
            .with(BackendId(1), Prediction::new(f64::NAN, ok))
            .with(BackendId(2), Prediction::new(1.5, ok))
            .with(
                BackendId(3),
                Prediction::new(0.5, TokenUsage::new(-1.0, 0.0)),
            )
            .with(BackendId(2), Prediction::new(-0.5, ok));
        assert_eq!(preds.len(), 3);
        assert_eq!(preds.get(BackendId(1)), None);
        assert_eq!(preds.get(BackendId(2)).map(|p| p.p_success), Some(0.0));
        assert_eq!(preds.get(BackendId(3)), None);
        let prices = Prices::new().with(BackendId(1), Price::new(1.0, f64::INFINITY, 0.0, 0.0));
        assert_eq!(prices.get(BackendId(1)), None);
    }

    #[test]
    fn residency_defaults_and_swap_bookkeeping() {
        let mut r = Residency {
            entries: vec![
                ResidencyEntry {
                    id: BackendId(1),
                    resident: true,
                    swap_seconds: Some(f64::NAN),
                },
                ResidencyEntry {
                    id: BackendId(2),
                    resident: false,
                    swap_seconds: Some(40.0),
                },
            ],
            history: SwapHistory::new(),
        };
        assert!(r.is_resident(BackendId(9)));
        assert_eq!(r.swap_seconds(BackendId(1)), None);
        assert_eq!(r.evicted_by(BackendId(2)), Some(BackendId(1)));
        let plan = SwapPlan {
            from: Some(BackendId(1)),
            to: BackendId(2),
            seconds: Some(40.0),
            forced: false,
        };
        r.apply_swap(&plan, Instant::now());
        assert!(!r.is_resident(BackendId(1)));
        assert!(r.is_resident(BackendId(2)));
        assert_eq!(r.history.len(), 1);
        assert_eq!(Queues::default().depth(BackendId(1)), 0);
    }
}
