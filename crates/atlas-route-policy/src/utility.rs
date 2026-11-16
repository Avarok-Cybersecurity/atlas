// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: expected utility. A candidate's utility is `V * p - sum_k n_k * rho_k / 1e6` (predicted token
//! counts times the caller's prices in dollars per million tokens), minus a switching cost at a user turn when
//! the candidate is not the current model: `L_t * [max(rho_in, rho_write) - rho_read]+ / 1e6` for rebuilding the
//! cached prefix. Local backends are priced at zero unless the caller sets a price (for example from measured
//! GPU-seconds, `local_price`). Formulas after the ARC technical report (section 2.2 and Appendix A) and the
//! cascade-routing expected-utility formalism; no code copied.

use std::cmp::{Ordering, Reverse};

use crate::eligibility::Member;
use crate::events::Event;
use crate::types::{BackendId, Predictions, Price, Prices, Queues, TokenUsage};

/// Prices are per million tokens.
pub const PER_MILLION: f64 = 1e6;

/// What every candidate's score shares in one decision.
#[derive(Clone, Copy, Debug)]
pub struct UtilityCtx<'a> {
    pub value_usd: f64,
    pub event: Event,
    pub last: Option<BackendId>,
    pub prices: &'a Prices,
    pub queues: &'a Queues,
}

/// One scored candidate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scored {
    pub id: BackendId,
    pub local: bool,
    pub p: f64,
    /// Usage cost plus switching cost, in US dollars.
    pub cost: f64,
    pub utility: f64,
    pub depth: u32,
}

/// The cost of a predicted usage at `price`, in US dollars.
pub fn usage_cost(usage: &TokenUsage, price: &Price) -> f64 {
    (usage.input * price.input
        + usage.output * price.output
        + usage.cache_read * price.cache_read
        + usage.cache_write * price.cache_write)
        / PER_MILLION
}

/// The cost of rebuilding a cached prefix of `context_tokens` on a candidate priced at `price`.
pub fn switch_cost(context_tokens: f64, price: &Price) -> f64 {
    let per_token = (price.input.max(price.cache_write) - price.cache_read).max(0.0);
    context_tokens.max(0.0) * per_token / PER_MILLION
}

/// `V * p - cost`.
pub fn utility(value_usd: f64, p: f64, cost: f64) -> f64 {
    value_usd * p - cost
}

/// The switching cost applies at a user turn to every candidate other than the current model.
pub fn charges_switch(event: Event, last: Option<BackendId>, candidate: BackendId) -> bool {
    event == Event::UserMessage && last.is_some_and(|l| l != candidate)
}

/// A local backend's price from measured throughput and what its GPU-hour is worth: input at the prefill rate,
/// output at the decode rate, cache reads free, cache writes at the input rate. A non-positive rate or price
/// gives zero for that part.
pub fn local_price(
    usd_per_gpu_hour: f64,
    prefill_tokens_per_s: f64,
    decode_tokens_per_s: f64,
) -> Price {
    let per_million = |rate: f64| {
        if usd_per_gpu_hour > 0.0 && rate > 0.0 && usd_per_gpu_hour.is_finite() && rate.is_finite()
        {
            usd_per_gpu_hour / 3600.0 / rate * PER_MILLION
        } else {
            0.0
        }
    };
    let input = per_million(prefill_tokens_per_s);
    Price {
        input,
        output: per_million(decode_tokens_per_s),
        cache_read: 0.0,
        cache_write: input,
    }
}

/// Scores one member. `None` when it has no usable prediction, or it is not local and has no valid price (an
/// unpriced cloud backend is never assumed free).
pub fn score(m: &Member, preds: &Predictions, ctx: &UtilityCtx<'_>) -> Option<Scored> {
    let pred = preds.get(m.id)?;
    let price = match ctx.prices.get(m.id) {
        Some(p) => p,
        None if m.local => Price::ZERO,
        None => return None,
    };
    let mut cost = usage_cost(&pred.usage, &price);
    if charges_switch(ctx.event, ctx.last, m.id) {
        let context = m
            .prompt_tokens
            .map_or(pred.usage.input + pred.usage.cache_read, |t| t as f64);
        cost += switch_cost(context, &price);
    }
    let value = utility(ctx.value_usd, pred.p_success, cost);
    if !value.is_finite() {
        return None;
    }
    Some(Scored {
        id: m.id,
        local: m.local,
        p: pred.p_success,
        cost,
        utility: value,
        depth: ctx.queues.depth(m.id),
    })
}

/// The deterministic preference between two scored candidates: higher utility, then local over cloud, then the
/// shorter queue, then the lower id.
pub fn compare(a: &Scored, b: &Scored) -> Ordering {
    a.utility
        .total_cmp(&b.utility)
        .then_with(|| a.local.cmp(&b.local))
        .then_with(|| Reverse(a.depth).cmp(&Reverse(b.depth)))
        .then_with(|| Reverse(a.id).cmp(&Reverse(b.id)))
}

/// The best scored candidate among `members` (`None` when none can be scored).
pub fn best(members: &[&Member], preds: &Predictions, ctx: &UtilityCtx<'_>) -> Option<Scored> {
    members
        .iter()
        .filter_map(|m| score(m, preds, ctx))
        .max_by(compare)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::Fit;
    use crate::types::Prediction;

    fn member(id: u32, local: bool) -> Member {
        Member {
            id: BackendId(id),
            local,
            fit: Fit::Fits,
            prompt_tokens: Some(10_000),
        }
    }

    #[test]
    fn usage_and_switch_cost() {
        let price = Price::new(3.0, 15.0, 0.3, 3.75);
        let usage = TokenUsage::new(10_000.0, 1_000.0);
        assert!((usage_cost(&usage, &price) - 0.045).abs() < 1e-12);
        assert!((switch_cost(10_000.0, &price) - 0.0345).abs() < 1e-12);
        // A cache read dearer than the rebuild costs nothing (positive part).
        assert_eq!(switch_cost(10_000.0, &Price::new(1.0, 1.0, 2.0, 1.0)), 0.0);
        assert!(charges_switch(
            Event::UserMessage,
            Some(BackendId(1)),
            BackendId(2)
        ));
        assert!(!charges_switch(
            Event::UserMessage,
            Some(BackendId(2)),
            BackendId(2)
        ));
        assert!(!charges_switch(
            Event::TaskStart,
            Some(BackendId(1)),
            BackendId(2)
        ));
        assert!(!charges_switch(Event::UserMessage, None, BackendId(2)));
    }

    #[test]
    fn local_price_from_gpu_seconds() {
        let p = local_price(0.36, 1000.0, 10.0);
        assert!((p.input - 0.1).abs() < 1e-12);
        assert!((p.output - 10.0).abs() < 1e-9);
        assert_eq!(p.cache_read, 0.0);
        assert_eq!(local_price(0.0, 1000.0, 10.0), Price::ZERO);
        assert_eq!(local_price(1.0, 0.0, f64::NAN), Price::ZERO);
    }

    #[test]
    fn scoring_rules() {
        let preds = Predictions::new()
            .with(
                BackendId(1),
                Prediction::new(0.9, TokenUsage::new(10_000.0, 1_000.0)),
            )
            .with(
                BackendId(3),
                Prediction::new(0.97, TokenUsage::new(10_000.0, 1_000.0)),
            )
            .with(
                BackendId(4),
                Prediction::new(0.5, TokenUsage::new(10_000.0, 1_000.0)),
            );
        let prices = Prices::new().with(BackendId(3), Price::new(3.0, 15.0, 0.3, 3.75));
        let queues = Queues::default();
        let ctx = UtilityCtx {
            value_usd: 4.5,
            event: Event::TaskStart,
            last: None,
            prices: &prices,
            queues: &queues,
        };
        let local = score(&member(1, true), &preds, &ctx);
        assert_eq!(local.map(|s| s.cost), Some(0.0));
        let cloud = score(&member(3, false), &preds, &ctx)
            .map(|s| s.utility)
            .unwrap_or(f64::NAN);
        assert!((cloud - (4.5 * 0.97 - 0.045)).abs() < 1e-12);
        // An unpriced cloud backend and a member without a prediction are not scored.
        assert_eq!(score(&member(4, false), &preds, &ctx), None);
        assert_eq!(score(&member(2, true), &preds, &ctx), None);
        let m = [member(1, true), member(3, false), member(4, false)];
        let refs: Vec<&Member> = m.iter().collect();
        assert_eq!(best(&refs, &preds, &ctx).map(|s| s.id), Some(BackendId(3)));
    }

    #[test]
    fn ties_prefer_local_then_short_queue_then_low_id() {
        let s = |id: u32, local: bool, depth: u32| Scored {
            id: BackendId(id),
            local,
            p: 0.5,
            cost: 0.0,
            utility: 1.0,
            depth,
        };
        assert_eq!(compare(&s(1, true, 0), &s(2, false, 0)), Ordering::Greater);
        assert_eq!(compare(&s(1, true, 3), &s(2, true, 1)), Ordering::Less);
        assert_eq!(compare(&s(1, true, 1), &s(2, true, 1)), Ordering::Greater);
    }
}
