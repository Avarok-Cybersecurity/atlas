// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: crate root of `atlas-route-policy`, the pure routing policy of an Atlas routing front.
//!
//! The front (lane X05) calls, per request:
//!
//! 1. [`eligible`]: the eligible set E from the static policy, the request class, the locality header and each
//!    backend's health and capacity, before any prediction exists;
//! 2. (optionally) [`tighten_for_pii`] once a PII signal was computed over E; it can only remove members;
//! 3. [`infer_event`] on the previous and current request of the conversation branch;
//! 4. [`choose`] with the predictions (computed over E only), prices, residency, queues, the last target and the
//!    event.
//!
//! The rules:
//! - the target is a member of E by construction; an empty E is [`Decision::Unavailable`] (the front answers 503),
//!   never a fallback outside E;
//! - local-only (policy flag, privileged class, `x-atlas-locality: local`) gives E = local backends only;
//! - prefer-local takes the resident local model when it fits and `p >= min_p_pass`, else utility over E;
//! - utility is `V * p - sum_k n_k * rho_k / 1e6`, minus the switching cost at a user turn for a candidate other
//!   than the current model;
//! - a target that is not resident needs a swap, allowed only when the policy allows swaps (off by default), the
//!   dwell time and hourly cap allow one, and the gain pays for the dead time; otherwise the request is served on
//!   a resident member of E;
//! - absent predictions fall back to the resident model; the model changes only at a task start, a subagent
//!   start, a compaction or a user turn.
//!
//! Everything is deterministic: no I/O, no clock reads (`now` is an argument), no randomness, no network code.

#![forbid(unsafe_code)]

pub mod capacity;
pub mod decision;
pub mod eligibility;
pub mod events;
pub mod hysteresis;
pub mod inputs;
pub mod prefer_local;
pub mod swap;
pub mod timing;
pub mod types;
pub mod utility;

use std::time::Instant;

pub use capacity::{CapacityFact, Fit, Overflow, Provenance, fits};
pub use decision::{Decision, Reason, Unavailable};
pub use eligibility::{
    DecisionParams, EligibleSet, Exclusion, Member, PiiSignal, eligible, tighten_for_pii,
};
pub use events::{
    Basis, Event, MAIN_BRANCH, MessageDigest, RequestDigest, Role, infer_event, infer_with_basis,
};
pub use hysteresis::SwapHistory;
pub use inputs::{
    Prediction, Predictions, Price, Prices, QueueEntry, Queues, Residency, ResidencyEntry,
    TokenUsage,
};
pub use swap::{SwapDenial, SwapPlan, SwapPolicy};
pub use types::{
    Backend, BackendId, Locality, PiiRule, PolicyError, Preference, RequestClass, StaticPolicy,
    validate_backends,
};

/// A picked candidate before the residency check: its id and, when it was scored, its `p` and cost.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Choice {
    id: BackendId,
    p: Option<f64>,
    cost: Option<f64>,
}

impl Choice {
    fn scored(s: &utility::Scored) -> Self {
        Self {
            id: s.id,
            p: Some(s.p),
            cost: Some(s.cost),
        }
    }

    fn bare(id: BackendId) -> Self {
        Self {
            id,
            p: None,
            cost: None,
        }
    }

    fn route(self, reason: Reason, swap: Option<SwapPlan>) -> Decision {
        Decision::Route {
            target: self.id,
            reason,
            p: self.p,
            cost: self.cost,
            swap,
        }
    }
}

/// The best candidate among `members`: scored by utility, else the resident fallback.
fn best_or_fallback(
    members: &[&Member],
    preds: &Predictions,
    ctx: &utility::UtilityCtx<'_>,
    residency: &Residency,
    last: Option<BackendId>,
    queues: &Queues,
) -> Option<(Choice, Reason)> {
    if let Some(s) = utility::best(members, preds, ctx) {
        return Some((Choice::scored(&s), Reason::Utility));
    }
    prefer_local::resident_fallback(members, residency, last, queues)
        .map(|id| (Choice::bare(id), Reason::NoPrediction))
}

/// Picks the target for one request from E. `pred` is `None` when no predictor ran; a member without a usable
/// prediction is never scored. `last` is the previous target of this conversation branch, `event` the inferred
/// execution event, `now` the caller's clock (used only by hysteresis).
pub fn choose(
    e: &EligibleSet,
    pred: Option<&Predictions>,
    prices: &Prices,
    residency: &Residency,
    queues: &Queues,
    last: Option<BackendId>,
    event: Event,
    now: Instant,
) -> Decision {
    if e.is_empty() {
        return Decision::Unavailable {
            reason: e.empty_reason(),
        };
    }
    let no_predictions = Predictions::new();
    let preds = pred.unwrap_or(&no_predictions);
    let params = *e.params();
    let ctx = utility::UtilityCtx {
        value_usd: params.value_usd,
        event,
        last,
        prices,
        queues,
    };
    let members: Vec<&Member> = e.members().iter().collect();

    // Timing: between switching boundaries the previous target is kept while it is eligible and resident.
    if let Some(id) = timing::hold(e, last, event, residency) {
        let held = e
            .member(id)
            .and_then(|m| utility::score(m, preds, &ctx))
            .map_or(Choice::bare(id), |s| Choice::scored(&s));
        return held.route(Reason::Hold, None);
    }

    let shortcut = match params.preference {
        Preference::PreferLocal => {
            prefer_local::shortcut(&members, preds, residency, params.min_p_pass, &ctx)
                .map(|s| (Choice::scored(&s), Reason::PreferLocal))
        }
        Preference::Utility => None,
    };
    let Some((target, reason)) =
        shortcut.or_else(|| best_or_fallback(&members, preds, &ctx, residency, last, queues))
    else {
        return Decision::Unavailable {
            reason: e.empty_reason(),
        };
    };
    if residency.is_resident(target.id) {
        return target.route(reason, None);
    }

    // The target needs a swap. The alternative is the best resident member of E; without one the swap is forced.
    let resident: Vec<&Member> = members
        .iter()
        .copied()
        .filter(|m| residency.is_resident(m.id))
        .collect();
    let alternative =
        best_or_fallback(&resident, preds, &ctx, residency, last, queues).map(|(c, _)| c);
    let gain = match (target.p, alternative.and_then(|a| a.p)) {
        (Some(t), Some(a)) => Some(t - a),
        _ => None,
    };
    let forced = alternative.is_none();
    match swap::evaluate(
        &params.swap,
        residency,
        target.id,
        gain,
        params.value_usd,
        queues.hard_queued,
        forced,
        now,
    ) {
        Ok(plan) => target.route(Reason::Swap, Some(plan)),
        Err(denial) => match alternative {
            Some(a) => a.route(Reason::SwapDenied(denial), None),
            None => Decision::Unavailable {
                reason: Unavailable::SwapRequired(denial),
            },
        },
    }
}
