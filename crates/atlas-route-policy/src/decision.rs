// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the policy's answer. A route (target, reason, predicted `p` and cost, an optional swap plan)
//! or unavailable (the front answers 503 and never falls back), with the stable labels the front puts in the
//! `x-atlas-route` header and in its metrics.

use crate::swap::{SwapDenial, SwapPlan};
use crate::types::BackendId;

/// Why a route was chosen. The labels are the `reason` in the `x-atlas-route` header and in metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The resident local model fits and passes `min_p_pass`.
    PreferLocal,
    /// Highest expected utility over the eligible set.
    Utility,
    /// Not a switching boundary: the previous target is kept.
    Hold,
    /// No usable prediction: the resident model.
    NoPrediction,
    /// The best target was not resident and the swap rule allowed loading it.
    Swap,
    /// The best target was not resident and the swap was denied: served on a resident model instead.
    SwapDenied(SwapDenial),
}

impl Reason {
    pub fn as_str(&self) -> &'static str {
        match self {
            Reason::PreferLocal => "prefer_local",
            Reason::Utility => "utility",
            Reason::Hold => "hold",
            Reason::NoPrediction => "no_prediction",
            Reason::Swap => "swap",
            Reason::SwapDenied(_) => "swap_denied",
        }
    }
}

/// Why no backend can serve. The front answers 503 with this label and never falls back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unavailable {
    NoBackend,
    /// Local-only (policy, request class or header) and no healthy, allowed local backend.
    LocalOnlyNoLocal,
    /// Every remaining backend reports that the prompt exceeds its window.
    ContextExceedsAll,
    /// A PII signal restricted the set to local backends and none is eligible.
    PiiNoLocal,
    /// Only non-resident models are eligible and the swap was denied.
    SwapRequired(SwapDenial),
}

impl Unavailable {
    pub fn as_str(&self) -> &'static str {
        match self {
            Unavailable::NoBackend => "no_backend",
            Unavailable::LocalOnlyNoLocal => "local_only_no_local",
            Unavailable::ContextExceedsAll => "context_exceeds_all",
            Unavailable::PiiNoLocal => "pii_no_local",
            Unavailable::SwapRequired(_) => "swap_required",
        }
    }
}

/// The policy's answer.
#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    Route {
        /// Always a member of the eligible set the decision was made from.
        target: BackendId,
        reason: Reason,
        /// The target's predicted success probability, when it had a prediction.
        p: Option<f64>,
        /// The target's predicted cost in US dollars (usage plus switching), when it was scored.
        cost: Option<f64>,
        /// Present when serving the target needs a swap the rule allowed.
        swap: Option<SwapPlan>,
    },
    Unavailable {
        reason: Unavailable,
    },
}

impl Decision {
    pub fn target(&self) -> Option<BackendId> {
        match self {
            Decision::Route { target, .. } => Some(*target),
            Decision::Unavailable { .. } => None,
        }
    }

    /// The reason label (`prefer_local`, `utility`, ..., or the unavailable label).
    pub fn reason_label(&self) -> &'static str {
        match self {
            Decision::Route { reason, .. } => reason.as_str(),
            Decision::Unavailable { reason } => reason.as_str(),
        }
    }

    /// The swap denial label, when the decision carries one.
    pub fn denial_label(&self) -> Option<&'static str> {
        match self {
            Decision::Route {
                reason: Reason::SwapDenied(d),
                ..
            }
            | Decision::Unavailable {
                reason: Unavailable::SwapRequired(d),
            } => Some(d.as_str()),
            _ => None,
        }
    }

    pub fn swap(&self) -> Option<&SwapPlan> {
        match self {
            Decision::Route { swap, .. } => swap.as_ref(),
            Decision::Unavailable { .. } => None,
        }
    }

    /// The `x-atlas-route` header value: `<target>;<reason>;p=<float>` (`p` omitted when unknown), or
    /// `none;<reason>` when unavailable.
    pub fn route_header(&self, target_name: &str) -> String {
        match self {
            Decision::Route {
                reason, p: Some(p), ..
            } => {
                format!("{target_name};{};p={p:.3}", reason.as_str())
            }
            Decision::Route {
                reason, p: None, ..
            } => format!("{target_name};{}", reason.as_str()),
            Decision::Unavailable { reason } => format!("none;{}", reason.as_str()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_and_header() {
        let r = Decision::Route {
            target: BackendId(1),
            reason: Reason::PreferLocal,
            p: Some(0.9),
            cost: Some(0.0),
            swap: None,
        };
        assert_eq!(r.route_header("atlas-4b"), "atlas-4b;prefer_local;p=0.900");
        assert_eq!(r.target(), Some(BackendId(1)));
        assert_eq!(r.denial_label(), None);
        let d = Decision::Route {
            target: BackendId(1),
            reason: Reason::SwapDenied(SwapDenial::Dwell),
            p: None,
            cost: None,
            swap: None,
        };
        assert_eq!(d.route_header("atlas-4b"), "atlas-4b;swap_denied");
        assert_eq!(d.denial_label(), Some("dwell"));
        let u = Decision::Unavailable {
            reason: Unavailable::SwapRequired(SwapDenial::Disabled),
        };
        assert_eq!(u.route_header("x"), "none;swap_required");
        assert_eq!(u.reason_label(), "swap_required");
        assert_eq!(u.denial_label(), Some("disabled"));
        assert_eq!(u.target(), None);
    }
}
