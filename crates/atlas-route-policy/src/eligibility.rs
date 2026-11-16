// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the eligible set E. It is computed from the static policy, the request class, the locality
//! header and the backends' health and capacity, before any prediction exists; the predictor and the decision
//! see only E, and only `eligible` and `tighten_for_pii` can build one, so a chosen target is a member of E by
//! construction. Local-only (policy flag, privileged class or `x-atlas-locality: local`) keeps local backends
//! only. A header or a PII signal can only remove members, never add any.

use crate::capacity::{self, Fit};
use crate::swap::SwapPolicy;
use crate::types::{
    Backend, BackendId, Locality, PiiRule, Preference, RequestClass, StaticPolicy, Unavailable,
};

/// Why a configured backend is not in E.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exclusion {
    /// Not on the policy's allow-list.
    NotAllowed,
    Unhealthy,
    /// Local-only and the backend is not local.
    NotLocal,
    /// The prompt exceeds the backend's window.
    Capacity,
    /// A PII signal restricted E to local backends.
    Pii,
    /// The same id appeared earlier in the backend list.
    Duplicate,
}

/// One member of E with the facts the decision needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Member {
    pub id: BackendId,
    pub local: bool,
    /// The capacity verdict after settling (`Fits` or `Unknown`; `Exceeds` never reaches E).
    pub fit: Fit,
    /// This request's prompt size in the member's own units, when counted (the switching-cost context length).
    pub prompt_tokens: Option<u64>,
}

/// The decision knobs of the static policy, carried inside E so `choose` decides under the policy that built E.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionParams {
    pub preference: Preference,
    pub min_p_pass: f64,
    pub value_usd: f64,
    pub swap: SwapPolicy,
}

impl DecisionParams {
    pub fn from_policy(policy: &StaticPolicy) -> Self {
        Self {
            preference: policy.preference,
            min_p_pass: policy.min_p_pass,
            value_usd: policy.value_usd,
            swap: policy.swap,
        }
    }
}

/// The eligible set: members sorted by id, the exclusions, and the policy knobs.
#[derive(Clone, Debug, PartialEq)]
pub struct EligibleSet {
    members: Vec<Member>,
    excluded: Vec<(BackendId, Exclusion)>,
    local_only: bool,
    params: DecisionParams,
    empty_reason: Unavailable,
}

impl EligibleSet {
    pub fn members(&self) -> &[Member] {
        &self.members
    }

    pub fn member(&self, id: BackendId) -> Option<&Member> {
        self.members.iter().find(|m| m.id == id)
    }

    pub fn contains(&self, id: BackendId) -> bool {
        self.member(id).is_some()
    }

    pub fn ids(&self) -> Vec<BackendId> {
        self.members.iter().map(|m| m.id).collect()
    }

    pub fn len(&self) -> usize {
        self.members.len()
    }

    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Every configured backend that is not a member, with the first reason that excluded it.
    pub fn excluded(&self) -> &[(BackendId, Exclusion)] {
        &self.excluded
    }

    /// Whether E is restricted to local backends (policy, class, header or PII).
    pub fn local_only(&self) -> bool {
        self.local_only
    }

    pub fn params(&self) -> &DecisionParams {
        &self.params
    }

    /// What an empty E means for the front (it answers 503 with this label). Meaningless for a non-empty E.
    pub fn empty_reason(&self) -> Unavailable {
        self.empty_reason
    }
}

/// Computes E. Backends are considered in id order; a repeated id keeps its first occurrence. A backend is
/// excluded, in this order, when it is off the allow-list, unhealthy, or not local while local-only holds; the
/// survivors' capacity verdicts are then settled together (`capacity::settle`) and a shortfall excludes.
pub fn eligible(
    class: RequestClass,
    headers: &Locality,
    policy: &StaticPolicy,
    backends: &[Backend],
) -> EligibleSet {
    let local_only =
        policy.local_only || class == RequestClass::Privileged || headers.require_local;
    let mut order: Vec<&Backend> = backends.iter().collect();
    order.sort_by_key(|b| b.id);

    let mut excluded: Vec<(BackendId, Exclusion)> = Vec::new();
    let mut candidates: Vec<&Backend> = Vec::new();
    for b in order {
        let duplicate = candidates.last().is_some_and(|c| c.id == b.id)
            || excluded.last().is_some_and(|(id, _)| *id == b.id);
        let allowed = policy
            .allowed
            .as_ref()
            .is_none_or(|list| list.contains(&b.id));
        let reason = if duplicate {
            Some(Exclusion::Duplicate)
        } else if !allowed {
            Some(Exclusion::NotAllowed)
        } else if !b.healthy {
            Some(Exclusion::Unhealthy)
        } else if local_only && !b.local {
            Some(Exclusion::NotLocal)
        } else {
            None
        };
        match reason {
            Some(r) => excluded.push((b.id, r)),
            None => candidates.push(b),
        }
    }

    let mut fits: Vec<Fit> = candidates
        .iter()
        .map(|b| {
            b.prompt_tokens
                .map_or(Fit::Unknown, |t| capacity::fits(&b.window, t))
        })
        .collect();
    let windows: Vec<&capacity::CapacityFact> = candidates.iter().map(|b| &b.window).collect();
    capacity::settle(&mut fits, &windows);

    let mut members = Vec::with_capacity(candidates.len());
    for (b, fit) in candidates.iter().zip(&fits) {
        if *fit == Fit::Exceeds {
            excluded.push((b.id, Exclusion::Capacity));
        } else {
            members.push(Member {
                id: b.id,
                local: b.local,
                fit: *fit,
                prompt_tokens: b.prompt_tokens,
            });
        }
    }
    excluded.sort_by_key(|(id, _)| *id);

    let empty_reason = if !candidates.is_empty() {
        Unavailable::ContextExceedsAll
    } else if local_only {
        Unavailable::LocalOnlyNoLocal
    } else {
        Unavailable::NoBackend
    };
    EligibleSet {
        members,
        excluded,
        local_only,
        params: DecisionParams::from_policy(policy),
        empty_reason,
    }
}

/// The PII signal for this request (ARC's PII head or any other detector). It may only add restrictions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PiiSignal {
    pub detected: bool,
}

/// Applies the PII signal to E (after the signals were computed over E). A detected signal under
/// `PiiRule::LocalOnly` removes every non-local member; anything else returns E unchanged. The result is always
/// a subset of `e`.
pub fn tighten_for_pii(e: &EligibleSet, pii: PiiSignal, policy: &StaticPolicy) -> EligibleSet {
    if !pii.detected || policy.pii != PiiRule::LocalOnly {
        return e.clone();
    }
    let mut out = e.clone();
    let (local, cloud): (Vec<Member>, Vec<Member>) =
        e.members.iter().copied().partition(|m| m.local);
    out.members = local;
    out.excluded
        .extend(cloud.iter().map(|m| (m.id, Exclusion::Pii)));
    out.excluded.sort_by_key(|(id, _)| *id);
    out.local_only = true;
    if out.members.is_empty() && !e.members.is_empty() {
        out.empty_reason = Unavailable::PiiNoLocal;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacity::CapacityFact;

    fn backends() -> Vec<Backend> {
        vec![
            Backend::cloud(3, "cloud"),
            Backend::local(1, "4b")
                .with_window(CapacityFact::measured(1000, 100))
                .with_prompt_tokens(500),
            Backend::local(2, "27b").unhealthy(),
        ]
    }

    #[test]
    fn members_are_sorted_and_reasons_recorded() {
        let e = eligible(
            RequestClass::Standard,
            &Locality::any(),
            &StaticPolicy::default(),
            &backends(),
        );
        assert_eq!(e.ids(), vec![BackendId(1), BackendId(3)]);
        assert_eq!(e.member(BackendId(1)).map(|m| m.fit), Some(Fit::Fits));
        assert_eq!(e.member(BackendId(3)).map(|m| m.fit), Some(Fit::Unknown));
        assert_eq!(e.excluded(), &[(BackendId(2), Exclusion::Unhealthy)]);
        assert!(!e.local_only());
    }

    #[test]
    fn every_local_only_source_restricts() {
        let p = StaticPolicy::default();
        let lo = StaticPolicy {
            local_only: true,
            ..StaticPolicy::default()
        };
        let cases = [
            eligible(RequestClass::Privileged, &Locality::any(), &p, &backends()),
            eligible(RequestClass::Standard, &Locality::local(), &p, &backends()),
            eligible(RequestClass::Standard, &Locality::any(), &lo, &backends()),
        ];
        for e in cases {
            assert_eq!(e.ids(), vec![BackendId(1)]);
            assert!(e.local_only());
        }
    }

    #[test]
    fn empty_reasons_and_duplicates() {
        let lo = StaticPolicy {
            local_only: true,
            ..StaticPolicy::default()
        };
        let only_cloud = [Backend::cloud(5, "c")];
        let e = eligible(RequestClass::Standard, &Locality::any(), &lo, &only_cloud);
        assert!(e.is_empty());
        assert_eq!(e.empty_reason(), Unavailable::LocalOnlyNoLocal);
        let none = eligible(
            RequestClass::Standard,
            &Locality::any(),
            &StaticPolicy::default(),
            &[],
        );
        assert_eq!(none.empty_reason(), Unavailable::NoBackend);
        let big = [Backend::local(1, "a")
            .with_window(CapacityFact::measured(10, 0))
            .with_prompt_tokens(11)];
        let short = eligible(
            RequestClass::Standard,
            &Locality::any(),
            &StaticPolicy::default(),
            &big,
        );
        assert_eq!(short.empty_reason(), Unavailable::ContextExceedsAll);
        let dup = [Backend::local(1, "a"), Backend::cloud(1, "b")];
        let d = eligible(
            RequestClass::Standard,
            &Locality::any(),
            &StaticPolicy::default(),
            &dup,
        );
        assert_eq!(d.members().len(), 1);
        assert!(d.members()[0].local);
        assert_eq!(d.excluded(), &[(BackendId(1), Exclusion::Duplicate)]);
    }

    #[test]
    fn pii_only_removes() {
        let p = StaticPolicy::default();
        let e = eligible(RequestClass::Standard, &Locality::any(), &p, &backends());
        assert_eq!(tighten_for_pii(&e, PiiSignal { detected: false }, &p), e);
        let t = tighten_for_pii(&e, PiiSignal { detected: true }, &p);
        assert_eq!(t.ids(), vec![BackendId(1)]);
        assert!(t.local_only());
        let ignore = StaticPolicy {
            pii: PiiRule::Ignore,
            ..StaticPolicy::default()
        };
        assert_eq!(
            tighten_for_pii(&e, PiiSignal { detected: true }, &ignore),
            e
        );
        let cloud_only = eligible(
            RequestClass::Standard,
            &Locality::any(),
            &p,
            &[Backend::cloud(4, "c")],
        );
        let t2 = tighten_for_pii(&cloud_only, PiiSignal { detected: true }, &p);
        assert!(t2.is_empty());
        assert_eq!(t2.empty_reason(), Unavailable::PiiNoLocal);
    }
}
