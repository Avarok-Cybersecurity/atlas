// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the eligibility properties of the routing policy over seeded random worlds: the target is
//! always in E and decisions are deterministic; local-only never routes to a non-local backend; adding a cloud
//! backend changes no local-only decision; a PII signal only removes members; and the mutation sanity check (an
//! eligibility filter that ignores locality must make the local-only property fail). The value, threshold and
//! hysteresis properties are in `tests/properties_value.rs`; the generator is in `tests/support/mod.rs`.

mod support;

use atlas_route_policy::{
    Backend, BackendId, Decision, EligibleSet, Locality, PiiRule, PiiSignal, Prediction, Price,
    QueueEntry, RequestClass, StaticPolicy, TokenUsage, Unavailable, eligible, tighten_for_pii,
};
use support::{
    CASES, Elig, Rng, SEED, base_now, decide_with, force_local_only, random_backend, random_world,
};

/// The mutant: an eligibility filter that drops every locality restriction.
fn mutant_eligible(
    _: RequestClass,
    _: &Locality,
    policy: &StaticPolicy,
    backends: &[Backend],
) -> EligibleSet {
    let relaxed = StaticPolicy {
        local_only: false,
        ..policy.clone()
    };
    eligible(RequestClass::Standard, &Locality::any(), &relaxed, backends)
}

/// Local-only (by one of its three sources) never routes to a non-local backend. `Err` names the first
/// counterexample.
fn check_local_only(elig: Elig, cases: u64, seed: u64) -> Result<(), String> {
    for case in 0..cases {
        let mut r = Rng::for_case(seed, case);
        let mut w = random_world(&mut r, base_now());
        w.pii = false;
        force_local_only(&mut r, &mut w);
        let (_, d) = decide_with(&w, elig);
        let outside = d
            .target()
            .is_some_and(|t| !w.backends.iter().any(|b| b.id == t && b.local));
        if outside {
            return Err(format!(
                "case {case}: local-only routed to a non-local backend: {d:?}"
            ));
        }
    }
    Ok(())
}

#[test]
fn target_is_always_in_e_and_decisions_are_deterministic() {
    for case in 0..CASES * 2 {
        let mut r = Rng::for_case(SEED, case);
        let w = random_world(&mut r, base_now());
        let full = eligible(w.class, &w.loc, &w.policy, &w.backends);
        let (e, d) = decide_with(&w, eligible);
        for m in e.members() {
            assert!(
                full.contains(m.id),
                "case {case}: PII tightening added {}",
                m.id
            );
            let b = w.backends.iter().find(|b| b.id == m.id);
            assert!(
                b.is_some_and(|b| b.healthy),
                "case {case}: unhealthy member {}",
                m.id
            );
            let allowed = w.policy.allowed.as_ref().is_none_or(|a| a.contains(&m.id));
            assert!(allowed, "case {case}: member {} is not allowed", m.id);
        }
        match &d {
            Decision::Route { target, .. } => {
                assert!(
                    e.contains(*target),
                    "case {case}: target not in E {:?}: {d:?}",
                    e.ids()
                );
            }
            Decision::Unavailable { reason } => {
                let swap_blocked = matches!(reason, Unavailable::SwapRequired(_));
                assert!(
                    e.is_empty() || swap_blocked,
                    "case {case}: non-empty E but {d:?}"
                );
            }
        }
        assert_eq!(
            decide_with(&w, eligible).1,
            d,
            "case {case}: not deterministic"
        );
    }
}

#[test]
fn local_only_never_routes_outside_local() {
    if let Err(e) = check_local_only(eligible, CASES, SEED) {
        panic!("{e}");
    }
}

#[test]
fn mutation_sanity_flipped_eligibility_filter_is_caught() {
    assert!(check_local_only(eligible, CASES, SEED).is_ok());
    assert!(
        check_local_only(mutant_eligible, CASES, SEED).is_err(),
        "the local-only property did not catch an eligibility filter that ignores locality"
    );
}

#[test]
fn adding_a_cloud_backend_changes_no_local_only_decision() {
    for case in 0..CASES {
        let mut r = Rng::for_case(SEED ^ 0x0C10_0D00, case);
        let mut w = random_world(&mut r, base_now());
        force_local_only(&mut r, &mut w);
        let (_, before) = decide_with(&w, eligible);
        let extra = BackendId(50);
        let mut w2 = w.clone();
        let mut cloud = random_backend(&mut r, extra.0, false);
        cloud.healthy = true;
        w2.backends.push(cloud);
        if let Some(a) = w2.policy.allowed.as_mut() {
            a.push(extra);
        }
        if let Some(p) = w2.preds.as_mut() {
            p.insert(extra, Prediction::new(1.0, TokenUsage::new(1.0, 1.0)));
        }
        w2.prices.insert(extra, Price::ZERO);
        w2.queues.entries.push(QueueEntry {
            id: extra,
            depth: 0,
        });
        let (_, after) = decide_with(&w2, eligible);
        assert_eq!(
            before, after,
            "case {case}: a cloud backend changed a local-only decision"
        );
    }
}

#[test]
fn pii_signal_only_removes_members() {
    for case in 0..CASES {
        let mut r = Rng::for_case(SEED ^ 0x0000_0911, case);
        let w = random_world(&mut r, base_now());
        let e = eligible(w.class, &w.loc, &w.policy, &w.backends);
        let off = tighten_for_pii(&e, PiiSignal { detected: false }, &w.policy);
        assert_eq!(off, e, "case {case}: a negative PII signal changed E");
        let t = tighten_for_pii(&e, PiiSignal { detected: true }, &w.policy);
        assert!(
            t.members().iter().all(|m| e.contains(m.id)),
            "case {case}: PII added a member"
        );
        if w.policy.pii == PiiRule::LocalOnly {
            assert!(
                t.members().iter().all(|m| m.local),
                "case {case}: PII kept a non-local member"
            );
        }
    }
}
