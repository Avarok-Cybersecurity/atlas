// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the value, threshold and hysteresis properties of the routing policy over seeded random
//! worlds: the chosen `p` never falls as V rises (on exact dyadic grids, so the argmax argument holds without
//! rounding); prefer-local decides exactly as utility at threshold 1.0 or when every local `p` is below the
//! threshold; and a 300-step swap simulation never allows two swaps inside the dwell window or more swaps in a
//! rolling hour than the cap. The generator is in `tests/support/mod.rs`.

mod support;

use std::time::{Duration, Instant};

use atlas_route_policy::hysteresis::HOUR;
use atlas_route_policy::utility;
use atlas_route_policy::{
    Backend, BackendId, Decision, Event, Locality, Prediction, Predictions, Preference, Price,
    Prices, Queues, Reason, RequestClass, Residency, ResidencyEntry, StaticPolicy, SwapHistory,
    SwapPolicy, TokenUsage, choose, eligible,
};
use support::{CASES, Rng, SEED, World, base_now, decide_with, random_world};

/// A world on exact dyadic grids (p in 1/64, V in 1/8, costs in 1/64 dollars), every backend healthy, resident,
/// predicted and priced, so every utility is computed without rounding.
fn grid_world(r: &mut Rng, value_usd: f64) -> World {
    let mut w = random_world(r, base_now());
    let n = 2 + r.below(5) as u32;
    w.backends = (1..=n)
        .map(|id| {
            if r.chance(0.5) {
                Backend::local(id, "l")
            } else {
                Backend::cloud(id, "c")
            }
        })
        .collect();
    let mut preds = Predictions::new();
    let mut prices = Prices::new();
    for b in &w.backends {
        let input = 15_625.0 * r.below(8) as f64;
        let output = 15_625.0 * r.below(8) as f64;
        let p = r.below(65) as f64 / 64.0;
        preds.insert(b.id, Prediction::new(p, TokenUsage::new(input, output)));
        prices.insert(
            b.id,
            Price::new(r.below(21) as f64, r.below(21) as f64, 0.0, 0.0),
        );
    }
    w.preds = Some(preds);
    w.prices = prices;
    w.residency = Residency::default();
    w.policy = StaticPolicy {
        preference: Preference::Utility,
        value_usd,
        ..StaticPolicy::default()
    };
    w.class = RequestClass::Standard;
    w.loc = Locality::any();
    w.pii = false;
    w.event = Event::TaskStart;
    w.last = None;
    w
}

#[test]
fn chosen_p_is_monotone_in_value() {
    for case in 0..CASES {
        let mut r = Rng::for_case(SEED ^ 0x0001_7A1E, case);
        let k1 = 1 + r.below(80);
        let k2 = k1 + 1 + r.below(80);
        let (v1, v2) = (k1 as f64 / 8.0, k2 as f64 / 8.0);
        let state = r.next_u64();
        let (_, d1) = decide_with(&grid_world(&mut Rng(state), v1), eligible);
        let (_, d2) = decide_with(&grid_world(&mut Rng(state), v2), eligible);
        let (
            Decision::Route {
                reason: Reason::Utility,
                p: Some(p1),
                cost: Some(c1),
                ..
            },
            Decision::Route {
                reason: Reason::Utility,
                p: Some(p2),
                ..
            },
        ) = (&d1, &d2)
        else {
            panic!("case {case}: expected two utility routes, got {d1:?} and {d2:?}");
        };
        assert!(
            p2 >= p1,
            "case {case}: p fell from {p1} at V={v1} to {p2} at V={v2}"
        );
        let (u1, u2) = (
            utility::utility(v1, *p1, *c1),
            utility::utility(v2, *p1, *c1),
        );
        assert!(
            u2 >= u1,
            "case {case}: a candidate's utility fell as V rose"
        );
    }
}

fn with_preference(w: &World, preference: Preference, min_p_pass: f64) -> World {
    let mut out = w.clone();
    out.policy.preference = preference;
    out.policy.min_p_pass = min_p_pass;
    out
}

#[test]
fn prefer_local_equals_utility_at_threshold_one_or_below_it() {
    for case in 0..CASES {
        let mut r = Rng::for_case(SEED ^ 0x0050_1CA1, case);
        let w = random_world(&mut r, base_now());
        let by_utility = decide_with(&with_preference(&w, Preference::Utility, 0.85), eligible).1;
        let at_one = decide_with(&with_preference(&w, Preference::PreferLocal, 1.0), eligible).1;
        assert_eq!(by_utility, at_one, "case {case}: threshold 1.0");

        let mut below = w.clone();
        if let Some(preds) = below.preds.as_mut() {
            for b in w.backends.iter().filter(|b| b.local) {
                if let Some(p) = preds.get(b.id) {
                    preds.insert(b.id, Prediction::new(p.p_success * 0.89, p.usage));
                }
            }
        }
        let u = decide_with(&with_preference(&below, Preference::Utility, 0.9), eligible).1;
        let pl = decide_with(
            &with_preference(&below, Preference::PreferLocal, 0.9),
            eligible,
        )
        .1;
        assert_eq!(u, pl, "case {case}: every local p below the threshold");
    }
}

/// One 300-step run with swaps allowed; returns the swap times after checking each against dwell and cap.
fn simulate_swaps(seed: u64) -> Vec<Instant> {
    let mut r = Rng::for_case(SEED ^ 0x0000_D3E1, seed);
    let cap = 1 + r.below(5) as u32;
    let dwell = Duration::from_secs(600);
    let mut backends = vec![
        Backend::local(1, "a"),
        Backend::local(2, "b"),
        Backend::local(3, "c"),
    ];
    if r.chance(0.5) {
        backends.push(Backend::cloud(4, "cloud"));
    }
    let entries = (1..=3)
        .map(|id| ResidencyEntry {
            id: BackendId(id),
            resident: id == 1,
            swap_seconds: Some(r.unit() * 60.0),
        })
        .collect();
    let mut residency = Residency {
        entries,
        history: SwapHistory::new(),
    };
    let preference = if r.chance(0.5) {
        Preference::Utility
    } else {
        Preference::PreferLocal
    };
    let swap = SwapPolicy {
        allow: true,
        lambda_wait_usd_per_s: 1e-4,
        dwell,
        hourly_cap: cap,
    };
    let policy = StaticPolicy {
        preference,
        swap,
        ..StaticPolicy::default()
    };
    let prices = Prices::new().with(BackendId(4), Price::new(1.0, 4.0, 0.1, 1.0));
    let queues = Queues {
        entries: vec![],
        hard_queued: 1 + r.below(5) as u32,
    };
    let mut now = base_now();
    let mut last = None;
    let mut times: Vec<Instant> = Vec::new();
    for step in 0..300 {
        now += Duration::from_secs(r.below(400));
        let mut preds = Predictions::new();
        for b in &backends {
            preds.insert(
                b.id,
                Prediction::new(r.unit(), TokenUsage::new(1_000.0, 100.0)),
            );
        }
        let event = r.pick(&Event::ALL);
        let e = eligible(RequestClass::Standard, &Locality::any(), &policy, &backends);
        let d = choose(
            &e,
            Some(&preds),
            &prices,
            &residency,
            &queues,
            last,
            event,
            now,
        );
        if let Some(plan) = d.swap().copied() {
            if let Some(prev) = times.last() {
                let gap = now.duration_since(*prev);
                assert!(gap >= dwell, "seed {seed} step {step}: swaps {gap:?} apart");
            }
            let in_hour = times
                .iter()
                .filter(|t| now.duration_since(**t) < HOUR)
                .count();
            assert!(
                in_hour < cap as usize,
                "seed {seed} step {step}: cap {cap} exceeded"
            );
            times.push(now);
            residency.apply_swap(&plan, now);
        }
        last = d.target();
    }
    times
}

#[test]
fn hysteresis_never_allows_two_swaps_inside_the_dwell_window() {
    let total: usize = (0..200u64).map(|seed| simulate_swaps(seed).len()).sum();
    assert!(
        total > 0,
        "the simulation never swapped, so it proved nothing"
    );
}
