// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the golden decision table (30 scripted requests over one fixed backend set, each with its
//! expected target, reason and swap-denial label, computed by hand from the rules) and the details of a few
//! rows (swap plans, the held target's `p`, the switching cost). Event inference pairs: `tests/event_pairs.rs`.
//!
//! Backends: 1 `atlas-4b` (local, resident), 2 `atlas-27b` (local, not resident, swap 40 s), 3 `cloud-a`
//! (3 / 15 / 0.3 / 3.75 dollars per million input / output / cache-read / cache-write), 4 `cloud-b`
//! (0.5 / 2 / 0.05 / 0.5). Every prediction uses 10,000 input and 1,000 output tokens, so a cloud-a call costs
//! 0.045 and a cloud-b call 0.007 dollars; local calls cost 0. Default `p`: 0.90, 0.95, 0.97, 0.93. Default
//! policy: prefer-local, `min_p_pass` 0.85, V = 4.5, swaps off, `lambda_wait` 0.01, dwell 10 min, cap 4 per hour.
//! Default utilities: 4b 4.05, 27b 4.275, cloud-a 4.32, cloud-b 4.178.

use std::time::{Duration, Instant};

use atlas_route_policy::{
    Backend, BackendId, CapacityFact, Decision, Event, Locality, Overflow, PiiSignal, Prediction,
    Predictions, Preference, Price, Prices, Provenance, Queues, RequestClass, Residency,
    ResidencyEntry, StaticPolicy, SwapHistory, TokenUsage, choose, eligible, tighten_for_pii,
};

const B4: BackendId = BackendId(1);
const B27: BackendId = BackendId(2);
const CA: BackendId = BackendId(3);
const CB: BackendId = BackendId(4);

struct Sc {
    class: RequestClass,
    loc: Locality,
    policy: StaticPolicy,
    backends: Vec<Backend>,
    preds: Option<Predictions>,
    prices: Prices,
    residency: Residency,
    queues: Queues,
    last: Option<BackendId>,
    event: Event,
    pii: bool,
    now: Instant,
}

fn usage() -> TokenUsage {
    TokenUsage::new(10_000.0, 1_000.0)
}

fn preds(p4: f64, p27: f64, pa: f64, pb: f64) -> Predictions {
    Predictions::new()
        .with(B4, Prediction::new(p4, usage()))
        .with(B27, Prediction::new(p27, usage()))
        .with(CA, Prediction::new(pa, usage()))
        .with(CB, Prediction::new(pb, usage()))
}

fn base() -> Sc {
    let local_window = CapacityFact::measured(262_144, 8_192);
    // A base far in the future, so subtracting history offsets never underflows the monotonic clock.
    let now = Instant::now() + Duration::from_secs(4 * 3600);
    Sc {
        class: RequestClass::Standard,
        loc: Locality::any(),
        policy: StaticPolicy::default(),
        backends: vec![
            Backend::local(1, "atlas-4b")
                .with_window(local_window)
                .with_prompt_tokens(10_000),
            Backend::local(2, "atlas-27b")
                .with_window(local_window)
                .with_prompt_tokens(10_000),
            Backend::cloud(3, "cloud-a").with_prompt_tokens(10_000),
            Backend::cloud(4, "cloud-b").with_prompt_tokens(10_000),
        ],
        preds: Some(preds(0.90, 0.95, 0.97, 0.93)),
        prices: Prices::new()
            .with(CA, Price::new(3.0, 15.0, 0.3, 3.75))
            .with(CB, Price::new(0.5, 2.0, 0.05, 0.5)),
        residency: Residency {
            entries: vec![
                ResidencyEntry {
                    id: B4,
                    resident: true,
                    swap_seconds: Some(5.0),
                },
                ResidencyEntry {
                    id: B27,
                    resident: false,
                    swap_seconds: Some(40.0),
                },
            ],
            history: SwapHistory::new(),
        },
        queues: Queues {
            entries: vec![],
            hard_queued: 1,
        },
        last: None,
        event: Event::TaskStart,
        pii: false,
        now,
    }
}

fn run(sc: &Sc) -> Decision {
    let e = eligible(sc.class, &sc.loc, &sc.policy, &sc.backends);
    let e = tighten_for_pii(&e, PiiSignal { detected: sc.pii }, &sc.policy);
    let d = choose(
        &e,
        sc.preds.as_ref(),
        &sc.prices,
        &sc.residency,
        &sc.queues,
        sc.last,
        sc.event,
        sc.now,
    );
    if let Some(t) = d.target() {
        assert!(e.contains(t), "target {t} is not in E");
    }
    d
}

fn set_p4(sc: &mut Sc, p: f64) {
    if let Some(ps) = sc.preds.as_mut() {
        ps.insert(B4, Prediction::new(p, usage()));
    }
}

fn swap_case(sc: &mut Sc) {
    sc.policy.local_only = true;
    sc.policy.swap.allow = true;
    set_p4(sc, 0.80);
}

fn set_swap_seconds(sc: &mut Sc, s: Option<f64>) {
    for e in &mut sc.residency.entries {
        if e.id == B27 {
            e.swap_seconds = s;
        }
    }
}

fn mins_ago(sc: &Sc, m: u64) -> Instant {
    sc.now - Duration::from_secs(m * 60)
}

/// Expected outcome: `Ok((target, reason, denial))` for a route, `Err((reason, denial))` for unavailable.
type Want =
    Result<(BackendId, &'static str, Option<&'static str>), (&'static str, Option<&'static str>)>;

type Row = (&'static str, fn(&mut Sc), Want);

/// One table row. The `fn(&mut Sc)` parameter gives each closure its expected type, so it coerces to a
/// function pointer.
fn row(name: &'static str, edit: fn(&mut Sc), want: Want) -> Row {
    (name, edit, want)
}

fn rows() -> Vec<Row> {
    vec![
        row(
            "01 default: resident 4b passes 0.85",
            |_| {},
            Ok((B4, "prefer_local", None)),
        ),
        row(
            "02 4b below threshold: utility picks cloud-a",
            |s| set_p4(s, 0.80),
            Ok((CA, "utility", None)),
        ),
        row(
            "03 utility mode",
            |s| s.policy.preference = Preference::Utility,
            Ok((CA, "utility", None)),
        ),
        row(
            "04 local-only: 27b best but swaps off",
            |s| {
                s.policy.local_only = true;
                set_p4(s, 0.80);
            },
            Ok((B4, "swap_denied", Some("disabled"))),
        ),
        row(
            "05 privileged class",
            |s| s.class = RequestClass::Privileged,
            Ok((B4, "prefer_local", None)),
        ),
        row(
            "06 header local",
            |s| {
                s.loc = Locality::local();
                set_p4(s, 0.80);
            },
            Ok((B4, "swap_denied", Some("disabled"))),
        ),
        row(
            "07 swap allowed: 0.15 * 4.5 * 1 > 40 * 0.01",
            swap_case,
            Ok((B27, "swap", None)),
        ),
        row(
            "08 dwell: last swap 5 minutes ago",
            |s| {
                swap_case(s);
                s.residency.history = SwapHistory::from_swaps([mins_ago(s, 5)]);
            },
            Ok((B4, "swap_denied", Some("dwell"))),
        ),
        row(
            "09 hourly cap: four swaps in the hour",
            |s| {
                swap_case(s);
                let t = [
                    mins_ago(s, 55),
                    mins_ago(s, 40),
                    mins_ago(s, 25),
                    mins_ago(s, 12),
                ];
                s.residency.history = SwapHistory::from_swaps(t);
            },
            Ok((B4, "swap_denied", Some("hourly_cap"))),
        ),
        row(
            "10 no hard request queued",
            |s| {
                swap_case(s);
                s.queues.hard_queued = 0;
            },
            Ok((B4, "swap_denied", Some("gain"))),
        ),
        row(
            "11 swap too slow: 200 s",
            |s| {
                swap_case(s);
                set_swap_seconds(s, Some(200.0));
            },
            Ok((B4, "swap_denied", Some("gain"))),
        ),
        row(
            "12 unknown swap seconds",
            |s| {
                swap_case(s);
                set_swap_seconds(s, None);
            },
            Ok((B4, "swap_denied", Some("unknown_swap_seconds"))),
        ),
        row(
            "13 no predictions: resident model",
            |s| s.preds = None,
            Ok((B4, "no_prediction", None)),
        ),
        row(
            "14 only 27b eligible, swaps off",
            |s| {
                s.preds = None;
                s.policy.local_only = true;
                s.backends[0].healthy = false;
            },
            Err(("swap_required", Some("disabled"))),
        ),
        row(
            "15 only 27b eligible, forced swap",
            |s| {
                s.preds = None;
                s.policy.local_only = true;
                s.policy.swap.allow = true;
                s.backends[0].healthy = false;
            },
            Ok((B27, "swap", None)),
        ),
        row(
            "16 local-only with no healthy local",
            |s| {
                s.policy.local_only = true;
                s.backends[0].healthy = false;
                s.backends[1].healthy = false;
            },
            Err(("local_only_no_local", None)),
        ),
        row(
            "17 tool result keeps cloud-a",
            |s| {
                s.event = Event::ToolResult;
                s.last = Some(CA);
            },
            Ok((CA, "hold", None)),
        ),
        row(
            "18 tool result, last no longer eligible",
            |s| {
                s.event = Event::ToolResult;
                s.last = Some(CA);
                s.policy.local_only = true;
            },
            Ok((B4, "prefer_local", None)),
        ),
        row(
            "19 user turn: switching cost keeps cloud-b",
            |s| {
                s.policy.preference = Preference::Utility;
                s.event = Event::UserMessage;
                s.last = Some(CB);
                s.preds = Some(preds(0.90, 0.95, 0.97, 0.955));
            },
            Ok((CB, "utility", None)),
        ),
        row(
            "20 task start: no switching cost",
            |s| {
                s.policy.preference = Preference::Utility;
                s.last = Some(CB);
                s.preds = Some(preds(0.90, 0.95, 0.97, 0.955));
            },
            Ok((CA, "utility", None)),
        ),
        row(
            "21 prompt over the local windows",
            |s| {
                s.backends[0].prompt_tokens = Some(300_000);
                s.backends[1].prompt_tokens = Some(300_000);
            },
            Ok((CA, "utility", None)),
        ),
        row(
            "22 prompt over every window",
            |s| {
                for b in &mut s.backends {
                    b.prompt_tokens = Some(300_000);
                }
                s.backends[2].window = CapacityFact::measured(200_000, 8_192);
                s.backends[3].window = CapacityFact::measured(200_000, 8_192);
            },
            Err(("context_exceeds_all", None)),
        ),
        row(
            "23 estimated shortfall on a refusing runtime yields",
            |s| {
                for b in &mut s.backends {
                    b.prompt_tokens = Some(300_000);
                }
                s.backends[2].window = CapacityFact::measured(200_000, 8_192)
                    .with_provenance(Provenance::Estimated)
                    .with_overflow(Overflow::Refuses);
                s.backends[3].window = CapacityFact::measured(200_000, 8_192);
            },
            Ok((CA, "utility", None)),
        ),
        row(
            "24 estimated local fit is not a proven fit",
            |s| {
                s.backends[0].window =
                    CapacityFact::measured(262_144, 8_192).with_provenance(Provenance::Estimated)
            },
            Ok((CA, "utility", None)),
        ),
        row(
            "25 PII restricts to local",
            |s| {
                s.pii = true;
                s.policy.preference = Preference::Utility;
            },
            Ok((B4, "swap_denied", Some("disabled"))),
        ),
        row(
            "26 partial predictions",
            |s| {
                s.preds = Some(
                    Predictions::new()
                        .with(B4, Prediction::new(0.80, usage()))
                        .with(CB, Prediction::new(0.93, usage())),
                );
            },
            Ok((CB, "utility", None)),
        ),
        row(
            "27 allow-list without cloud-a: resident alternative is cloud-b",
            |s| {
                s.policy.preference = Preference::Utility;
                s.policy.allowed = Some(vec![B4, B27, CB]);
            },
            Ok((CB, "swap_denied", Some("disabled"))),
        ),
        row(
            "28 threshold 1.0 is utility",
            |s| s.policy.min_p_pass = 1.0,
            Ok((CA, "utility", None)),
        ),
        row(
            "29 Swift preset",
            |s| {
                s.policy = StaticPolicy {
                    preference: Preference::Utility,
                    ..StaticPolicy::swift()
                };
            },
            Ok((B4, "swap_denied", Some("disabled"))),
        ),
        row(
            "30 side call keeps 4b",
            |s| {
                s.policy.preference = Preference::Utility;
                s.event = Event::Sidecall;
                s.last = Some(B4);
            },
            Ok((B4, "hold", None)),
        ),
    ]
}

#[test]
fn golden_decision_table() {
    let table = rows();
    assert_eq!(table.len(), 30);
    let mut failures = Vec::new();
    for (name, edit, want) in table {
        let mut sc = base();
        edit(&mut sc);
        let d = run(&sc);
        let got: Want = match d.target() {
            Some(t) => Ok((t, d.reason_label(), d.denial_label())),
            None => Err((d.reason_label(), d.denial_label())),
        };
        if got != want {
            failures.push(format!("{name}: want {want:?}, got {got:?} ({d:?})"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} rows differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn golden_details() {
    // Row 07: the plan evicts 4b, loads 27b, carries the estimate, and is not forced.
    let mut sc = base();
    swap_case(&mut sc);
    let plan = run(&sc).swap().copied();
    assert_eq!(
        plan.map(|p| (p.from, p.to, p.seconds, p.forced)),
        Some((Some(B4), B27, Some(40.0), false))
    );
    // Row 15: a forced swap.
    let mut sc = base();
    sc.preds = None;
    sc.policy.local_only = true;
    sc.policy.swap.allow = true;
    sc.backends[0].healthy = false;
    assert_eq!(run(&sc).swap().map(|p| p.forced), Some(true));
    // Row 17: the held target reports its own p; row 01 reports 4b's p and zero cost.
    let mut sc = base();
    sc.event = Event::ToolResult;
    sc.last = Some(CA);
    assert!(matches!(run(&sc), Decision::Route { p: Some(p), .. } if (p - 0.97).abs() < 1e-12));
    let d = run(&base());
    assert!(
        matches!(d, Decision::Route { p: Some(p), cost: Some(c), .. } if (p - 0.90).abs() < 1e-12 && c == 0.0)
    );
    assert_eq!(d.route_header("atlas-4b"), "atlas-4b;prefer_local;p=0.900");
    // Row 19: the switching cost of cloud-a is 10,000 * (3.75 - 0.3) / 1e6.
    let mut sc = base();
    sc.policy.preference = Preference::Utility;
    sc.event = Event::UserMessage;
    sc.last = Some(CB);
    sc.policy.allowed = Some(vec![CA]);
    assert!(
        matches!(run(&sc), Decision::Route { cost: Some(c), .. } if (c - (0.045 + 0.0345)).abs() < 1e-12)
    );
}
