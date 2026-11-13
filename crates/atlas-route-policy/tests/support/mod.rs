// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the seeded world generator shared by the property tests: a SplitMix64 generator and random
//! backends (locality, health, windows, prompt sizes), policies, predictions, prices, residency with swap
//! history, queues, last target, event and PII signal, plus the decide helper and the local-only forcing.
//!
//! A seeded generator stands in for `proptest`, which is not in Cargo.lock (adding it would change the lock
//! file, a hotspot). Every failure message names its case number; a case is reproducible from it.
#![allow(dead_code)]

use std::time::{Duration, Instant};

use atlas_route_policy::{
    Backend, BackendId, CapacityFact, Decision, EligibleSet, Event, Locality, Overflow, PiiRule,
    PiiSignal, Prediction, Predictions, Preference, Price, Prices, Provenance, QueueEntry, Queues,
    RequestClass, Residency, ResidencyEntry, StaticPolicy, SwapHistory, SwapPolicy, TokenUsage,
    choose, tighten_for_pii,
};

pub const CASES: u64 = 2_000;
pub const SEED: u64 = 0x4154_4c41_535f_4c34;

/// SplitMix64: small, seeded, deterministic.
pub struct Rng(pub u64);

impl Rng {
    pub fn for_case(seed: u64, case: u64) -> Self {
        Self(seed ^ case.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

/// A base far in the future, so subtracting history offsets never underflows the monotonic clock.
pub fn base_now() -> Instant {
    Instant::now() + Duration::from_secs(4 * 3600)
}

#[derive(Clone)]
pub struct World {
    pub class: RequestClass,
    pub loc: Locality,
    pub policy: StaticPolicy,
    pub backends: Vec<Backend>,
    pub preds: Option<Predictions>,
    pub prices: Prices,
    pub residency: Residency,
    pub queues: Queues,
    pub last: Option<BackendId>,
    pub event: Event,
    pub pii: bool,
    pub now: Instant,
}

pub fn random_window(r: &mut Rng) -> CapacityFact {
    let window = 4_096 + r.below(300_000);
    let base = CapacityFact::measured(window, r.below(16_384)).with_margin(r.below(512));
    match r.below(7) {
        0 => CapacityFact::UNKNOWN,
        1 => base.with_provenance(Provenance::Estimated),
        2 => base
            .with_provenance(Provenance::Bounded)
            .with_overflow(Overflow::Refuses),
        3 => base
            .with_provenance(Provenance::Estimated)
            .with_overflow(Overflow::Refuses),
        4 => base.with_provenance(Provenance::ProviderReported),
        _ => base,
    }
}

pub fn random_prediction(r: &mut Rng) -> Prediction {
    let usage = TokenUsage {
        input: r.below(50_000) as f64,
        output: r.below(5_000) as f64,
        cache_read: r.below(20_000) as f64,
        cache_write: r.below(5_000) as f64,
    };
    Prediction::new(r.unit(), usage)
}

pub fn random_price(r: &mut Rng) -> Price {
    Price::new(
        r.unit() * 20.0,
        r.unit() * 80.0,
        r.unit() * 2.0,
        r.unit() * 25.0,
    )
}

pub fn random_backend(r: &mut Rng, id: u32, local: bool) -> Backend {
    let mut b = if local {
        Backend::local(id, format!("local-{id}"))
    } else {
        Backend::cloud(id, format!("cloud-{id}"))
    };
    b.healthy = r.chance(0.9);
    b.window = random_window(r);
    if r.chance(0.8) {
        b.prompt_tokens = Some(r.below(200_000));
    }
    b
}

fn random_policy(r: &mut Rng, n: u32) -> StaticPolicy {
    let local_only = r.chance(0.3);
    let allowed = if r.chance(0.3) {
        Some((1..=n).filter(|_| r.chance(0.7)).map(BackendId).collect())
    } else {
        None
    };
    let preference = if r.chance(0.5) {
        Preference::Utility
    } else {
        Preference::PreferLocal
    };
    let min_p_pass = r.pick(&[0.3, 0.5, 0.85, 0.95, 1.0]);
    let value_usd = r.pick(&[0.1, 1.0, 4.5, 30.0]);
    let pii = if r.chance(0.2) {
        PiiRule::Ignore
    } else {
        PiiRule::LocalOnly
    };
    let swap = SwapPolicy {
        allow: r.chance(0.5),
        lambda_wait_usd_per_s: r.unit() * 0.05,
        dwell: Duration::from_secs(600),
        hourly_cap: 1 + r.below(6) as u32,
    };
    StaticPolicy {
        local_only,
        allowed,
        preference,
        min_p_pass,
        value_usd,
        pii,
        swap,
    }
}

pub fn random_world(r: &mut Rng, now: Instant) -> World {
    let n = 1 + r.below(6) as u32;
    let backends: Vec<Backend> = (1..=n)
        .map(|id| {
            let local = r.chance(0.5);
            random_backend(r, id, local)
        })
        .collect();
    let policy = random_policy(r, n);
    let mut preds = Predictions::new();
    let mut prices = Prices::new();
    let mut entries = Vec::new();
    let mut queue = Vec::new();
    for b in &backends {
        if r.chance(0.85) {
            preds.insert(b.id, random_prediction(r));
        }
        let priced = if b.local { 0.3 } else { 0.9 };
        if r.chance(priced) {
            prices.insert(b.id, random_price(r));
        }
        if b.local {
            let swap_seconds = r.chance(0.8).then(|| r.unit() * 120.0);
            entries.push(ResidencyEntry {
                id: b.id,
                resident: r.chance(0.5),
                swap_seconds,
            });
        }
        queue.push(QueueEntry {
            id: b.id,
            depth: r.below(8) as u32,
        });
    }
    let swaps: Vec<Instant> = (0..r.below(4))
        .map(|_| now - Duration::from_secs(r.below(5_400)))
        .collect();
    let mut w = World {
        class: RequestClass::Standard,
        loc: Locality::any(),
        policy,
        backends,
        preds: Some(preds),
        prices,
        residency: Residency {
            entries,
            history: SwapHistory::from_swaps(swaps),
        },
        queues: Queues {
            entries: queue,
            hard_queued: r.below(6) as u32,
        },
        last: None,
        event: r.pick(&Event::ALL),
        pii: r.chance(0.2),
        now,
    };
    if r.chance(0.15) {
        w.preds = None;
    }
    if r.chance(0.2) {
        w.class = RequestClass::Privileged;
    }
    if r.chance(0.2) {
        w.loc = Locality::local();
    }
    if r.chance(0.6) {
        w.last = Some(BackendId(1 + r.below(u64::from(n) + 1) as u32));
    }
    w
}

/// An eligibility function: the real `eligible` or a mutant of it.
pub type Elig = fn(RequestClass, &Locality, &StaticPolicy, &[Backend]) -> EligibleSet;

pub fn decide_with(w: &World, elig: Elig) -> (EligibleSet, Decision) {
    let e = elig(w.class, &w.loc, &w.policy, &w.backends);
    let e = tighten_for_pii(&e, PiiSignal { detected: w.pii }, &w.policy);
    let d = choose(
        &e,
        w.preds.as_ref(),
        &w.prices,
        &w.residency,
        &w.queues,
        w.last,
        w.event,
        w.now,
    );
    (e, d)
}

/// Makes the world local-only through one of the three sources (policy flag, privileged class, header).
pub fn force_local_only(r: &mut Rng, w: &mut World) {
    match r.below(3) {
        0 => w.policy.local_only = true,
        1 => w.class = RequestClass::Privileged,
        _ => w.loc = Locality::local(),
    }
}
