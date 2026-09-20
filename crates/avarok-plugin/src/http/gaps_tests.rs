// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn sample(gaps: &[f64]) -> GapSample {
    let mut s = GapSample::default();
    for g in gaps {
        s.push(*g);
    }
    s
}

#[test]
fn no_gap_means_no_stats_not_a_perfectly_smooth_zero() {
    assert_eq!(GapSample::default().stats(), None);
    assert_eq!(sample(&[]).count(), 0);
}

/// The online moments must equal the closed-form ones on the retained
/// gaps: a Welford slip reports a wrong σ for every jitter number.
#[test]
fn online_moments_match_the_closed_form() {
    let gaps = [20.0, 22.0, 19.0, 21.0, 180.0, 20.5, 20.0, 21.5];
    let st = sample(&gaps).stats().unwrap();
    let n = gaps.len() as f64;
    let mean = gaps.iter().sum::<f64>() / n;
    let var = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / n;
    assert_eq!(st.count, 8);
    assert_eq!(st.retained, 8);
    assert!((st.mean_ms - mean).abs() < 1e-9);
    assert!((st.stddev_ms - var.sqrt()).abs() < 1e-9);
    assert_eq!(st.max_ms, 180.0);
    // Nearest-rank, the harness rule: p50 of 8 sorted is index 4.
    assert_eq!(st.p50_ms, 21.0);
    assert_eq!(st.p99_ms, 180.0);
}

/// THE STALL SIGNAL. One 180 ms hiccup in a run of ~20 ms steps barely
/// moves the mean but must dominate the max, the p99 and `stability`
/// (which RISES: lower is better, so a stall makes it worse).
#[test]
fn a_single_stall_is_a_tail_event_the_mean_hides_and_the_index_shows() {
    let smooth = sample(&[20.0; 100]).stats().unwrap();
    let mut gaps = vec![20.0; 100];
    gaps[50] = 180.0;
    let stalled = sample(&gaps).stats().unwrap();
    assert_eq!(smooth.stability(), Some(0.0));
    assert_eq!(smooth.cv(), Some(0.0));
    assert!((stalled.mean_ms - 21.6).abs() < 1e-9, "mean barely moves");
    assert_eq!(stalled.max_ms, 180.0);
    assert_eq!(stalled.p99_ms, 180.0);
    assert_eq!(stalled.stability(), Some(8.0));
    assert!(stalled.cv().unwrap() > 0.7);
}

/// Dimensionless: a model twice as slow with the same SHAPE of jitter
/// reports the same index, which is what lets rungs be compared.
#[test]
fn stability_is_scale_free() {
    let fast = sample(&[10.0, 10.0, 10.0, 12.0, 30.0]).stats().unwrap();
    let slow = sample(&[20.0, 20.0, 20.0, 24.0, 60.0]).stats().unwrap();
    assert!((fast.stability().unwrap() - slow.stability().unwrap()).abs() < 1e-12);
    assert!((fast.cv().unwrap() - slow.cv().unwrap()).abs() < 1e-12);
}

/// Percentiles stop at the cap; the online count, mean, σ and max do not,
/// and the truncation is visible in `retained`.
#[test]
fn the_retain_cap_bounds_memory_without_stopping_the_online_statistics() {
    let mut s = GapSample::default();
    for _ in 0..RETAIN_CAP {
        s.push(10.0);
    }
    for _ in 0..100 {
        s.push(1000.0);
    }
    let st = s.stats().unwrap();
    assert_eq!(st.count, RETAIN_CAP as u64 + 100);
    assert_eq!(st.retained, RETAIN_CAP);
    assert_eq!(st.max_ms, 1000.0, "the max is online, not from the buffer");
    assert!(st.mean_ms > 10.0, "the mean is online, not from the buffer");
    assert_eq!(st.p99_ms, 10.0, "percentiles cover only the retained gaps");
}

/// Pooling two responses must give the moments of the concatenation —
/// this is what makes a cell's number re-aggregatable from its requests.
#[test]
fn merge_equals_the_concatenation() {
    let a = [20.0, 21.0, 19.5, 40.0];
    let b = [22.0, 18.0, 20.0, 20.0, 95.0];
    let mut pooled = sample(&a);
    pooled.merge(&sample(&b));
    let all: Vec<f64> = a.iter().chain(b.iter()).copied().collect();
    let want = sample(&all).stats().unwrap();
    let got = pooled.stats().unwrap();
    assert_eq!(got.count, want.count);
    assert!((got.mean_ms - want.mean_ms).abs() < 1e-9);
    assert!((got.stddev_ms - want.stddev_ms).abs() < 1e-9);
    assert_eq!(got.max_ms, want.max_ms);
    assert_eq!((got.p50_ms, got.p90_ms, got.p99_ms), (want.p50_ms, want.p90_ms, want.p99_ms));
    // Merging an empty sample is a no-op, and merging INTO an empty one
    // yields the other side exactly.
    let mut untouched = sample(&a);
    untouched.merge(&GapSample::default());
    assert_eq!(untouched, sample(&a));
    let mut fresh = GapSample::default();
    fresh.merge(&sample(&b));
    assert_eq!(fresh.stats(), sample(&b).stats());
}

#[test]
fn metrics_emit_the_primitives_and_the_derived_ratios_under_the_prefix() {
    let mut m = BTreeMap::new();
    sample(&[20.0, 20.0, 20.0, 40.0])
        .stats()
        .unwrap()
        .metrics("c8_", &mut m);
    assert_eq!(m["c8_arrival_gap_count"], 4.0);
    assert_eq!(m["c8_arrival_gap_retained"], 4.0);
    assert_eq!(m["c8_arrival_gap_max_ms"], 40.0);
    assert_eq!(m["c8_arrival_gap_p50_ms"], 20.0);
    assert_eq!(m["c8_arrival_gap_p99_ms"], 40.0);
    assert_eq!(m["c8_stability"], 1.0);
    assert!(m.contains_key("c8_arrival_gap_cv"));
    assert!(m.contains_key("c8_arrival_gap_mean_ms"));
    assert!(m.contains_key("c8_arrival_gap_stddev_ms"));
    assert!(m.contains_key("c8_arrival_gap_p90_ms"));
    assert_eq!(m.len(), 10);
}

// ── ITL ───────────────────────────────────────────────────────────────

/// AIPerf's definition on both clocks: the numerator is the whole decode
/// window and the divisor is `n − 1`; under two tokens it is undefined
/// and must be `None`, never 0.
#[test]
fn itl_is_the_decode_window_per_token_and_undefined_below_two_tokens() {
    assert_eq!(itl_ms(900.0, 10), Some(100.0));
    assert_eq!(itl_ms(900.0, 2), Some(900.0));
    assert_eq!(itl_ms(900.0, 1), None);
    assert_eq!(itl_ms(900.0, 0), None);
    assert_eq!(itl_ms(0.0, 10), None);
    assert_eq!(itl_ms(-1.0, 10), None);
    assert_eq!(itl_ms(f64::NAN, 10), None);
}
