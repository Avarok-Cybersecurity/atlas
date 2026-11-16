// SPDX-License-Identifier: AGPL-3.0-only

//! Oracles for the confidence summary: log-softmax values worked out by hand
//! from fixed logits, the accumulator's limits, the request plan, the
//! extraction release and the metric text.

use super::*;

fn near(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}

fn rec(token_id: u32, top: &[(u32, f32)]) -> TokenLogprobs {
    TokenLogprobs {
        token_id,
        logprob: top.first().map_or(f32::NEG_INFINITY, |t| t.1),
        top: top.to_vec(),
    }
}

// softmax([2, 1, 0]) = e^x / (e^2 + e^1 + 1) with the sum 11.107337927:
//   p1 = 7.389056099 / 11.107337927 = 0.66524
//   p2 = 2.718281828 / 11.107337927 = 0.24473
//   margin = 0.42051
const P1: f32 = 0.66524;
const MARGIN: f32 = 0.42051;

#[test]
fn one_position_matches_the_hand_computed_softmax() {
    let mut acc = ConfAcc::new(8);
    assert!(acc.observe_logits(&[2.0, 1.0, 0.0]));
    let s = acc.summary().expect("one token counted");
    assert_eq!(s.tokens, 1);
    assert!(near(s.mean_top1, P1, 1e-4), "{s:?}");
    assert!(near(s.min_margin, MARGIN, 1e-4), "{s:?}");
    assert_eq!(s.low_frac, 0.0, "0.665 is not below 0.5");
}

#[test]
fn uniform_logits_are_a_zero_margin_low_confidence_token() {
    let mut acc = ConfAcc::new(8);
    assert!(acc.observe_logits(&[0.0, 0.0, 0.0, 0.0]));
    let s = acc.summary().unwrap();
    assert!(near(s.mean_top1, 0.25, 1e-5), "{s:?}");
    assert!(near(s.min_margin, 0.0, 1e-5), "{s:?}");
    assert_eq!(s.low_frac, 1.0);
}

#[test]
fn two_positions_combine_mean_min_and_fraction() {
    let mut acc = ConfAcc::new(8);
    acc.observe_logits(&[2.0, 1.0, 0.0]);
    acc.observe_logits(&[0.0, 0.0, 0.0, 0.0]);
    let s = acc.summary().unwrap();
    assert_eq!(s.tokens, 2);
    assert!(near(s.mean_top1, (P1 + 0.25) / 2.0, 1e-4), "{s:?}");
    assert!(
        near(s.min_margin, 0.0, 1e-5),
        "the smaller margin wins: {s:?}"
    );
    assert!(near(s.low_frac, 0.5, 1e-6), "{s:?}");
}

#[test]
fn the_accumulator_stops_at_its_limit() {
    let mut acc = ConfAcc::new(2);
    assert!(acc.observe_logits(&[2.0, 1.0, 0.0]));
    assert!(acc.observe_logits(&[2.0, 1.0, 0.0]));
    assert!(acc.is_full());
    // A third, very different position must not move the summary.
    assert!(!acc.observe_logits(&[0.0, 0.0, 0.0, 0.0]));
    let s = acc.summary().unwrap();
    assert_eq!(s.tokens, 2);
    assert!(near(s.mean_top1, P1, 1e-4), "{s:?}");
    assert_eq!(s.low_frac, 0.0);
}

#[test]
fn positions_without_data_are_not_counted() {
    let mut acc = ConfAcc::new(4);
    assert!(!acc.observe_top(&[]), "empty list");
    assert!(!acc.observe_top(&[(1, f32::NAN)]), "NaN logprob");
    assert!(acc.summary().is_none(), "nothing counted, no summary");
    // A one-entry list (k = 1) counts with p2 = 0, so the margin is p1.
    assert!(acc.observe_top(&[(5, 0.8f32.ln())]));
    let s = acc.summary().unwrap();
    assert!(near(s.mean_top1, 0.8, 1e-5), "{s:?}");
    assert!(near(s.min_margin, 0.8, 1e-5), "{s:?}");
}

#[test]
fn logits_and_logprob_records_agree() {
    // The scheduler hands over `logprob_of`'s top-k; the oracle path derives
    // the same list from raw logits. Both must give one summary.
    let logits = [3.0f32, 1.5, -1.0, 0.25];
    let (_, top) = spark_model::traits::logprob_of(&logits, 0, 2);
    let mut a = ConfAcc::new(4);
    let mut b = ConfAcc::new(4);
    a.observe_logits(&logits);
    b.observe_lp(&rec(0, &top));
    assert_eq!(a.summary(), b.summary());
}

#[test]
fn summarize_reads_the_first_limit_records_only() {
    let strong = rec(1, &[(1, 0.9f32.ln()), (2, 0.05f32.ln())]);
    let weak = rec(2, &[(2, 0.3f32.ln()), (3, 0.29f32.ln())]);
    let lps = vec![strong.clone(), strong.clone(), weak.clone(), weak];
    let s = summarize(&lps, 2).expect("two records");
    assert_eq!(s.tokens, 2);
    assert!(near(s.mean_top1, 0.9, 1e-5), "{s:?}");
    assert!(near(s.min_margin, 0.85, 1e-5), "{s:?}");
    assert_eq!(s.low_frac, 0.0);
    assert!(summarize(&[], 8).is_none());
}

#[test]
fn a_repeated_record_for_a_token_without_one_is_skipped() {
    let mut acc = ConfAcc::new(8);
    let lp = rec(7, &[(7, 0.9f32.ln()), (8, 0.05f32.ln())]);
    // The streamed token matches its record: counted.
    observe_stream(Some(&mut acc), 7, &lp);
    // The scheduler resent the last record for a different token: skipped.
    observe_stream(Some(&mut acc), 9, &lp);
    // No accumulator (summary not wanted): nothing happens.
    observe_stream(None, 7, &lp);
    assert_eq!(acc.summary().unwrap().tokens, 1);
}

#[test]
fn k_parses_with_a_default_and_rejects_nonsense() {
    assert_eq!(parse_k(None), Ok(DEFAULT_K));
    assert_eq!(parse_k(Some("8")), Ok(8));
    assert_eq!(parse_k(Some(" 16 ")), Ok(16));
    assert_eq!(parse_k(Some("4096")), Ok(MAX_K));
    for bad in ["0", "4097", "-1", "abc", "", "1.5"] {
        let e = parse_k(Some(bad)).expect_err(bad);
        assert!(e.contains(ENV_K), "{e}");
    }
}

#[test]
fn the_plan_leaves_an_unwanted_request_untouched() {
    for client in [None, Some(0), Some(5), Some(20)] {
        let p = plan(false, false, client);
        assert!(!p.collect && !p.report);
        assert_eq!(p.sched_top_logprobs, client, "byte-identical hand-off");
        assert_eq!(p.client_top_logprobs, client);
    }
}

#[test]
fn the_plan_reports_only_when_asked_and_collects_when_the_row_is_on() {
    // Row on, request silent: collect for the metrics, no usage field.
    let p = plan(true, false, None);
    assert!(p.collect && !p.report);
    assert_eq!(p.sched_top_logprobs, Some(INTERNAL_TOP_LOGPROBS));
    // Row off, request asks: collect and report.
    let p = plan(false, true, None);
    assert!(p.collect && p.report);
    assert_eq!(p.sched_top_logprobs, Some(INTERNAL_TOP_LOGPROBS));
}

#[test]
fn the_plan_keeps_a_clients_logprobs_and_lifts_them_to_top_two() {
    let p = plan(false, true, Some(0));
    assert_eq!(
        p.sched_top_logprobs,
        Some(2),
        "the margin needs the top two"
    );
    assert_eq!(p.client_top_logprobs, Some(0), "the client still gets none");
    let p = plan(false, true, Some(7));
    assert_eq!(p.sched_top_logprobs, Some(7));
}

#[test]
fn the_internal_value_is_outside_the_client_range() {
    // `resolve_top_logprobs` clamps a client to 20; the sentinel must never
    // collide with a value a client can send.
    let sentinel = std::hint::black_box(INTERNAL_TOP_LOGPROBS);
    assert!(sentinel > 20);
}

#[test]
fn the_extraction_is_released_after_k_and_only_when_internal() {
    let k = conf_k() as usize;
    // Summary-only extraction: kept until K records, then given back.
    let mut t = Some(INTERNAL_TOP_LOGPROBS);
    release_internal_logprobs(&mut t, k - 1);
    assert_eq!(t, Some(INTERNAL_TOP_LOGPROBS));
    release_internal_logprobs(&mut t, k);
    assert_eq!(t, None);
    // A client's own request is never touched, however long the response.
    let mut client = Some(5);
    release_internal_logprobs(&mut client, k * 10);
    assert_eq!(client, Some(5));
    let mut none: Option<u8> = None;
    release_internal_logprobs(&mut none, k * 10);
    assert_eq!(none, None);
}

#[test]
fn recorded_summaries_show_up_in_the_metric_text() {
    let before = SUMMARIES.load(Ordering::Relaxed);
    record(&ConfSummary {
        tokens: 4,
        mean_top1: 0.75,
        min_margin: 0.5,
        low_frac: 0.25,
    });
    assert!(SUMMARIES.load(Ordering::Relaxed) > before);
    let mut out = String::new();
    render_prometheus(&mut out);
    for name in [
        "atlas_conf_mean_top1",
        "atlas_conf_min_margin",
        "atlas_conf_summaries_total",
    ] {
        assert!(out.contains(&format!("# HELP {name} ")), "{name}: {out}");
    }
    assert!(out.contains("# TYPE atlas_conf_mean_top1 summary"), "{out}");
    assert!(
        out.contains("# TYPE atlas_conf_summaries_total counter"),
        "{out}"
    );
    let count = out
        .lines()
        .find_map(|l| l.strip_prefix("atlas_conf_mean_top1_count "))
        .expect("count line")
        .parse::<u64>()
        .unwrap();
    assert!(count >= 1, "{out}");
}

#[test]
fn microsecond_accumulation_rounds_and_clamps() {
    assert_eq!(micros(0.5), 500_000);
    assert_eq!(micros(1.0), 1_000_000);
    assert_eq!(
        micros(-0.2),
        0,
        "a negative margin cannot underflow the sum"
    );
}
