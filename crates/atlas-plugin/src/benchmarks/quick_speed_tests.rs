// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn configured() -> QuickSpeed {
    let mut b = QuickSpeed::default();
    let v = ParamValues::defaults(&b.parameters());
    b.configure(&v).unwrap();
    b
}

#[test]
fn defaults_match_the_python_probe() {
    let b = QuickSpeed::default();
    let v = ParamValues::defaults(&b.parameters());
    assert_eq!(v.int("isl").unwrap(), 60);
    assert_eq!(v.int("osl").unwrap(), 128);
    assert_eq!(v.int("runs").unwrap(), 5);
    assert_eq!(v.int("warmup").unwrap(), 1);
    assert_eq!(v.int("request_timeout_s").unwrap(), 300);
}

#[test]
fn a_fixture_isl_loads_the_committed_text_and_any_other_synthesizes() {
    for isl in [128usize, 512, 1024, 4096] {
        let p = prompt_for(isl);
        assert!(
            p.ends_with(COUNT_SUFFIX),
            "{isl}: fixture prompt lost the forcing suffix"
        );
        // Fixtures are natural text, not the filler corpus.
        assert!(
            !p.starts_with("The quick brown fox"),
            "{isl}: expected fixture text, got synthesized filler"
        );
    }
    // Any non-fixture size is exactly the shared synthesizer's output — one
    // corpus, one rule, no drift from the concurrency sweep's prompts.
    assert_eq!(
        prompt_for(60),
        stats::make_prompt(60, PromptMode::Count, "")
    );
    assert_eq!(
        prompt_for(200),
        stats::make_prompt(200, PromptMode::Count, "")
    );
    // The same ISL always yields the same bytes — the warm-path premise.
    assert_eq!(prompt_for(128), prompt_for(128));
}

fn sample(tokens: usize, e2e_ms: f64, ttft: Option<f64>, tps: Option<f64>) -> RunSample {
    RunSample {
        prompt_tokens: 70,
        completion_tokens: tokens,
        e2e_ms,
        server_ttft_ms: ttft,
        server_tps: tps,
    }
}

#[test]
fn averages_are_computed_from_the_server_timings() {
    let runs = [
        sample(128, 4000.0, Some(100.0), Some(50.0)),
        sample(128, 4000.0, Some(200.0), Some(70.0)),
    ];
    let avg = Averages::of(&runs);
    assert_eq!(avg.server_decode_tok_s, Some(60.0));
    assert_eq!(avg.server_ttft_ms, Some(150.0));
    // TPOT is derived from the server decode rate, per run then averaged:
    // (1000/50 + 1000/70) / 2.
    let want_tpot = (1000.0 / 50.0 + 1000.0 / 70.0) / 2.0;
    assert!((avg.server_tpot_ms.unwrap() - want_tpot).abs() < 1e-9);
    // Client E2E rate includes prefill: 128 tok / 4 s = 32 tok/s — visibly
    // lower than the 60 tok/s decode rate, which is the point of the label.
    assert_eq!(avg.client_e2e_tok_s, Some(32.0));
    assert_eq!(avg.output_tokens, Some(128.0));
}

/// ★ The defect the port exists to fix: no server timing ⇒ no TPOT, never a
/// client-clock substitute. The buffered-read TPOT the Python printed implied
/// 101 tok/s on hardware that cannot exceed ~60.
#[test]
fn without_server_timings_no_decode_rate_or_tpot_is_fabricated() {
    let runs = [sample(128, 4000.0, None, None)];
    let avg = Averages::of(&runs);
    assert_eq!(avg.server_decode_tok_s, None);
    assert_eq!(avg.server_tpot_ms, None);
    assert_eq!(avg.server_ttft_ms, None);
    // The client-side numbers survive — they are honest, just labelled.
    assert_eq!(avg.client_e2e_tok_s, Some(32.0));

    // And the metrics map omits the absent keys rather than writing 0.0.
    let mut b = configured();
    b.samples = runs.to_vec();
    let m = b.metrics(&avg);
    assert!(!m.contains_key("server_decode_tok_s"));
    assert!(!m.contains_key("server_tpot_ms"));
    assert_eq!(m["client_e2e_tok_s"], 32.0);
    // A mixed set averages only the runs that reported.
    let mixed = [
        sample(100, 2000.0, Some(80.0), Some(40.0)),
        sample(100, 2000.0, None, None),
    ];
    assert_eq!(Averages::of(&mixed).server_decode_tok_s, Some(40.0));
}

/// EOS before the OSL cap is a data point, not an error: the arithmetic uses
/// the tokens actually produced (the recorded 49-vs-128 case), and the summary
/// names the cap so the shortfall is visible.
#[test]
fn eos_before_the_osl_cap_reports_actual_tokens_against_the_cap() {
    let runs = [sample(49, 1000.0, Some(90.0), Some(60.0))];
    let avg = Averages::of(&runs);
    assert_eq!(avg.output_tokens, Some(49.0));
    assert_eq!(avg.client_e2e_tok_s, Some(49.0));

    let mut b = configured();
    b.samples = runs.to_vec();
    let stats = b.summary(&avg);
    let out = stats
        .iter()
        .find(|s| s.label == "Output tok")
        .expect("output stat");
    assert_eq!(out.value, "49 / 128 cap");
}

#[test]
fn the_headline_stat_is_the_server_decode_rate_and_says_so() {
    let runs = [sample(128, 4000.0, Some(100.0), Some(60.0))];
    let avg = Averages::of(&runs);
    let b = configured();
    let stats = b.summary(&avg);
    assert_eq!(stats[0].label, "Decode tok/s (server)");
    assert_eq!(stats[0].value, "60.0");
    // The client rate cannot be quoted as the decode rate by accident.
    assert!(stats[1].label.contains("client"), "{}", stats[1].label);
    assert!(
        stats[1].label.contains("incl. prefill"),
        "{}",
        stats[1].label
    );
}

#[test]
fn zero_tokens_yields_no_rate_rather_than_zero_or_a_panic() {
    let s = sample(0, 1000.0, None, None);
    assert_eq!(s.client_e2e_tok_s(), None);
    assert_eq!(s.server_tpot_ms(), None);
    assert_eq!(Averages::of(&[]), Averages::default());
}

/// The trap this port was warned about: registered, but NOT a required PR
/// gate. `coverage_map_tests` forces the NOT_REQUIRED excusal; this pins the
/// descriptor's own gate-free shape.
#[test]
fn registered_as_a_measurement_tool_not_a_gate() {
    let d = crate::registry::find("quick-speed-bench").expect("registered");
    assert!(d.intended_for.is_none());
    assert!(d.threshold_params.is_empty());
    assert!(!d.needs_confirmation);
    assert!(
        !crate::gate::REQUIRED_GATES.contains(&"quick-speed-bench"),
        "quick-speed-bench must never be a required PR gate"
    );
}
