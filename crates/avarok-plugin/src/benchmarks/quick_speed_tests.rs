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
fn reconfiguring_clears_run_and_probe_state() {
    let mut b = configured();
    b.warmups_done = 1;
    b.samples.push(sample(7, 80.0, Some(5.0), Some(2.0)));
    b.probed = true;

    let values = ParamValues::defaults(&b.parameters());
    b.configure(&values).unwrap();

    assert_eq!((b.isl, b.osl, b.runs, b.warmup), (60, 128, 5, 1));
    assert_eq!(b.timeout, Duration::from_secs(300));
    assert_eq!(b.warmups_done, 0);
    assert!(b.samples.is_empty());
    assert!(!b.probed);
}

#[test]
fn a_fixture_isl_loads_the_committed_text_and_any_other_synthesizes() {
    use sha2::{Digest, Sha256};

    let expected = [
        (
            128usize,
            "f9d03ba9af6778941b923cb5b6871c0e9633aef25cebfb6a63b3082f30b21244",
        ),
        (
            512,
            "7d8a5fc10b77a710ea5957a1933f77f34bc21a85a7ce96d1959e6391d7a4af37",
        ),
        (
            1024,
            "3191c4db03b3a0b33a75e309e93846239c474298006ad9ae5afbaf9740daf6df",
        ),
        (
            4096,
            "60c5743a46ea72f5962f39a5aa1d73bbe54fac9bdf797d10bdb0bce1b058673f",
        ),
    ];
    for (isl, digest) in expected {
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
        assert_eq!(format!("{:x}", Sha256::digest(p.as_bytes())), digest);
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

#[test]
fn request_body_pins_the_measurement_instrument() {
    let b = configured();
    assert_eq!(
        b.request_body("fixture-model"),
        serde_json::json!({
            "model": "fixture-model",
            "stream": true,
            "max_tokens": 128,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": prompt_for(60),
            }],
        })
    );
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
fn run_sample_preserves_live_timing_evidence() {
    let outcome = http::ChatOutcome {
        prompt_tokens: 61,
        completion_tokens: 127,
        e2e_ms: 4_321.5,
        server_ttft_ms: Some(87.25),
        server_tps: Some(59.75),
        ..Default::default()
    };
    assert_eq!(
        RunSample::from_outcome(&outcome),
        RunSample {
            prompt_tokens: 61,
            completion_tokens: 127,
            e2e_ms: 4_321.5,
            server_ttft_ms: Some(87.25),
            server_tps: Some(59.75),
        }
    );
}

#[test]
fn averages_are_computed_from_the_server_timings() {
    let runs = [
        sample(128, 4000.0, Some(100.0), Some(50.0)),
        sample(128, 4000.0, Some(200.0), Some(70.0)),
    ];
    let avg = Averages::of(&runs);
    assert_eq!(avg.prompt_tokens, Some(70.0));
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
    assert_eq!(avg.e2e_ms, Some(4000.0));
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

#[test]
fn invalid_timings_are_omitted_instead_of_averaged() {
    let runs = [
        sample(100, f64::INFINITY, Some(-1.0), Some(0.0)),
        sample(100, -1.0, Some(f64::NAN), Some(-20.0)),
        sample(100, f64::NAN, Some(f64::INFINITY), Some(f64::INFINITY)),
    ];
    let avg = Averages::of(&runs);
    assert_eq!(avg.server_decode_tok_s, None);
    assert_eq!(avg.server_tpot_ms, None);
    assert_eq!(avg.server_ttft_ms, None);
    assert_eq!(avg.client_e2e_tok_s, None);
    assert_eq!(avg.e2e_ms, None);
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
