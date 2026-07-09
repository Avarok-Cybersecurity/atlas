// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn sig_text_only(text: &str) -> Signature {
    Signature::build(text, std::iter::empty())
}

fn sig_with_tool(text: &str, name: &str, args: &str) -> Signature {
    Signature::build(text, std::iter::once((name, args)))
}

#[test]
fn empty_recent_returns_none() {
    assert_eq!(detect(&[]), LoopState::None);
    assert_eq!(
        detect(&[sig_text_only("hello world how are you doing today")]),
        LoopState::None
    );
}

#[test]
fn three_distinct_messages_no_loop() {
    let recent = vec![
        sig_text_only("the quick brown fox jumps over the lazy dog"),
        sig_text_only("once upon a time there was a small village"),
        sig_text_only("rust is a systems programming language designed for safety"),
    ];
    assert_eq!(detect(&recent), LoopState::None);
}

#[test]
fn three_identical_intros_fire_loop() {
    // Cross-turn prose loop — failure pattern from dump 2026-04-25
    // seq=35 msgs[11/15/17/19].
    let intro = "I will create a proper echo server for you. Let me write \
                     a clean implementation with axum and serde.";
    let recent = vec![
        sig_text_only(intro),
        sig_text_only(intro),
        sig_text_only(intro),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. } | LoopState::Hint { .. }),
        "got {v:?}"
    );
}

#[test]
fn identical_tool_args_fire_loop() {
    // The classic 4× cargo init failure mode (claude-export.txt).
    let cmd = r#"{"command":"mkdir -p /tmp/axum-test-5 && cd /tmp/axum-test-5 && cargo init --name axum_echo_server"}"#;
    let recent = vec![
        sig_with_tool("", "Bash", cmd),
        sig_with_tool("", "Bash", cmd),
        sig_with_tool("", "Bash", cmd),
        sig_with_tool("", "Bash", cmd),
    ];
    let v = detect(&recent);
    assert!(matches!(v, LoopState::Suppress { .. }), "got {v:?}");
}

#[test]
fn different_tool_args_with_same_name_do_not_fire() {
    let recent = vec![
        sig_with_tool("", "Bash", r#"{"command":"ls /a"}"#),
        sig_with_tool("", "Bash", r#"{"command":"echo hi"}"#),
        sig_with_tool("", "Bash", r#"{"command":"pwd"}"#),
    ];
    // Token overlap is "Bash command" — high enough to fire on
    // small sets. Our threshold should resist this; if it
    // fails, raise SHINGLE_ORDER.
    let v = detect(&recent);
    assert_eq!(v, LoopState::None, "got {v:?}");
}

#[test]
fn slightly_varied_intros_still_fire() {
    // The model often paraphrases its intro slightly — must
    // catch this too.
    let a = "I'll create a proper echo server using axum and serde with full tests.";
    let b = "I'll create a proper echo server using axum and serde with passing tests.";
    let c = "I'll create a proper echo server using axum and serde, including tests.";
    let recent = vec![sig_text_only(a), sig_text_only(b), sig_text_only(c)];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Hint { .. } | LoopState::Suppress { .. }),
        "near-identical paraphrases must trigger detection: {v:?}"
    );
}

#[test]
fn moderate_similarity_3_turns_now_suppress() {
    // Regression for the threshold recalibrationibration (2026-04-25).
    // Three turns where the assistant repeats the same core
    // sentence with a small trailing variation — empirically this
    // is what the dump seq=54 prose loop looked like (Agent 2
    // measured 0.73 / 0.62 / 0.52 Jaccard on the actual messages).
    //
    // Pre-recalibration (HIGH=0.80, MODERATE=0.55): returned `Hint` —
    // model kept looping.
    // Post-recalibration (HIGH=0.65, MODERATE=0.50): `Suppress` fires.
    let common = "Let me create the necessary files for the Axum server with an echo endpoint and passing tests in the project directory now";
    let recent = vec![
        sig_text_only(&format!("{common} immediately")),
        sig_text_only(&format!("{common} carefully")),
        sig_text_only(&format!("{common} step by step")),
        sig_text_only(&format!("{common} as planned")),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. }),
        "post-recalibration: 4 turns sharing the same long prose stem must Suppress: {v:?}"
    );
}

#[test]
fn moderate_band_with_065_max_suppress() {
    // After lowering HIGH_SIMILARITY to 0.65, a 3-turn run whose
    // max pair similarity is just above 0.65 must escalate to
    // Suppress (was Hint at the old 0.80 bar). The trick: build
    // signatures whose Jaccard lands in the [0.65, 0.80) band.
    // Three identical sentences w/ one different trailing word
    // produces ~0.92 Jaccard at SHINGLE_ORDER=4 — solidly above.
    let s = "the quick brown fox jumps over the lazy dog and chases the mouse around the garden";
    let recent = vec![
        sig_text_only(&format!("{s} alpha")),
        sig_text_only(&format!("{s} bravo")),
        sig_text_only(&format!("{s} charlie")),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. }),
        "highly-similar 3-turn run must Suppress post-recalibration: {v:?}"
    );
}

#[test]
fn empty_signatures_are_skipped_not_counted() {
    // A 4-byte heartbeat in the middle should not break a real
    // 3-turn loop on either side of it.
    let intro =
        "I will create a proper echo server for you. Let me write a clean implementation now.";
    let recent = vec![
        sig_text_only(intro),
        sig_text_only(""), // empty — skipped by detector
        sig_text_only(intro),
        sig_text_only(intro),
    ];
    let v = detect(&recent);
    assert!(
        matches!(v, LoopState::Suppress { .. } | LoopState::Hint { .. }),
        "got {v:?}"
    );
}

#[test]
fn one_off_repeat_does_not_trigger_suppress() {
    // Two identical messages is normal (e.g., the model retries
    // a transient failure). Must NOT escalate to Suppress on two
    // turns alone.
    let intro =
        "I will create a proper echo server for you. Let me write a clean implementation now.";
    let recent = vec![sig_text_only(intro), sig_text_only(intro)];
    let v = detect(&recent);
    assert_eq!(v, LoopState::None, "two-turn repeat is not yet a loop");
}

// ─── P1-5 (2026-07-09): exact-match failing-call fast path ──────────

#[test]
fn p1_5_three_identical_failing_short_calls_detected_without_suppress() {
    // 45k-collapse shape: 3 byte-identical FAILING calls whose
    // ~3-token unit sits below MIN_CHANNEL_TOKENS — the legacy
    // detector is blind (empty signatures ⇒ None ⇒ no Suppress), and
    // the fast path must flag the loop candidate instead. The fast
    // path never drives suppress_tool_call: the orchestrator only
    // bumps tool_call_repeat_count (soft bias decay), leaving
    // <tool_call> — the escape action — available.
    let sig = sig_with_tool("", "write", r#"{"p":""}"#);
    assert!(
        sig.is_empty(),
        "short call must be below MIN_CHANNEL_TOKENS for this test to be meaningful"
    );
    let sigs = vec![sig.clone(), sig.clone(), sig];
    assert_eq!(
        detect(&sigs),
        LoopState::None,
        "legacy detect() must stay blind (⇒ suppress NOT set via Suppress verdict)"
    );

    let turn = CallOutcome {
        call_unit: Some("write\u{1f}{\"p\":\"\"}".to_string()),
        failing: true,
        result_unit: None,
    };
    let turns = vec![turn.clone(), turn.clone(), turn];
    assert_eq!(
        detect_exact_failing_repeat(&turns),
        Some(3),
        "fast path must fire on 3 byte-identical failing calls"
    );
}

#[test]
fn p1_5_three_identical_succeeding_short_calls_unchanged_legacy() {
    // Same 3 identical short calls but the results SUCCEEDED — the
    // fast path must NOT fire without error-shaped results, and the
    // legacy path stays unchanged (no detection below
    // MIN_CHANNEL_TOKENS).
    let turn = CallOutcome {
        call_unit: Some("write\u{1f}{\"p\":\"\"}".to_string()),
        failing: false,
        result_unit: None,
    };
    let turns = vec![turn.clone(), turn.clone(), turn];
    assert_eq!(detect_exact_failing_repeat(&turns), None);

    let sig = sig_with_tool("", "write", r#"{"p":""}"#);
    assert_eq!(
        detect(&[sig.clone(), sig.clone(), sig]),
        LoopState::None,
        "legacy behavior unchanged for short succeeding repeats"
    );
}

#[test]
fn p1_5_two_identical_failing_calls_not_enough() {
    let turn = CallOutcome {
        call_unit: Some("x\u{1f}{}".to_string()),
        failing: true,
        result_unit: None,
    };
    assert_eq!(detect_exact_failing_repeat(&[turn.clone(), turn]), None);
}

#[test]
fn p1_5_differing_units_break_the_run() {
    let a = CallOutcome {
        call_unit: Some("write\u{1f}{\"p\":\"a\"}".to_string()),
        failing: true,
        result_unit: None,
    };
    let b = CallOutcome {
        call_unit: Some("write\u{1f}{\"p\":\"b\"}".to_string()),
        failing: true,
        result_unit: None,
    };
    assert_eq!(detect_exact_failing_repeat(&[a.clone(), a, b]), None);
}

#[test]
fn p1_5_no_tool_call_turn_breaks_the_run() {
    let call = CallOutcome {
        call_unit: Some("x\u{1f}{}".to_string()),
        failing: true,
        result_unit: None,
    };
    let prose = CallOutcome {
        call_unit: None,
        failing: false,
        result_unit: None,
    };
    assert_eq!(
        detect_exact_failing_repeat(&[call.clone(), prose, call]),
        None
    );
}

#[test]
fn p1_5_recent_calls_all_failing_gate() {
    let fail = CallOutcome {
        call_unit: Some("x\u{1f}{}".to_string()),
        failing: true,
        result_unit: None,
    };
    let ok = CallOutcome {
        call_unit: Some("x\u{1f}{}".to_string()),
        failing: false,
        result_unit: None,
    };
    // All three failing ⇒ Suppress hard-mask must be skipped.
    assert!(recent_calls_all_failing(
        &[fail.clone(), fail.clone(), fail.clone()],
        3
    ));
    // Any succeeding call in the window ⇒ legacy Suppress applies.
    assert!(!recent_calls_all_failing(
        &[fail.clone(), ok, fail.clone()],
        3
    ));
    // Fewer outcomes than requested ⇒ conservative false.
    assert!(!recent_calls_all_failing(&[fail], 3));
}

#[test]
fn signature_below_min_tokens_is_empty() {
    // Even if some shingles could be formed at order < SHINGLE_ORDER,
    // we explicitly zero short channels so trivial echoes don't
    // produce noise.
    let s = Signature::build("yes", std::iter::empty());
    assert!(s.is_empty(), "3-token text must yield empty signature");
}

// ── P1-5b (2026-07-09): result-progress gate ──

fn outcome(unit: &str, failing: bool, result: &str) -> CallOutcome {
    CallOutcome {
        call_unit: Some(unit.into()),
        failing,
        result_unit: Some(result.into()),
    }
}

#[test]
fn progressing_cycle_detected_when_results_differ() {
    // cargo check-fix-check: similar calls, DIFFERENT error lists.
    let outcomes = vec![
        outcome(
            "bash\u{1f}cargo check",
            false,
            "error[E0308]: mismatched types in transactions.rs line 41",
        ),
        outcome(
            "bash\u{1f}cargo check",
            false,
            "error[E0433]: unresolved import sqlx::SqlitePool in db.rs",
        ),
        outcome(
            "bash\u{1f}cargo check",
            false,
            "error[E0599]: no method named fetch_all found for Pool",
        ),
    ];
    assert!(
        recent_results_progressing(&outcomes, 3),
        "differing results round-to-round = progress; must not hard-mask"
    );
}

#[test]
fn true_loop_not_progressing_when_results_identical() {
    let outcomes = vec![
        outcome("write\u{1f}{\"f\":1}", false, "Wrote file successfully."),
        outcome("write\u{1f}{\"f\":1}", false, "Wrote file successfully."),
        outcome("write\u{1f}{\"f\":1}", false, "Wrote file successfully."),
    ];
    assert!(
        !recent_results_progressing(&outcomes, 3),
        "identical results = true loop; legacy Suppress must apply"
    );
}

#[test]
fn missing_results_are_conservatively_not_progressing() {
    let outcomes = vec![
        CallOutcome {
            call_unit: Some("x".into()),
            failing: false,
            result_unit: None,
        },
        outcome("x", false, "a result"),
    ];
    assert!(!recent_results_progressing(&outcomes, 2));
}
