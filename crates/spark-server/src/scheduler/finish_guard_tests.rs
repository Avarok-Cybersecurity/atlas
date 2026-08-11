// SPDX-License-Identifier: AGPL-3.0-only

//! The scheduler-side half of the cancel-guard invariant.
//!
//! **A control-token hard stop must name its guard.** When the scheduler ends a
//! turn because the model emitted a token it must never emit, that is a SERVER
//! cut. If the site does not set `a.guard_stop`, `derive_finish_reason` skips
//! its guard rung, sees budget remaining, and wires `"stop"` — telling the
//! client the model finished.
//!
//! ★ Why this file exists separately from
//! `api/chat_stream/cancel_guard_tests.rs`: that test scans `cancel_flag`
//! stores under `api/chat_stream/`, and it was **green** while the
//! `<tool_response>` hard stop sat bare — because a scheduler guard never
//! touches `cancel_flag` at all. It sets `finished` directly. Two layers, two
//! tests; neither covers the other's sites.
//!
//! The cost was concrete: the bare hard stop mislabelled a turn, the agentic
//! harness's `was_cut_off()` (which nudges only on `"length"`) stayed silent,
//! and a benchmark run ended at 9 turns instead of 32.
//!
//! # Scope, deliberately narrow
//!
//! This asserts the **hard-stop family** only — the class that shipped broken
//! twice (`<tool_response>` and its MTP twin). It does NOT try to classify every
//! `finished = true` in the scheduler. An earlier draft did, by matching words
//! like "watchdog" in a lookback window, and it flagged nine sites of which most
//! were comment prose or intentional. A structural test that fires on comments
//! is worse than no test, because it trains people to ignore it.
//!
//! Sites outside this family that may or may not want naming are recorded as
//! open questions in the PR, not silently changed.

/// Scheduler files carrying control-token hard stops.
const HARD_STOP_FILES: &[(&str, &str)] = &[
    (
        "decode_logits_step.rs",
        include_str!("decode_logits_step.rs"),
    ),
    ("emit_step.rs", include_str!("emit_step.rs")),
];

/// The log line every hard-stop site emits. This is the anchor: it is emitted
/// at the site, in the same statement group as the `finished` assignment.
const HARD_STOP_ANCHOR: &str = "hard-stop fired";

/// Proof the guard named itself.
const NAMED: &str = "guard_stop = Some(";

/// Hard stops that intentionally wire `"stop"`, with the reason.
///
/// `<|im_start|>` is **registered in `eos_tokens`** at startup
/// (`tokenizer_runtime.rs::im_start_id`), so the site pushes the token and
/// `derive_finish_reason`'s `is_eos` check succeeds — `"stop"` is correct and
/// deliberate. Naming a guard there would flip a correct `"stop"` to `"length"`
/// and reintroduce the exact mislabel that motivated the push
/// (`"Done: 13 tokens (length) despite max_tokens=8192"`).
///
/// `<tool_response>` is NOT eos-registered — an instrumented leg recorded
/// `is_eos=false` for it — which is precisely why it fell through to `"stop"`
/// by accident rather than by design. That asymmetry is the whole point of
/// this exemption list: it is not "which stops we bothered to fix".
/// ★ Matched against the ANCHOR LINE ONLY — the `tracing!` message emitted at
/// the site, which names which hard stop fired. Not against the surrounding
/// window: an earlier draft searched the whole window and was immediately
/// fooled by a comment in `decode_logits_step.rs` that merely *mentions*
/// `<|im_start|>` twelve lines above the `<tool_response>` site, silently
/// exempting the very site this test exists to catch. Prose is not evidence.
const INTENTIONAL_STOP: &[&str] = &["<|im_start|>"];

/// Lookback/lookahead around the anchor.
const WINDOW: usize = 12;

fn scan(src: &str) -> (usize, Vec<usize>) {
    let lines: Vec<&str> = src.lines().collect();
    let mut sites = 0usize;
    let mut bare = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !line.contains(HARD_STOP_ANCHOR) {
            continue;
        }
        let from = i.saturating_sub(WINDOW);
        let window = lines[from..=i].join("\n");
        if !window.contains("finished = true") {
            continue; // an anchor with no finish next to it is not a cut site
        }
        if INTENTIONAL_STOP.iter().any(|t| line.contains(t)) {
            continue;
        }
        sites += 1;
        if !window.contains(NAMED) {
            bare.push(i + 1);
        }
    }
    (sites, bare)
}

#[test]
fn every_control_token_hard_stop_names_its_guard() {
    let mut total = 0usize;
    let mut bare: Vec<String> = Vec::new();
    for (name, src) in HARD_STOP_FILES {
        let (n, lines) = scan(src);
        total += n;
        bare.extend(lines.iter().map(|l| format!("{name}:{l}")));
    }

    // Floor: the `<tool_response>` stop and its MTP twin. If a rename or move
    // drops the scan to zero this must fail loudly rather than report a green
    // it never earned.
    assert!(
        total >= 2,
        "found only {total} non-exempt hard-stop sites — the scan stopped \
         matching. Fix the detection before trusting a green here."
    );

    assert!(
        bare.is_empty(),
        "a control-token hard stop ended a turn without naming its guard at: \
         {bare:?}\n\
         An unnamed guard cut wires finish_reason \"stop\", claiming the model \
         finished when the server truncated it. Set \
         `a.guard_stop = Some(GUARD_STOP_*)` at the site, or add the token to \
         INTENTIONAL_STOP with the reason it is genuinely a natural end."
    );
}

#[test]
fn the_scan_flags_a_bare_stop_and_clears_a_named_one() {
    // NEGATIVE half: prove the detector fires. Without this the test above
    // passes for two reasons — the invariant holding, or the scan matching
    // nothing — and only one is good news.
    let bare = "\
        if tok == trs {\n\
        \x20   a.output_tokens.push(tok);\n\
        \x20   a.finished = true;\n\
        \x20   tracing::debug!(\"<tool_response> hard-stop fired (id={trs})\");\n\
        }\n";
    let (n, flagged) = scan(bare);
    assert_eq!(n, 1, "the site should be counted");
    assert_eq!(flagged, vec![4], "a bare hard stop must be flagged");

    // POSITIVE half: naming it clears the finding.
    let named = bare.replace(
        "    a.finished = true;",
        "    a.finished = true;\n    a.guard_stop = Some(GUARD_STOP_TOOL_RESPONSE);",
    );
    let (n, flagged) = scan(&named);
    assert_eq!(n, 1);
    assert!(flagged.is_empty(), "a named hard stop must not be flagged");
}

/// Channel 2 — the finish-site DECISION LEDGER.
///
/// Channel 1 (`scan` above) anchors on the `hard-stop fired` LOG LINE:
/// detection by confession. A cut site that never logs is invisible to it —
/// and that is exactly how the think-skip watchdog (50 consecutive stray
/// `</think>` → `finished = true`, no log, no guard) shipped bare: a
/// control-token hard stop squarely inside this file's declared family,
/// mute, therefore never scanned, wiring `"stop"` for a server cut and
/// silently ending agentic runs.
///
/// This channel anchors on the STATE MUTATION instead: every code-line
/// `finished = true` is a site, and the ledger pins, per file, how many
/// exist and how many name a guard within ±`WINDOW` lines. It deliberately
/// does NOT classify the unnamed remainder (the keyword-classifying draft
/// documented above was noisy and remains a non-goal); it forces the
/// DECISION — adding or unnaming a site moves a count, and the failure
/// message demands either a `GUARD_STOP_*` at the site or an updated pin
/// with the natural-stop rationale. Same idiom as
/// `w4a16_gemm_t_ldb_drift_is_exactly_the_known_set`.
fn scan_finish_ledger(src: &str) -> (usize, usize) {
    // Comment lines are neither sites nor evidence of naming — emit_step.rs
    // SAYS "a.finished = true" in prose, and prose is not a cut.
    let lines: Vec<&str> = src
        .lines()
        .map(|l| {
            if l.trim_start().starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect();
    let mut total = 0usize;
    let mut named = 0usize;
    for (i, line) in lines.iter().enumerate() {
        if !line.contains("finished = true") {
            continue;
        }
        total += 1;
        let from = i.saturating_sub(WINDOW);
        let to = (i + WINDOW).min(lines.len() - 1);
        if lines[from..=to].join("\n").contains(NAMED) {
            named += 1;
        }
    }
    (total, named)
}

#[test]
fn every_finish_site_is_a_recorded_decision() {
    // (file, source, total `finished = true` sites, sites naming a guard).
    //
    // Named — decode_logits_step: `<tool_response>`, think-skip watchdog,
    // fuzzy_repetition. emit_step: `<tool_response>`, think-skip watchdog,
    // inter_tool_prose_budget, tool_envelope_stuck.
    // Unnamed remainder (deliberate): natural stops — the model sampled EOS,
    // the token budget / context ceiling hit, the cooperative cancel flag,
    // and `<|im_start|>` (eos-registered; see INTENTIONAL_STOP above for
    // why naming it would reintroduce the opposite mislabel).
    const LEDGER: &[(&str, &str, usize, usize)] = &[
        (
            "decode_logits_step.rs",
            include_str!("decode_logits_step.rs"),
            11,
            3,
        ),
        ("emit_step.rs", include_str!("emit_step.rs"), 11, 4),
    ];
    for (name, src, want_total, want_named) in LEDGER {
        let (total, named) = scan_finish_ledger(src);
        assert_eq!(
            (total, named),
            (*want_total, *want_named),
            "{name}: finish-site ledger drift — found {total} sites / {named} named, \
             pinned {want_total} / {want_named}. If you ADDED a site: a cut the \
             SERVER decides (watchdog, control token, degeneration) must set \
             `a.guard_stop = Some(GUARD_STOP_*)` at the site — unnamed it wires \
             finish_reason \"stop\" and agentic clients treat the truncation as a \
             finished turn. A NATURAL stop (model EOS / budget / client cancel) \
             updates this pin AND the rationale comment above it. If a count \
             DROPPED, a site or its guard name was removed — verify that was the \
             intent."
        );
    }
}

#[test]
fn the_ledger_counts_sites_credits_naming_and_ignores_prose() {
    // Detector proof for channel 2, mirroring channel 1's fixture test: a
    // bare site counts unnamed, a named site is credited, and comment prose
    // is invisible — the exact false-positive class that killed the earlier
    // keyword-classifying draft.
    let bare = "\
        if a.think_skip_count >= 50 {\n\
        \x20   a.finished = true;\n\
        }\n";
    assert_eq!(
        scan_finish_ledger(bare),
        (1, 0),
        "a bare finish site is 1 site, 0 named"
    );

    let named = "\
        if a.think_skip_count >= 50 {\n\
        \x20   a.finished = true;\n\
        \x20   a.guard_stop = Some(GUARD_STOP_THINK_SKIP);\n\
        }\n";
    assert_eq!(
        scan_finish_ledger(named),
        (1, 1),
        "a guard named in the window must be credited"
    );

    let prose = "// Previously we set `a.finished = true` here -- a\n";
    assert_eq!(
        scan_finish_ledger(prose),
        (0, 0),
        "prose mentioning the mutation is not a cut site"
    );
}

#[test]
fn the_intentional_stop_exemption_is_load_bearing_and_narrow() {
    // `<|im_start|>` must be exempt (it is eos-registered, so "stop" is right)...
    let im_start = "\
        if tok == ims {\n\
        \x20   a.output_tokens.push(tok);\n\
        \x20   a.finished = true;\n\
        \x20   tracing::debug!(\"<|im_start|> hard-stop fired (id={ims})\");\n\
        }\n";
    let (n, flagged) = scan(im_start);
    assert_eq!(n, 0, "<|im_start|> must be exempt — it is eos-registered");
    assert!(flagged.is_empty());

    // ...but the exemption must NOT be so broad it swallows the sibling it
    // sits next to. This is the guard against "fixing" the test by widening
    // the exemption until nothing is ever flagged.
    // Separated by more than WINDOW, as the real sites are — otherwise the
    // fixture itself would put the exemption inside the sibling's window and
    // "prove" a narrowness the scan does not have.
    let gap = "\n".repeat(WINDOW + 2);
    let both = format!(
        "{im_start}{gap}if tok == trs {{\n    a.finished = true;\n    \
         tracing::debug!(\"<tool_response> hard-stop fired\");\n}}\n"
    );
    let (n, flagged) = scan(&both);
    assert_eq!(n, 1, "the <tool_response> site must still be counted");
    assert_eq!(flagged.len(), 1, "and still flagged when bare");
}
