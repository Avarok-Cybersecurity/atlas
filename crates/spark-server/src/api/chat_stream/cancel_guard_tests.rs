// SPDX-License-Identifier: AGPL-3.0-only

//! THE invariant behind three separate shipped regressions:
//!
//! **Every site that flips `cancel_flag` must also name why**, by setting
//! one of `guard_stop`, `tool_loop_capped`, or `stop_string_matched`.
//!
//! A bare `cancel_flag` reaches the scheduler as "generation stopped with
//! budget left and no guard", which `derive_finish_reason` correctly maps
//! to `"stop"` — telling the client *the model finished*. For a stream we
//! truncated mid-doom-loop that is false, and it silently breaks every
//! client that keys truncation recovery on `"length"`: `openai-python`
//! raises `LengthFinishReasonError`, aider's continuation fires, Instructor
//! raises `IncompleteOutputException`, pydantic-ai refuses a half tool call.
//!
//! ★ This was found the expensive way. Three separate cut paths shipped
//! bare, and each was fixed only after a gate failure pointed at it:
//!   1. scheduler-side guards        (fixed: guard_stop_wire_reason)
//!   2. stream simhash / token-loop  (fixed: resolve_wire_finish_reason rung)
//!   3. orphan-suppression streak    (fixed: sets guard_stop at the site)
//! Fixes 1 and 2 left path 3 bare, and it cost the agentic gate runs 0 and
//! 7 on three consecutive shas — a failure that looked like a model
//! regression and was actually a missing field.
//!
//! A per-path unit test would not have caught the class: each fix was
//! locally correct and locally tested. Only an invariant over ALL sites
//! catches the fourth one, so this test reads the source rather than
//! exercising behaviour. Precedent in-tree: `coverage_map_tests` asserts
//! benchmark drivers do not import each other for the same reason — the
//! property is structural, so the test is too.

/// Files in `chat_stream` that may cancel a stream.
///
/// Derived by grepping `spark-server` for a `.store(true` within three lines
/// of a `cancel_flag`; keep it that way when adding an entry. `state.rs` was
/// missed on the first pass and its site — `note_stop_string_match` — was
/// correctly guarded by luck, not by this test. A file omitted here is a
/// silent hole, which is why `checked` has a floor below.
const CANCEL_SITE_FILES: &[(&str, &str)] = &[
    ("handle_token.rs", include_str!("handle_token.rs")),
    ("tool_handlers.rs", include_str!("tool_handlers.rs")),
    ("state.rs", include_str!("state.rs")),
];

/// Any one of these, within the window above a `cancel_flag` store, means
/// the cut is named and will reach the wire as `"length"`.
const GUARD_MARKERS: &[&str] = &[
    "guard_stop = Some(",
    "tool_loop_capped = true",
    "stop_string_matched = true",
];

/// How far back to look. The marker is set in the same statement group as
/// the store in every current site; 25 lines is slack, not licence.
const WINDOW: usize = 25;

/// True when this store is provably unreachable and so cannot mislabel.
fn is_dead_path(preceding: &str) -> bool {
    // The retry cut in `tool_handlers.rs` is gated behind `tool_retry_enabled`,
    // a `const false`, so it can never execute. Kept compiling as documentation
    // of the retry design, so exempt it explicitly rather than by silence.
    //
    // ★ Match the ASSIGNMENT, not the bare identifier. The first version of
    // this test looked for `pending_retry` anywhere in the window, which also
    // matched the unrelated `pending_retry: None` struct-field initializer in
    // `state.rs` — silently exempting a live cut site 15 lines below it and
    // making the whole invariant vacuous for that file. The exemption must be
    // narrower than the window it is searched in.
    preceding.contains("pending_retry = Some(")
}

#[test]
fn every_cancel_flag_store_names_its_guard() {
    let mut bare: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for (name, src) in CANCEL_SITE_FILES {
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains(".store(true") {
                continue;
            }
            // Confirm it is the cancel flag and not some other AtomicBool.
            let lo = i.saturating_sub(3);
            if !lines[lo..=i].iter().any(|l| l.contains("cancel_flag")) {
                continue;
            }
            checked += 1;
            let from = i.saturating_sub(WINDOW);
            let preceding = lines[from..=i].join("\n");
            if is_dead_path(&preceding) {
                continue;
            }
            if !GUARD_MARKERS.iter().any(|m| preceding.contains(m)) {
                bare.push(format!("{name}:{}", i + 1));
            }
        }
    }

    // The floor is the negative half: if a refactor moves or renames the
    // sites, this test must fail loudly rather than pass over an empty
    // scan and report a green it never earned.
    assert!(
        checked >= 4,
        "found only {checked} cancel_flag stores — the scan stopped matching. \
         Fix the detection before trusting a green here."
    );

    assert!(
        bare.is_empty(),
        "cancel_flag flipped without naming a guard at: {bare:?}\n\
         A bare cancel reaches the client as finish_reason \"stop\", claiming \
         the model finished when it was truncated. Set guard_stop (or \
         tool_loop_capped / stop_string_matched) at the site so the wire \
         reason is \"length\"."
    );
}

#[test]
fn the_scan_would_notice_a_bare_store() {
    // NEGATIVE case: prove the detector fires. Without this, the test
    // above passes for two reasons — the invariant holding, or the scan
    // silently matching nothing — and only one of them is good news.
    let synthetic = "\
        if streak > MAX {\n\
        \x20   tracing::warn!(\"ending stream\");\n\
        \x20   state.loop_watchdog_triggered = true;\n\
        \x20   state.cancel_flag.store(true, Ordering::Release);\n\
        }\n";
    let lines: Vec<&str> = synthetic.lines().collect();
    let idx = lines
        .iter()
        .position(|l| l.contains(".store(true"))
        .expect("fixture must contain a store");
    let preceding = lines[..=idx].join("\n");
    assert!(
        !GUARD_MARKERS.iter().any(|m| preceding.contains(m)),
        "the detector failed to flag a store with no guard marker — it \
         would pass the real invariant test vacuously"
    );
    // And the positive half: adding the marker clears it.
    let fixed = preceding.replace(
        "state.loop_watchdog_triggered = true;",
        "state.guard_stop = Some(\"suppress_streak\");",
    );
    assert!(GUARD_MARKERS.iter().any(|m| fixed.contains(m)));
}
