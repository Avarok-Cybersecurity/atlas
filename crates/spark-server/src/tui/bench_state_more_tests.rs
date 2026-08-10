// SPDX-License-Identifier: AGPL-3.0-only

//! Selection, parameter validation and the run-pane bookkeeping. Kept apart
//! from `bench_state_tests.rs` for the 500-line cap, not because the subject
//! differs.

use super::*;

/// The first benchmark's form, pointed at an endpoint nothing is listening on.
/// No executor: nothing here can start a run, which is the point.
fn state() -> BenchState {
    let mut s = BenchState {
        target: TargetEndpoint::local(8888, "test-model"),
        ..Default::default()
    };
    s.select(0);
    s
}

#[test]
fn selecting_past_the_end_of_the_registry_lands_on_the_last_benchmark() {
    // `select` is called with a raw index from the list cursor and from `d`.
    let mut s = state();
    s.select(usize::MAX);
    assert_eq!(s.selected, atlas_plugin::registry::all().len() - 1);
    assert!(s.descriptor().is_some(), "and a descriptor still resolves");
}

#[test]
fn selecting_a_benchmark_clears_the_previous_ones_errors_and_cursor() {
    let mut s = state();
    s.edit[0] = "nonsense".into();
    s.commit_row(0);
    s.row = s.row_count() - 1;
    s.editing = true;
    s.select(1);
    assert!(
        s.errors.is_empty(),
        "another benchmark's errors are not ours"
    );
    assert_eq!(s.row, 0);
    assert!(!s.is_editing());
}

#[test]
fn every_row_of_every_benchmark_has_something_to_draw() {
    // The detail pane renders all three of these for each row; an empty hint or
    // label is a blank cell in the form.
    let mut s = state();
    for i in 0..atlas_plugin::registry::all().len() {
        s.select(i);
        for row in 0..s.row_count() {
            let (label, help, hint) = s.row_meta(row);
            assert!(!label.is_empty(), "{i}/{row}");
            assert!(!help.is_empty(), "{i}/{row}");
            assert!(!hint.is_empty(), "{i}/{row}");
        }
    }
}

#[test]
fn an_empty_value_is_refused_for_every_parameter_of_every_benchmark() {
    // "Silently coerced" is the failure mode that matters here: an empty field
    // that fell back to a default would run a different sweep than the form
    // shows.
    let mut s = state();
    for i in 0..atlas_plugin::registry::all().len() {
        s.select(i);
        for row in 0..s.specs.len() {
            let key = s.specs[row].key;
            let before = s.values.get(key).cloned();
            s.edit[row] = "   ".into();
            s.commit_row(row);
            assert!(
                s.row_error(row).is_some(),
                "{}: {key} accepted an empty value",
                s.descriptor().expect("selected").id
            );
            assert_eq!(
                s.values.get(key).cloned(),
                before,
                "{key} was changed by a value that did not parse"
            );
        }
    }
}

#[test]
fn a_number_outside_its_domain_names_the_bound_it_broke() {
    let mut s = state();
    let row = s
        .specs
        .iter()
        .position(|spec| matches!(spec.kind, atlas_plugin::ParamKind::Int { .. }))
        .expect("the sweep has an integer parameter");
    let key = s.specs[row].key;
    let before = s.values.get(key).cloned();
    s.edit[row] = "-1".into();
    s.commit_row(row);
    let error = s.row_error(row).expect("out of range is refused");
    assert!(error.contains("between"), "{error}");
    assert_eq!(s.values.get(key).cloned(), before, "and nothing was stored");
}

#[test]
fn fixing_one_field_does_not_clear_another_fields_error() {
    // The errors are keyed per field precisely so a form with two mistakes
    // reports two.
    let mut s = state();
    assert!(s.specs.len() >= 2, "the sweep has several parameters");
    s.edit[0] = "nonsense".into();
    s.edit[1] = "nonsense".into();
    s.commit_row(0);
    s.commit_row(1);
    assert_eq!(s.errors.len(), 2);
    s.edit[0] = s.specs[0].default.to_edit_string();
    s.commit_row(0);
    assert_eq!(s.errors.len(), 1);
    assert!(s.row_error(1).is_some());
    let refusal = s.start().unwrap_err();
    assert!(refusal.contains("1 field(s)"), "{refusal}");
}

#[test]
fn a_row_that_does_not_exist_is_answered_rather_than_panicking() {
    // `row` is clamped by the key handler, so this is about totality: every
    // accessor takes a raw index and must not index out of its vectors.
    let s = state();
    let far = s.row_count() + 99;
    let (label, help, hint) = s.row_meta(far);
    assert!(!label.is_empty() && !help.is_empty() && !hint.is_empty());
    assert!(s.row_error(far).is_none());
}

#[test]
fn typing_a_model_into_the_form_pins_the_target_against_the_live_one() {
    // The trap this closes: with --auto-swap, a benchmark request carrying a
    // stale model name is itself what swaps the server back, mid-run.
    let mut s = state();
    let row = s.specs.len() + 1;
    s.edit[row] = "  org/deliberate  ".into();
    s.commit_row(row);
    assert_eq!(s.target.model, "org/deliberate", "trimmed and stored");
    assert!(s.target_model_pinned);
    s.follow_live_model("org/whatever-is-serving");
    assert_eq!(s.target.model, "org/deliberate");
}

#[test]
fn an_empty_model_is_refused_and_does_not_pin_anything() {
    let mut s = state();
    let row = s.specs.len() + 1;
    s.edit[row] = String::new();
    s.commit_row(row);
    assert!(s.row_error(row).is_some());
    assert_eq!(s.target.model, "test-model", "unchanged");
    assert!(!s.target_model_pinned);
}

#[test]
fn following_the_model_already_targeted_changes_nothing() {
    let mut s = state();
    s.follow_live_model("test-model");
    assert_eq!(s.target, TargetEndpoint::local(8888, "test-model"));
    assert!(!s.target_model_pinned, "an unchanged follow is not a pin");
}

#[test]
fn a_start_with_the_probe_switched_off_still_refuses_for_the_real_reason() {
    // `begin_start` with `Skip` IS `start`; the refusal must be the same one.
    let mut s = state();
    s.coherence = atlas_plugin::CoherencePolicy::Skip;
    let err = s.begin_start().unwrap_err();
    assert!(err.contains("executor"), "{err}");
    assert!(s.preflight.is_none(), "and no check was opened");
}

#[test]
fn a_doomed_run_never_reaches_the_endpoint_check() {
    // Refusing up front is what keeps a bad form from costing two completions.
    let mut s = state();
    s.edit[0] = "nonsense".into();
    s.commit_row(0);
    let err = s.begin_start().unwrap_err();
    assert!(err.contains("need fixing"), "{err}");
    assert!(s.preflight.is_none());
}

#[test]
fn a_reported_concern_keeps_the_modal_up_until_it_is_answered() {
    let mut s = state();
    s.preflight = Some(crate::tui::bench_preflight::Preflight::with_concern(
        "wrong model".into(),
    ));
    s.poll_preflight();
    assert!(s.preflight.is_some(), "a concern is the user's to decide");
    assert_ne!(s.view, View::Run, "and nothing started behind it");
}

#[test]
fn a_check_still_in_flight_neither_starts_nor_clears() {
    let mut s = state();
    s.preflight = Some(crate::tui::bench_preflight::Preflight::pending());
    s.poll_preflight();
    assert!(s.preflight.is_some());
    assert_ne!(s.view, View::Run);
}

#[test]
fn polling_with_no_check_open_is_a_no_op() {
    let mut s = state();
    s.poll_preflight();
    assert!(s.preflight.is_none());
}

#[test]
fn overruling_a_concern_dismisses_it_and_reports_why_the_run_still_cannot_start() {
    let mut s = state();
    s.preflight = Some(crate::tui::bench_preflight::Preflight::with_concern(
        "wrong model".into(),
    ));
    let err = s.accept_preflight().unwrap_err();
    assert!(err.contains("executor"), "{err}");
    assert!(s.preflight.is_none(), "the modal comes down either way");
}

#[test]
fn abandoning_a_check_leaves_the_form_as_it_was() {
    let mut s = state();
    s.preflight = Some(crate::tui::bench_preflight::Preflight::pending());
    s.cancel_preflight();
    assert!(s.preflight.is_none());
    assert!(!s.is_running());
    assert_eq!(s.view, View::List);
}

#[test]
fn pumping_with_no_run_in_flight_does_nothing() {
    let mut s = state();
    s.status = "idle".into();
    s.pump();
    assert_eq!(s.status, "idle");
    assert!(s.log.is_empty());
    assert!(!s.glow);
}

#[test]
fn the_run_log_is_a_window_that_drops_its_oldest_lines() {
    // A benchmark that logs for three hours must not grow the pane's buffer
    // without bound.
    let mut s = state();
    for i in 0..LOG_CAPACITY + 25 {
        s.push_log(LogLine::info(format!("line {i}")));
    }
    assert_eq!(s.log.len(), LOG_CAPACITY);
    assert_eq!(s.log.front().expect("kept").text, format!("line {}", 25));
    assert_eq!(
        s.log.back().expect("kept").text,
        format!("line {}", LOG_CAPACITY + 24)
    );
}

#[test]
fn the_elapsed_clock_reads_zero_until_a_run_starts() {
    let mut s = state();
    assert_eq!(s.elapsed_text(), "00:00:00");
    s.started = Some(std::time::Instant::now());
    assert_eq!(s.elapsed_text(), "00:00:00", "a run that just began");
}

#[test]
fn history_is_empty_rather_than_an_error_when_no_run_has_been_recorded() {
    let mut s = state();
    s.load_history();
    assert!(s.history.is_empty());
    assert_eq!(s.history_row, 0, "the cursor is clamped, not left dangling");
}

#[test]
fn the_detail_pane_has_something_to_say_before_anything_is_selected() {
    // `attach` has not run yet on the first frame.
    let s = BenchState::default();
    assert!(s.descriptor().is_some(), "the registry is never empty");
    assert_eq!(s.plugin_metadata().description, "no benchmark selected");
    assert!(s.plugin_metadata().official);
    assert!(s.specs.is_empty());
    assert_eq!(s.row_count(), 2, "the two target rows exist regardless");
}

#[test]
fn a_selected_benchmark_reports_its_own_provenance() {
    let mut s = state();
    for i in 0..atlas_plugin::registry::all().len() {
        s.select(i);
        let meta = s.plugin_metadata();
        assert_ne!(meta.description, "no benchmark selected");
        assert!(meta.official, "everything in the registry ships with Atlas");
        assert!(!meta.bug_report_url.is_empty());
    }
}
