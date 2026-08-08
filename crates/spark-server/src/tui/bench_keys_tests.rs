// SPDX-License-Identifier: AGPL-3.0-only

//! What a key MEANS in each of the four panes. Nothing here starts a run: no
//! executor is attached, so `start` refuses and the refusal is the observable.

use crossterm::event::{KeyCode, KeyEvent};

use super::*;
use crate::tui::bench_preflight::{Phase, Preflight};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

/// The first benchmark's form, pointed at an endpoint nothing is listening on.
///
/// `attach` is the only public way to set a target and it needs an executor —
/// which would let a stray `s` in one of these tests actually start a run.
fn state() -> BenchState {
    let mut s = BenchState::default();
    s.select(0);
    s.target = atlas_plugin::TargetEndpoint::local(8888, "test-model");
    // Reload so the two target rows carry the endpoint set above.
    s.select(0);
    s
}

fn params() -> BenchState {
    let mut s = state();
    s.view = View::Params;
    s
}

fn toast(outcome: Outcome) -> String {
    match outcome {
        Outcome::Toast { text, error } => {
            assert!(error, "a refusal is an error toast: {text}");
            text
        }
        Outcome::None => panic!("expected a toast"),
    }
}

fn is_silent(outcome: Outcome) -> bool {
    matches!(outcome, Outcome::None)
}

#[test]
fn the_arrow_keys_and_the_vim_keys_are_the_same_keys() {
    let mut arrows = state();
    let mut vim = state();
    arrows.on_key(key(KeyCode::Down), BenchSub::Suite);
    vim.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    assert_eq!(arrows.selected, vim.selected);
    assert_eq!(arrows.selected, 1);
    arrows.on_key(key(KeyCode::Up), BenchSub::Suite);
    vim.on_key(key(KeyCode::Char('k')), BenchSub::Suite);
    assert_eq!(arrows.selected, vim.selected);
    assert_eq!(arrows.selected, 0);
}

#[test]
fn an_unbound_key_in_the_list_changes_nothing() {
    let mut s = state();
    assert!(is_silent(
        s.on_key(key(KeyCode::Char('z')), BenchSub::Suite)
    ));
    assert_eq!(s.selected, 0);
    assert_eq!(s.view, View::List);
}

#[test]
fn the_run_pane_is_only_reachable_from_the_list_once_there_is_a_run_to_see() {
    let mut s = state();
    s.on_key(key(KeyCode::Char('v')), BenchSub::Suite);
    assert_eq!(s.view, View::List, "nothing has run yet");
}

#[test]
fn a_finished_run_stays_reachable_after_navigating_away() {
    let mut s = state();
    s.frame = Some(atlas_plugin::BenchmarkResult {
        status: atlas_plugin::RunStatus::Completed,
        phase: "done".into(),
        progress: None,
        summary: Vec::new(),
        table: None,
        verdict: None,
        metrics: std::collections::BTreeMap::new(),
        log: Vec::new(),
        elapsed: std::time::Duration::from_secs(1),
    });
    s.on_key(key(KeyCode::Char('v')), BenchSub::Suite);
    assert_eq!(s.view, View::Run);
}

#[test]
fn opening_a_benchmark_that_is_not_running_opens_its_form() {
    // `is_running` is false with no executor, which is exactly the case where
    // Enter must offer the form rather than an empty run pane.
    let mut s = state();
    s.on_key(key(KeyCode::Right), BenchSub::Suite);
    assert_eq!(s.view, View::Params);
}

#[test]
fn row_navigation_in_the_form_is_clamped_to_the_fields_that_exist() {
    let mut s = params();
    for _ in 0..s.row_count() + 5 {
        s.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    }
    assert_eq!(s.row, s.row_count() - 1, "the last row is the Model field");
    for _ in 0..s.row_count() + 5 {
        s.on_key(key(KeyCode::Up), BenchSub::Suite);
    }
    assert_eq!(s.row, 0);
}

#[test]
fn typing_edits_only_the_focused_row() {
    let mut s = params();
    let untouched = s.edit[1].clone();
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    for c in "12".chars() {
        s.on_key(key(KeyCode::Char(c)), BenchSub::Suite);
    }
    s.on_key(key(KeyCode::Backspace), BenchSub::Suite);
    assert!(s.edit[0].ends_with('1'), "{}", s.edit[0]);
    assert_eq!(s.edit[1], untouched);
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    assert!(!s.is_editing(), "Enter commits and hands the keyboard back");
}

#[test]
fn backspace_on_an_empty_buffer_is_not_an_underflow() {
    let mut s = params();
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    for _ in 0..s.edit[0].len() + 5 {
        s.on_key(key(KeyCode::Backspace), BenchSub::Suite);
    }
    assert_eq!(s.edit[0], "");
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    assert!(
        s.row_error(0).is_some(),
        "an empty value is refused, not kept"
    );
}

#[test]
fn a_cancelled_edit_of_the_endpoint_restores_the_committed_url() {
    // Esc restores what the value IS, not what the schema's default was.
    let mut s = params();
    s.row = s.specs.len();
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    s.on_key(key(KeyCode::Char('x')), BenchSub::Suite);
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert_eq!(s.edit[s.row], s.target.base_url);
    assert!(!s.is_editing());
}

#[test]
fn a_cancelled_edit_of_the_model_restores_the_committed_model() {
    let mut s = params();
    s.row = s.specs.len() + 1;
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    s.on_key(key(KeyCode::Backspace), BenchSub::Suite);
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert_eq!(s.edit[s.row], "test-model");
}

#[test]
fn the_coherence_probe_toggles_both_ways() {
    // A base checkpoint cannot answer the probe's questions and is still a
    // valid latency target, so the operator has to be able to turn it off — and
    // back on.
    let mut s = params();
    assert_eq!(s.coherence, atlas_plugin::CoherencePolicy::Probe);
    s.on_key(key(KeyCode::Char('p')), BenchSub::Suite);
    assert_eq!(s.coherence, atlas_plugin::CoherencePolicy::Skip);
    s.on_key(key(KeyCode::Char('p')), BenchSub::Suite);
    assert_eq!(s.coherence, atlas_plugin::CoherencePolicy::Probe);
}

#[test]
fn leaving_the_form_keeps_what_was_typed_into_it() {
    let mut s = params();
    s.edit[0] = "7".into();
    s.commit_row(0);
    let committed = s.values.clone();
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert_eq!(s.view, View::List);
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    assert_eq!(s.view, View::Params);
    assert_eq!(s.values, committed, "stepping back is not a reset");
}

#[test]
fn re_selecting_the_same_benchmark_from_the_list_does_reset_the_form() {
    // `d` and a re-`select` are the only resets, and they are deliberate.
    let mut s = params();
    s.edit[0] = "7".into();
    s.commit_row(0);
    s.on_key(key(KeyCode::Char('d')), BenchSub::Suite);
    assert_eq!(s.edit[0], s.specs[0].default.to_edit_string());
}

#[test]
fn starting_without_an_executor_toasts_the_reason_rather_than_switching_view() {
    let mut s = params();
    let text = toast(s.on_key(key(KeyCode::Char('s')), BenchSub::Suite));
    assert!(text.contains("executor"), "{text}");
    assert_eq!(s.view, View::Params, "a refused start stays on the form");
}

#[test]
fn a_start_refused_for_an_invalid_field_says_how_many_need_fixing() {
    let mut s = params();
    s.edit[0] = "nonsense".into();
    s.commit_row(0);
    let text = toast(s.on_key(key(KeyCode::Char('s')), BenchSub::Suite));
    assert!(text.contains("1 field(s) need fixing"), "{text}");
}

#[test]
fn the_consent_gate_only_opens_for_the_benchmark_that_runs_model_authored_shell() {
    let mut s = params();
    let index = atlas_plugin::registry::all()
        .iter()
        .position(|d| d.needs_confirmation)
        .expect("the agentic benchmark requires confirmation");
    s.select(index);
    s.view = View::Params;
    assert!(is_silent(
        s.on_key(key(KeyCode::Char('s')), BenchSub::Suite)
    ));
    assert!(s.confirm_open);
    // `y` is the only key that proceeds; it then refuses for the honest reason.
    let text = toast(s.on_key(key(KeyCode::Char('y')), BenchSub::Suite));
    assert!(text.contains("executor"), "{text}");
    assert!(!s.confirm_open, "the gate closes once answered");
}

#[test]
fn the_form_underneath_the_consent_gate_is_not_edited_by_the_answer() {
    let mut s = params();
    let index = atlas_plugin::registry::all()
        .iter()
        .position(|d| d.needs_confirmation)
        .expect("a benchmark that asks");
    s.select(index);
    s.view = View::Params;
    s.on_key(key(KeyCode::Char('s')), BenchSub::Suite);
    let before = s.edit.clone();
    // `d` would reset the form if it reached the form.
    s.on_key(key(KeyCode::Char('d')), BenchSub::Suite);
    assert!(!s.confirm_open, "anything but y backs out");
    assert_eq!(s.edit, before);
}

#[test]
fn only_esc_answers_a_pre_flight_that_is_still_checking() {
    // There is nothing to decide yet, so every other key has to be inert —
    // including the ones that would edit the form underneath.
    let mut s = params();
    s.preflight = Some(Preflight::pending());
    for code in [
        KeyCode::Char('d'),
        KeyCode::Char('p'),
        KeyCode::Char('j'),
        KeyCode::Enter,
    ] {
        s.on_key(key(code), BenchSub::Suite);
        assert!(s.preflight.is_some(), "{code:?} must not answer the check");
    }
    assert_eq!(s.row, 0, "and must not move the form underneath");
    assert_eq!(s.coherence, atlas_plugin::CoherencePolicy::Probe);
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert!(s.preflight.is_none(), "Esc abandons the run");
    assert_eq!(s.view, View::Params);
}

#[test]
fn a_reported_concern_can_be_overruled_and_the_run_proceeds() {
    // The probe is a warning, never a veto.
    for code in [KeyCode::Char('p'), KeyCode::Char('P'), KeyCode::Enter] {
        let mut s = params();
        s.preflight = Some(Preflight::with_concern("a concern".into()));
        let text = toast(s.on_key(key(code), BenchSub::Suite));
        assert!(text.contains("executor"), "{code:?} tried to start: {text}");
        assert!(s.preflight.is_none(), "{code:?} dismissed the modal");
    }
}

#[test]
fn any_other_key_backs_out_of_a_reported_concern() {
    let mut s = params();
    s.preflight = Some(Preflight::with_concern("a concern".into()));
    assert!(is_silent(
        s.on_key(key(KeyCode::Char('n')), BenchSub::Suite)
    ));
    assert!(s.preflight.is_none());
    assert_eq!(s.view, View::Params, "back to the form, not into a run");
}

#[test]
fn a_pending_pre_flight_reports_that_it_is_still_checking() {
    let s = Preflight::pending();
    assert!(s.is_checking());
    assert_eq!(s.phase, Phase::Checking);
}

#[test]
fn the_run_pane_scrolls_and_clamps_at_the_top() {
    let mut s = state();
    s.view = View::Run;
    for _ in 0..3 {
        s.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    }
    assert_eq!(s.table_scroll, 3);
    for _ in 0..10 {
        s.on_key(key(KeyCode::Up), BenchSub::Suite);
    }
    assert_eq!(s.table_scroll, 0, "no underflow past the first row");
    s.on_key(key(KeyCode::Down), BenchSub::Suite);
    s.on_key(key(KeyCode::Home), BenchSub::Suite);
    assert_eq!(s.table_scroll, 0);
    s.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    s.on_key(key(KeyCode::Char('g')), BenchSub::Suite);
    assert_eq!(s.table_scroll, 0);
}

#[test]
fn cancelling_when_nothing_is_running_says_nothing_and_does_nothing() {
    let mut s = state();
    s.view = View::Run;
    assert!(is_silent(
        s.on_key(key(KeyCode::Char('c')), BenchSub::Suite)
    ));
    assert_eq!(s.status, "", "there is no run to report on");
    assert_eq!(s.view, View::Run);
}

#[test]
fn the_run_pane_steps_back_to_the_list() {
    for code in [KeyCode::Esc, KeyCode::Left, KeyCode::Char('h')] {
        let mut s = state();
        s.view = View::Run;
        s.on_key(key(code), BenchSub::Suite);
        assert_eq!(s.view, View::List, "{code:?}");
    }
}

#[test]
fn history_navigation_on_an_empty_history_does_not_move_a_cursor_it_has_no_rows_for() {
    let mut s = state();
    assert!(s.history.is_empty());
    s.on_key(key(KeyCode::Char('j')), BenchSub::History);
    s.on_key(key(KeyCode::Down), BenchSub::History);
    assert_eq!(s.history_row, 0);
    assert_eq!(s.view, View::List, "History does not drive the Suite view");
}

#[test]
fn history_navigation_is_clamped_to_the_rows_that_exist() {
    let mut s = state();
    s.history = vec![record(), record(), record()];
    for _ in 0..10 {
        s.on_key(key(KeyCode::Char('j')), BenchSub::History);
    }
    assert_eq!(s.history_row, 2);
    for _ in 0..10 {
        s.on_key(key(KeyCode::Char('k')), BenchSub::History);
    }
    assert_eq!(s.history_row, 0);
}

/// A minimal persisted run, for the History pane's cursor.
fn record() -> atlas_plugin::RunRecord {
    let descriptor = atlas_plugin::registry::find("concurrency-sweep").expect("registered");
    atlas_plugin::RunRecord::new(
        descriptor,
        &atlas_plugin::ParamValues::default(),
        &atlas_plugin::TargetEndpoint::local(8888, "m"),
        atlas_plugin::RunSource::Tui,
        crate::cli::ATLAS_VERSION,
        atlas_plugin::BenchmarkResult {
            status: atlas_plugin::RunStatus::Completed,
            phase: "done".into(),
            progress: None,
            summary: Vec::new(),
            table: None,
            verdict: None,
            metrics: std::collections::BTreeMap::new(),
            log: Vec::new(),
            elapsed: std::time::Duration::from_secs(1),
        },
    )
}
