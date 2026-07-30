// SPDX-License-Identifier: AGPL-3.0-only

use crossterm::event::{KeyCode, KeyEvent};

use super::*;
use crate::tui::app::BenchSub;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

fn typed(state: &mut BenchState, text: &str) {
    for c in text.chars() {
        state.on_key(key(KeyCode::Char(c)), BenchSub::Suite);
    }
}

/// A state with the first benchmark selected. No executor is attached, so
/// nothing can actually start — which is what the "refuses to start" tests want.
fn state() -> BenchState {
    let mut s = BenchState {
        target: TargetEndpoint::local(8888, "test-model"),
        ..Default::default()
    };
    s.select(0);
    s
}

#[test]
fn selecting_loads_the_schema_and_its_defaults() {
    let s = state();
    assert!(
        !s.specs.is_empty(),
        "the first benchmark declares parameters"
    );
    // One edit buffer per parameter, plus URL and model.
    assert_eq!(s.edit.len(), s.specs.len() + 2);
    assert_eq!(s.row_count(), s.specs.len() + 2);
    for (i, spec) in s.specs.iter().enumerate() {
        assert_eq!(s.edit[i], spec.default.to_edit_string());
    }
}

#[test]
fn every_registered_benchmark_can_be_selected() {
    let mut s = state();
    for i in 0..atlas_plugin::registry::all().len() {
        s.select(i);
        assert!(s.descriptor().is_some());
        assert!(!s.plugin_metadata().description.is_empty());
    }
}

#[test]
fn the_last_two_rows_are_always_the_endpoint() {
    let s = state();
    let (url_label, _, _) = s.row_meta(s.specs.len());
    let (model_label, _, _) = s.row_meta(s.specs.len() + 1);
    assert_eq!(url_label, "Endpoint URL");
    assert_eq!(model_label, "Model");
}

#[test]
fn committing_a_valid_edit_updates_the_value_and_clears_the_error() {
    let mut s = state();
    let key_name = s.specs[0].key;
    s.edit[0] = "nonsense".into();
    s.commit_row(0);
    assert!(s.row_error(0).is_some(), "a bad value must be reported");
    s.edit[0] = s.specs[0].default.to_edit_string();
    s.commit_row(0);
    assert!(s.row_error(0).is_none());
    assert_eq!(s.values.get(key_name), Some(&s.specs[0].default));
}

#[test]
fn editing_the_endpoint_normalises_the_trailing_slash() {
    let mut s = state();
    let row = s.specs.len();
    s.edit[row] = "http://10.10.10.3:8888/".into();
    s.commit_row(row);
    assert_eq!(s.target.base_url, "http://10.10.10.3:8888");
    assert_eq!(s.edit[row], "http://10.10.10.3:8888");
    assert!(s.row_error(row).is_none());
}

#[test]
fn an_empty_endpoint_is_rejected_rather_than_silently_kept() {
    let mut s = state();
    let row = s.specs.len();
    s.edit[row] = "   ".into();
    s.commit_row(row);
    assert!(s.row_error(row).is_some());
    assert_eq!(s.target.base_url, "http://127.0.0.1:8888", "unchanged");
}

#[test]
fn a_cancelled_edit_restores_the_committed_value() {
    let mut s = state();
    s.view = View::Params;
    let original = s.edit[0].clone();
    s.on_key(key(KeyCode::Enter), BenchSub::Suite); // start editing
    assert!(s.is_editing());
    typed(&mut s, "9");
    assert_ne!(s.edit[0], original);
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert!(!s.is_editing());
    assert_eq!(s.edit[0], original, "Esc must not leave a half-typed value");
}

#[test]
fn digits_typed_into_a_field_do_not_escape_to_the_section_keys() {
    let mut s = state();
    s.view = View::Params;
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    assert!(
        s.is_editing(),
        "the app checks is_editing() before treating digits as section jumps"
    );
}

#[test]
fn the_form_can_be_reset_to_the_schema_defaults() {
    let mut s = state();
    s.view = View::Params;
    s.edit[0] = "nonsense".into();
    s.commit_row(0);
    assert!(!s.errors.is_empty());
    s.on_key(key(KeyCode::Char('d')), BenchSub::Suite);
    assert!(s.errors.is_empty());
    assert_eq!(s.edit[0], s.specs[0].default.to_edit_string());
}

#[test]
fn starting_with_an_invalid_field_is_refused_and_says_why() {
    let mut s = state();
    s.edit[0] = "nonsense".into();
    s.commit_row(0);
    let err = s.start().unwrap_err();
    assert!(err.contains("need fixing"), "{err}");
}

#[test]
fn starting_without_an_executor_is_refused_rather_than_panicking() {
    let mut s = state();
    let err = s.start().unwrap_err();
    assert!(err.contains("executor"), "{err}");
}

#[test]
fn the_shell_running_benchmark_asks_for_confirmation_first() {
    let mut s = state();
    let index = atlas_plugin::registry::all()
        .iter()
        .position(|d| d.needs_confirmation)
        .expect("the agentic benchmark requires confirmation");
    s.select(index);
    s.view = View::Params;
    s.on_key(key(KeyCode::Char('s')), BenchSub::Suite);
    assert!(s.confirm_open, "s must open the consent gate, not start");
    assert_ne!(s.view, View::Run);
    // Anything other than `y` backs out.
    s.on_key(key(KeyCode::Char('n')), BenchSub::Suite);
    assert!(!s.confirm_open);
    assert_ne!(s.view, View::Run);
}

#[test]
fn a_benchmark_without_side_effects_does_not_ask() {
    let mut s = state();
    let index = atlas_plugin::registry::all()
        .iter()
        .position(|d| !d.needs_confirmation)
        .expect("most benchmarks need no confirmation");
    s.select(index);
    s.view = View::Params;
    // No executor, so it refuses — but it must refuse for that reason, not by
    // opening a consent gate it does not need.
    s.on_key(key(KeyCode::Char('s')), BenchSub::Suite);
    assert!(!s.confirm_open);
}

#[test]
fn list_navigation_is_clamped_to_the_registry() {
    let mut s = state();
    let n = atlas_plugin::registry::all().len();
    for _ in 0..n + 5 {
        s.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    }
    assert_eq!(s.selected, n - 1);
    for _ in 0..n + 5 {
        s.on_key(key(KeyCode::Char('k')), BenchSub::Suite);
    }
    assert_eq!(s.selected, 0);
}

#[test]
fn enter_and_esc_walk_the_three_views() {
    let mut s = state();
    assert_eq!(s.view, View::List);
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    assert_eq!(s.view, View::Params);
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert_eq!(s.view, View::List);
}

#[test]
fn the_glow_is_off_until_a_run_turns_it_on() {
    let s = state();
    assert!(!s.glow);
    assert!(!s.is_running());
}
