// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the [`super`] input reducer.

use super::*;

/// The ⇥ order must contain the subsection rows, in the order the sidebar draws
/// them. This is the regression: the traversal list was top-level-only, so
/// Main ▸ Kernels and Terminal ▸ Chat could not be reached with Tab at all.
#[test]
fn nav_rows_include_subsections_in_sidebar_order() {
    let labels: Vec<String> = App::nav_rows()
        .iter()
        .map(|(s, i)| match s.subs().get(*i) {
            Some(sub) => format!("{}/{}", s.label(), sub),
            None => s.label().to_string(),
        })
        .collect();
    assert_eq!(
        labels,
        [
            "Main/Overview",
            "Main/Kernels",
            "Stats",
            "Network",
            "Library",
            "Benchmarks/Suite",
            "Benchmarks/History",
            "Terminal/Ops",
            "Terminal/Chat",
        ]
    );
}

/// The digit keys and `Section::ALL` must stay in step. Benchmarks was inserted
/// BEFORE Terminal, which moved Terminal from `5` to `6` — a mismatch here is
/// invisible until someone presses a number and lands in the wrong place.
#[test]
fn digit_keys_match_the_sidebar_order() {
    use clap::Parser;
    use crossterm::event::{KeyCode, KeyEvent};
    let mut app = App::new(crate::cli::ServeArgs::parse_from(["spark", "some/model"]));
    for (i, section) in Section::ALL.iter().enumerate() {
        let digit = char::from_digit(i as u32 + 1, 10).expect("<=9 sections");
        app.on_key(KeyEvent::from(KeyCode::Char(digit)));
        assert_eq!(
            app.section,
            *section,
            "key {digit} must select {}",
            section.label()
        );
    }
}

/// Every section the sidebar draws must have a content renderer; the match in
/// `render::draw` is exhaustive, so this is really a guard on `subs()` staying
/// consistent with what the Benchmarks section actually implements.
#[test]
fn benchmarks_declares_both_of_its_subsections() {
    assert_eq!(Section::Benchmarks.subs(), &["Suite", "History"]);
    assert_eq!(
        Section::Benchmarks.icon().chars().count(),
        1,
        "sidebar is 1 cell wide"
    );
}

/// A section without subsections must still contribute exactly one stop, or ⇥
/// would silently skip it.
#[test]
fn every_section_is_reachable() {
    let rows = App::nav_rows();
    for s in Section::ALL {
        assert!(
            rows.iter().any(|(r, _)| *r == s),
            "{} unreachable",
            s.label()
        );
    }
}

/// Chat scrollback contract, mirroring the Main log pane: `None` follows the tip,
/// and scrolling back down to (or past) the bottom restores follow rather than
/// parking at `Some(0)`, which would freeze the view one row off the live tip.
#[test]
fn chat_scroll_returns_to_follow_at_the_bottom() {
    let mut c = crate::tui::chat::ChatState::default();
    assert_eq!(c.scroll, None, "starts following");
    c.scroll_by(3);
    assert_eq!(c.scroll, Some(3));
    c.scroll_by(-1);
    assert_eq!(c.scroll, Some(2));
    c.scroll_by(-5); // overshoot past the bottom
    assert_eq!(c.scroll, None, "overshoot resumes follow");
    c.scroll_by(10);
    c.follow();
    assert_eq!(c.scroll, None);
}

#[test]
fn the_watchdog_command_toggles_the_running_run_not_a_process_global() {
    // `/watchdog on|off` used to store into a `OnceLock<bool>` shared by every
    // run in the process — and, being a `OnceLock`, the boot value could not be
    // changed at all once set. It now reaches the run's own levers through the
    // handle `serve` publishes, so the toggle is observable on exactly the
    // `Arc` the scheduler is reading.
    use clap::Parser as _;
    let mut app = App::new(crate::cli::ServeArgs::parse_from(["spark", "some/model"]));
    let levers = std::sync::Arc::new(crate::scheduler::levers::SchedLevers::from_env());
    let other = std::sync::Arc::new(crate::scheduler::levers::SchedLevers::from_env());
    app.run = Some(crate::tui::RunHandles {
        levers: levers.clone(),
        snapshot: std::sync::Arc::new(crate::scheduler::snapshot::SnapshotCell::default()),
    });

    crate::tui::commands::execute("/watchdog on", &mut app);
    assert!(levers.loop_watchdog(), "the attached run is armed");
    assert!(!other.loop_watchdog(), "another run is untouched");

    crate::tui::commands::execute("/watchdog off", &mut app);
    assert!(!levers.loop_watchdog());
}

#[test]
fn the_watchdog_command_says_so_when_no_run_is_attached() {
    // Before the scheduler starts there is nothing to toggle. Saying so beats
    // the old behaviour, where the store succeeded against a global the run
    // had not read yet.
    use clap::Parser as _;
    let mut app = App::new(crate::cli::ServeArgs::parse_from(["spark", "some/model"]));
    crate::tui::commands::execute("/watchdog on", &mut app);
    assert!(
        app.ops.output.iter().any(|l| l.contains("no run yet")),
        "got {:?}",
        app.ops.output
    );
}

#[test]
fn a_no_argument_boot_opens_the_library_rather_than_an_empty_main() {
    // Main has nothing to show without a model: a 0/12 checklist and a LOADING
    // pill for a load that is not running. The Library is the only screen that
    // can move the user forward.
    use clap::Parser as _;
    let mut args = crate::cli::ServeArgs::parse_from(["spark", "m"]);
    args.model = None;
    let app = App::new(args);
    assert!(app.awaiting_model);
    assert_eq!(app.section, Section::Library);
}

#[test]
fn a_boot_with_a_model_still_opens_main() {
    use clap::Parser as _;
    let args = crate::cli::ServeArgs::parse_from(["spark", "org/m"]);
    let app = App::new(args);
    assert!(!app.awaiting_model, "a model was named");
    assert_eq!(app.section, Section::Main);
}

#[test]
fn launching_from_the_library_stops_claiming_there_is_no_model() {
    // Otherwise the pill stays on NO MODEL through the whole load.
    use clap::Parser as _;
    let mut args = crate::cli::ServeArgs::parse_from(["spark", "m"]);
    args.model = None;
    let mut app = App::new(args);
    assert!(app.awaiting_model);
    // No host attached, so the launch is refused — and the flag must survive
    // that, because nothing was started.
    app.launch_selected_recipe();
    assert!(app.awaiting_model, "a refused launch loaded nothing");
}
