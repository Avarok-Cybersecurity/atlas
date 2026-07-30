// SPDX-License-Identifier: AGPL-3.0-only

//! Render smoke tests over `TestBackend`.
//!
//! Layout code is where a TUI actually crashes: a `Rect` computed past the
//! frame, a `split` with more constraints than cells, a subtraction that
//! underflows on a narrow terminal. None of that is visible to `cargo check`,
//! and all of it takes the dashboard — and with it the server's foreground —
//! down at runtime. Rendering every section into a buffer at several sizes is
//! the cheapest thing that catches it.

use ratatui::Terminal;
use ratatui::backend::TestBackend;

use super::draw;
use crate::tui::app::{App, BenchSub, Section};
use crate::tui::bench_state::View;

fn app() -> App {
    use clap::Parser;
    let mut app = App::new(crate::cli::ServeArgs::parse_from([
        "spark",
        "nvidia/Qwen3.6-27B-NVFP4",
    ]));
    // `attach` needs a tokio handle; the Benchmarks panes must render without
    // one, which is exactly the pre-attach state a failed store discovery
    // leaves behind.
    app.bench.select(0);
    app
}

fn render(app: &App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("backend");
    terminal.draw(|f| draw(f, app)).expect("draw");
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect()
}

/// The sizes that matter: the wide layout, the narrow-sidebar layout
/// (width < 96), the short-header layout (height < 28), and a terminal small
/// enough that every `saturating_sub` in the tree is exercised.
const SIZES: [(u16, u16); 4] = [(160, 48), (100, 30), (80, 24), (40, 12)];

#[test]
fn every_section_renders_at_every_size() {
    for section in Section::ALL {
        for (w, h) in SIZES {
            let mut a = app();
            a.section = section;
            let out = render(&a, w, h);
            assert!(
                !out.is_empty(),
                "{} at {w}x{h} drew nothing",
                section.label()
            );
        }
    }
}

#[test]
fn every_benchmarks_view_renders_at_every_size() {
    for sub in [BenchSub::Suite, BenchSub::History] {
        for view in [View::List, View::Params, View::Run] {
            for (w, h) in SIZES {
                let mut a = app();
                a.section = Section::Benchmarks;
                a.bench_sub = sub;
                a.bench.view = view;
                let out = render(&a, w, h);
                assert!(!out.is_empty(), "bench view at {w}x{h} drew nothing");
            }
        }
    }
}

#[test]
fn the_suite_list_shows_the_benchmarks_and_their_provenance() {
    let mut a = app();
    a.section = Section::Benchmarks;
    let out = render(&a, 160, 48);
    for descriptor in atlas_plugin::registry::all() {
        // Names can wrap at narrow widths; at 160 columns they must be intact.
        assert!(out.contains(descriptor.name), "missing {}", descriptor.name);
    }
    assert!(out.contains("OFFICIAL"), "first-party badge is missing");
    assert!(out.contains("Avarok"), "author is missing");
}

#[test]
fn the_parameter_form_shows_every_field_plus_the_endpoint() {
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench.view = View::Params;
    let out = render(&a, 160, 48);
    for spec in &a.bench.specs {
        assert!(out.contains(spec.label), "missing field {}", spec.label);
    }
    assert!(out.contains("TARGET"));
    assert!(out.contains("START"), "the start key must be discoverable");
}

#[test]
fn the_confirmation_modal_says_what_it_will_do() {
    let mut a = app();
    a.section = Section::Benchmarks;
    let index = atlas_plugin::registry::all()
        .iter()
        .position(|d| d.needs_confirmation)
        .expect("one benchmark runs shell");
    a.bench.select(index);
    a.bench.view = View::Params;
    a.bench.confirm_open = true;
    let out = render(&a, 160, 48);
    assert!(out.contains("shell"), "the consent gate must name the risk");
    assert!(out.contains("sandbox"));
}

#[test]
fn the_glow_ring_is_titled_only_while_a_benchmark_runs() {
    let mut a = app();
    a.section = Section::Stats;
    assert!(
        !render(&a, 160, 48).contains("⏱"),
        "an idle ring carries no title"
    );
    a.bench.glow = true;
    let running = render(&a, 160, 48);
    assert!(
        running.contains("⏱"),
        "the run signal must follow you out of the Benchmarks section"
    );
    assert!(running.contains("Concurrency Sweep"));
}

#[test]
fn the_history_pane_says_so_when_there_is_nothing_to_show() {
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench_sub = BenchSub::History;
    let out = render(&a, 160, 48);
    assert!(out.contains("No runs recorded yet"));
    assert!(out.contains(".atlas/runs"), "say where they will appear");
}

#[test]
fn a_terminal_one_cell_wide_does_not_panic() {
    // Underflow guard: every layout in the tree subtracts from the width.
    for (w, h) in [(1, 1), (2, 3), (1, 40), (40, 1)] {
        let mut a = app();
        a.section = Section::Benchmarks;
        let _ = render(&a, w, h);
    }
}
