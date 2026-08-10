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

pub(super) fn app() -> App {
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

pub(super) fn render(app: &App, width: u16, height: u16) -> String {
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

/// The Library panes must render at realistic and hostile sizes.
mod library {
    use super::*;
    use crate::tui::lib_state::View as LibView;

    fn with_rows() -> App {
        let mut app = app();
        app.section = Section::Library;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
        let recipe = crate::recipe::Recipe::parse(
            "qwen3.6/flagship",
            &std::fs::read_to_string(path).expect("fixture"),
        )
        .expect("parses");
        app.library = vec![crate::tui::data::library::LibraryEntry {
            id: recipe.model.clone(),
            snapshot_dir: Default::default(),
            size_bytes: 34_900_000_000,
            has_weights: true,
            model_type: "qwen3_6_moe".into(),
            quant: "fp8".into(),
            layers: 40,
            hidden: 4096,
            heads: 32,
            experts: 128,
            context: 65536,
            optimized: true,
        }];
        app.lib.index = crate::recipe::fetch::Index {
            recipes: vec![recipe],
            ..Default::default()
        };
        app.lib.rebuild(&app.library);
        app
    }

    /// The defect this test exists for: the panel title reported a row count
    /// that disagreed with the rows drawn beneath it, because the count and the
    /// list were read from different places.
    #[test]
    fn the_title_agrees_with_the_rows_it_draws() {
        let app = with_rows();
        let out = render(&app, 200, 50);
        assert!(out.contains("MODELS"), "the panel is drawn");
        assert!(
            !out.contains("MODELS ─ 0"),
            "a populated list must not claim 0 rows:\n{out}"
        );
        assert!(
            !out.contains("no models or recipes yet"),
            "the empty hint must not appear beside real rows:\n{out}"
        );
        assert!(out.contains("Qwen3.6-35B-A3B-FP8"), "the row is drawn");
    }

    #[test]
    fn the_empty_state_says_what_to_do() {
        let mut app = app();
        app.section = Section::Library;
        let out = render(&app, 200, 50);
        assert!(out.contains("press r to fetch recipes"), "{out}");
    }

    #[test]
    fn the_config_pane_renders_and_shows_the_command() {
        let mut app = with_rows();
        app.lib.open_cards().expect("opens");
        app.lib.open_config().expect("opens");
        assert_eq!(app.lib.view, LibView::Config);
        let out = render(&app, 200, 50);
        assert!(out.contains("SETTINGS"), "{out}");
        assert!(out.contains("spark serve"), "the command preview: {out}");
    }

    /// The cards pane: the choice between sibling recipes, with the room the
    /// list row never had.
    #[test]
    fn the_cards_pane_shows_the_recipe_and_its_rationale() {
        let mut app = with_rows();
        app.lib.open_cards().expect("opens");
        assert_eq!(app.lib.view, LibView::Cards);
        let out = render(&app, 200, 50);
        assert!(out.contains("recipe"), "the header counts them: {out}");
        // The description is the measured rationale — the reason this pane
        // exists rather than a one-line row.
        assert!(out.contains("FLAGSHIP"), "the recipe's own text: {out}");
        assert!(out.contains("configure and start"), "{out}");
    }

    /// A one-recipe model still gets a card, by explicit request.
    #[test]
    fn one_recipe_still_renders_a_card() {
        let mut app = with_rows();
        assert_eq!(app.lib.cards().len(), 1, "the fixture has one");
        app.lib.open_cards().expect("opens");
        let out = render(&app, 200, 50);
        assert!(
            out.contains("1 recipe"),
            "singular, not \"1 recipes\": {out}"
        );
    }

    /// Narrow and short terminals are where layout maths underflows.
    #[test]
    fn the_library_survives_hostile_sizes() {
        let app = with_rows();
        for (w, h) in [(40, 12), (60, 20), (80, 24), (120, 30), (240, 80)] {
            let _ = render(&app, w, h);
        }
        let mut cards = with_rows();
        cards.lib.open_cards().expect("opens");
        for (w, h) in [(40, 12), (60, 20), (80, 24), (240, 80)] {
            let _ = render(&cards, w, h);
        }
        let mut config = with_rows();
        config.lib.open_cards().expect("opens");
        config.lib.open_config().expect("opens");
        for (w, h) in [(40, 12), (60, 20), (80, 24), (240, 80)] {
            let _ = render(&config, w, h);
        }
    }
}

/// Frames are drawn into a LIVE terminal, one after another, not into a fresh
/// buffer each time. The Library's first frame has no rows (the local scan and
/// the recipe cache both land a tick later), so the empty state is genuinely
/// shown and then replaced — and a stale title left behind by that transition
/// is exactly what a single-frame test cannot see.
#[test]
fn the_library_leaves_nothing_behind_when_it_fills_in() {
    let mut terminal = Terminal::new(TestBackend::new(200, 50)).expect("backend");

    // Frame 1: empty, as on first entry.
    let mut app = app();
    app.section = Section::Library;
    terminal.draw(|f| draw(f, &app)).expect("draw");

    // Frame 2: populated, as one tick later.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let recipe = crate::recipe::Recipe::parse(
        "qwen3.6/flagship",
        &std::fs::read_to_string(path).expect("fixture"),
    )
    .expect("parses");
    app.lib.index = crate::recipe::fetch::Index {
        recipes: vec![recipe],
        ..Default::default()
    };
    app.lib.rebuild(&[]);
    terminal.draw(|f| draw(f, &app)).expect("draw");

    let out: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect();
    assert!(
        !out.contains("no models or recipes yet"),
        "the empty hint survived into the populated frame:\n{out}"
    );
    assert!(
        !out.contains("MODELS ─ 0"),
        "the empty title survived into the populated frame:\n{out}"
    );
    assert!(out.contains("MODELS ─ 1"), "the new title is drawn:\n{out}");
}

/// The pre-flight modal and the run log, which are what the user actually
/// reads when a benchmark is about to go wrong.
mod preflight {
    use super::*;
    use crate::tui::bench_preflight::Preflight;

    fn on_params() -> App {
        let mut app = app();
        app.section = Section::Benchmarks;
        app.bench_sub = BenchSub::Suite;
        app.bench.view = View::Params;
        app
    }

    #[test]
    fn the_checking_modal_shows_a_spinner_and_the_target() {
        let mut app = on_params();
        app.bench.preflight = Some(Preflight::pending());
        let out = render(&app, 200, 50);
        assert!(out.contains("CHECKING THE ENDPOINT"), "{out}");
        assert!(out.contains("known-answer"), "says what it is doing: {out}");
    }

    /// The reason must be READABLE — the whole bug was a concern clipped at the
    /// panel edge, so the half that mattered never reached the screen.
    #[test]
    fn a_concern_is_wrapped_not_truncated() {
        let mut app = on_params();
        let long = "http://127.0.0.1:8888 is serving \"nvidia/Qwen3.6-27B-NVFP4\", which did not \
                    answer as expected (recall answered nothing). This benchmark may be aimed at \
                    a different model, or the checkpoint may be a base (non-instruct) one — the \
                    run is still valid, but read the numbers with that in mind.";
        app.bench.preflight = Some(Preflight::with_concern(long.to_string()));
        let out = render(&app, 200, 50);
        assert!(out.contains("BEFORE YOU START"), "{out}");
        assert!(out.contains("run it anyway"), "offers to proceed: {out}");
        assert!(out.contains("back to the form"), "offers to go back: {out}");
        // The tail of the message must survive the wrap.
        assert!(
            out.contains("with that in mind"),
            "the end of the reason was lost:\n{out}"
        );
    }

    /// A long log line must wrap inside its panel rather than being cut.
    #[test]
    fn run_log_lines_wrap_inside_the_panel() {
        let mut app = on_params();
        app.bench.view = View::Run;
        let tail = "and this tail must still be on screen";
        app.bench.log.push_back(atlas_plugin::LogLine {
            level: atlas_plugin::LogLevel::Warn,
            text: format!(
                "http://127.0.0.1:8888 is serving a model that did not answer as expected, \
                 which usually means the benchmark is aimed somewhere else — {tail}"
            ),
        });
        let out = render(&app, 120, 40);
        assert!(
            out.contains(tail),
            "the line was truncated, not wrapped:\n{out}"
        );
    }
}

#[test]
fn the_benchmark_detail_pane_says_when_the_measurement_last_changed() {
    // The same question the Library answers for a recipe, asked of a
    // benchmark: two runs are only comparable if the definition between them
    // did not move. Without this the reader has no way to know.
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench.view = View::List;
    let out = render(&a, 200, 50);
    assert!(out.contains("Updated"), "the row is drawn:\n{out}");
    let d = a
        .bench
        .descriptor()
        .expect("a benchmark is selected")
        .updated;
    assert!(out.contains(d), "and carries the date {d}:\n{out}");
}

/// An App whose Library has exactly one row for `model`, so the per-row
/// progress line is actually reached by the renderer.
fn app_with_row(model: &str) -> App {
    let mut a = app();
    a.section = Section::Library;
    a.library = vec![crate::tui::data::library::LibraryEntry {
        id: model.to_string(),
        snapshot_dir: Default::default(),
        size_bytes: 1024,
        has_weights: false,
        model_type: "qwen3_6_moe".into(),
        quant: "nvfp4".into(),
        layers: 40,
        hidden: 4096,
        heads: 32,
        experts: 128,
        context: 65536,
        optimized: false,
    }];
    a.lib.rebuild(&a.library);
    a
}

/// A 20 GB download spends its first minutes under one percent. Reported from
/// a real run: "I don't see anything happen … no downloading as far as I can
/// see" — the download was fine, the LINE was not.
#[test]
fn a_download_under_one_percent_still_looks_alive() {
    let mut a = app_with_row("org/big");
    let root = std::env::temp_dir().join("atlas-render-dl");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/big", root);
    {
        let job = a.download.job.as_mut().expect("a job");
        job.total = 19_800_000_000; // 19.8 GB
        job.done = 149_000_000; //  149 MB  =  0.75%
        job.rate_bps = 1_600_000.0;
    }
    let out = render(&a, 200, 50);

    // The bar must not be twelve empty cells: at 0.75% of 12, `round()` gives
    // ZERO filled, which is what made a working download look dead.
    assert!(
        out.contains('▓'),
        "some of the bar must be filled once bytes have moved:\n{out}"
    );
    // And the percentage must not read a flat "0%".
    assert!(
        !out.contains("  0%"),
        "sub-1% progress must not render as a bare 0%:\n{out}"
    );
    assert!(out.contains("0.8%"), "one decimal below 10%:\n{out}");
    // The rate is the field that proves bytes are moving; it must fit in the
    // list pane, which is only about half the terminal width.
    assert!(out.contains("MB/s"), "the rate must be visible:\n{out}");
}

#[test]
fn a_download_at_exactly_zero_bytes_shows_an_empty_bar_honestly() {
    // The min-one-cell rule applies only once something has moved — before
    // that, an empty bar is the truth.
    let mut a = app_with_row("org/big");
    let root = std::env::temp_dir().join("atlas-render-dl0");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/big", root);
    {
        let job = a.download.job.as_mut().expect("a job");
        job.total = 19_800_000_000;
        job.done = 0;
    }
    let out = render(&a, 200, 50);
    assert!(out.contains('░'), "the empty track is drawn:\n{out}");
    assert!(out.contains("0.0%"));
}
