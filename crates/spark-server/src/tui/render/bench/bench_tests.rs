// SPDX-License-Identifier: AGPL-3.0-only

//! The Benchmarks section's shared pieces and its three panes.
//!
//! A stored run and a live one are the same `BenchmarkResult`, so the table,
//! the tiles and the verdict are asserted once here rather than twice — but
//! the assertions that matter are the ones about what the pane REFUSES to say:
//! a measurement that gated nothing must not render a green PASS, and a run
//! that cannot know its total must not draw a full bar.

use std::time::Duration;

use atlas_plugin::{
    BenchmarkResult, Cell, CellStyle, Column, LogLevel, LogLine, ResultTable, RunRecord, RunSource,
    Stat, Verdict, VerdictKind,
};

use super::*;
use crate::tui::app::{BenchSub, Section};
use crate::tui::bench_state::View;
use crate::tui::render::harness::{draw_into, has, screen};

fn table(rows: usize) -> ResultTable {
    let mut t = ResultTable::new(
        "CONCURRENCY",
        vec![Column::left("conc", 8), Column::right("tok/s", 10)],
    );
    for i in 0..rows {
        t.push(vec![
            Cell::new(format!("c{i}")),
            Cell::styled(format!("{}.0", i * 3), CellStyle::Accent),
        ]);
    }
    t
}

fn bench_app() -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Benchmarks;
    a
}

#[test]
fn the_results_table_scrolls_its_rows_and_keeps_its_header_in_place() {
    let t = table(60);
    let rows = draw_into(80, 20, |f, area| draw_table(f, &t, 30, area));
    assert!(has(&rows, "CONCURRENCY ─ 60 rows"), "{rows:#?}");
    assert!(has(&rows, "conc"), "the header does not scroll:\n{rows:#?}");
    assert!(has(&rows, "c30"), "{rows:#?}");
    assert!(!has(&rows, "c0 "), "the rows above are gone:\n{rows:#?}");
}

#[test]
fn a_scroll_past_the_last_row_still_shows_the_last_page() {
    // The wheel is clamped by the app, but a stored scroll from a longer table
    // must not leave the pane blank.
    let t = table(60);
    let rows = draw_into(80, 20, |f, area| draw_table(f, &t, 10_000, area));
    assert!(has(&rows, "c59"), "{rows:#?}");
}

#[test]
fn a_table_with_no_rows_still_draws_its_frame_and_says_zero() {
    let t = table(0);
    let rows = draw_into(80, 20, |f, area| draw_table(f, &t, 0, area));
    assert!(has(&rows, "CONCURRENCY ─ 0 rows"), "{rows:#?}");
    assert!(has(&rows, "tok/s"), "the columns are still named");
}

#[test]
fn the_stat_tiles_carry_the_value_and_its_unit() {
    let stats = vec![
        Stat::new("wall", "6430.9", "s"),
        Stat::new("bfcl", "85.53", "%").with_style(CellStyle::Good),
        Stat::new("ttft p50", "1451", "ms"),
    ];
    let rows = draw_into(90, 5, |f, area| draw_stats(f, &stats, area));
    assert!(has(&rows, "WALL"), "{rows:#?}");
    assert!(has(&rows, "6430.9 s"));
    assert!(has(&rows, "BFCL"));
    assert!(has(&rows, "85.53 %"));
    assert!(has(&rows, "TTFT P50"));
}

#[test]
fn an_empty_summary_draws_no_tiles_at_all() {
    let rows = draw_into(90, 5, |f, area| draw_stats(f, &[], area));
    assert!(
        rows.iter().all(|r| r.is_empty()),
        "an empty tile row is blank, not an empty box:\n{rows:#?}"
    );
}

#[test]
fn a_measurement_that_gated_nothing_is_never_rendered_as_a_pass() {
    for (verdict, label, absent) in [
        (Verdict::pass("wall 6430s ≤ 7438s"), " PASS ", " FAIL "),
        (Verdict::fail("bfcl 82.1 < 83.64"), " FAIL ", " PASS "),
        (Verdict::info("swept 8 concurrencies"), " INFO ", " PASS "),
    ] {
        let rows = draw_into(80, 3, |f, area| {
            f.render_widget(
                ratatui::widgets::Paragraph::new(verdict_line(&verdict)),
                area,
            );
        });
        assert!(has(&rows, label), "{rows:#?}");
        assert!(!has(&rows, absent), "{rows:#?}");
        assert!(has(&rows, &verdict.reason), "the reason is stated");
    }
}

#[test]
fn the_origin_badge_reserves_brand_green_for_first_party_benchmarks() {
    let official = atlas_plugin::PluginMetadata {
        official: true,
        ..*crate::tui::render::tests::app().bench.plugin_metadata()
    };
    let community = atlas_plugin::PluginMetadata {
        official: false,
        ..official
    };
    assert_eq!(origin_badge(&official).content, " OFFICIAL ");
    assert_eq!(origin_badge(&community).content, " COMMUNITY ");
    assert_ne!(
        origin_badge(&official).style,
        origin_badge(&community).style,
        "the trust signal is the colour, not only the word"
    );
}

#[test]
fn the_plugin_block_names_who_wrote_the_benchmark_and_where_to_report_it() {
    let meta = crate::tui::render::tests::app().bench.plugin_metadata();
    let lines = metadata_lines(meta);
    let text: String = lines
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
        .collect();
    assert!(text.contains(&format!("v{}", meta.version)), "{text}");
    for (label, value) in meta.rows() {
        assert!(text.contains(label), "missing label {label}: {text}");
        assert!(text.contains(value), "missing value {value}: {text}");
    }
}

/// The live run pane.
mod run {
    use super::*;

    fn running(frame: BenchmarkResult) -> App {
        let mut a = bench_app();
        a.bench.view = View::Run;
        a.bench.status = "isl 1024 · conc 8".into();
        a.bench.progress = frame.progress;
        a.bench.frame = Some(frame);
        a
    }

    #[test]
    fn a_run_with_a_known_total_reports_a_fraction_and_a_percentage() {
        let a = running(
            BenchmarkResult::running("sweep", Duration::from_secs(90)).with_progress(25, 100),
        );
        let rows = screen(&a, 160, 48);
        assert!(has(&rows, "25/100  25%"), "{rows:#?}");
        assert!(has(&rows, "isl 1024 · conc 8"), "the phase is named");
    }

    #[test]
    fn a_run_that_cannot_know_its_total_says_so_rather_than_drawing_a_full_bar() {
        let a = running(BenchmarkResult::running(
            "provisioning",
            Duration::from_secs(3),
        ));
        let rows = screen(&a, 160, 48);
        assert!(!has(&rows, "100%"), "{rows:#?}");
        // Nothing is executing in a test, so the pane is honestly idle.
        assert!(has(&rows, "idle"), "{rows:#?}");
        assert!(has(&rows, "Esc back to suite"));
    }

    #[test]
    fn a_finished_run_shows_its_tiles_its_table_and_its_verdict_together() {
        let a = running(
            BenchmarkResult::completed("done", Duration::from_secs(6430))
                .with_summary(vec![Stat::new("wall", "6430.9", "s")])
                .with_table(table(12))
                .with_verdict(Verdict::info("swept 8 concurrencies")),
        );
        let rows = screen(&a, 160, 48);
        assert!(has(&rows, "WALL"), "{rows:#?}");
        assert!(has(&rows, "6430.9 s"));
        assert!(has(&rows, "CONCURRENCY ─ 12 rows"));
        assert!(has(&rows, " INFO "));
    }

    #[test]
    fn the_run_log_keeps_the_newest_lines_and_wraps_them_inside_the_panel() {
        let mut a = running(BenchmarkResult::running("sweep", Duration::from_secs(1)));
        for i in 0..40 {
            a.bench.log.push_back(LogLine::info(format!("entry-{i}")));
        }
        a.bench.log.push_back(LogLine {
            level: LogLevel::Warn,
            text: format!(
                "the endpoint answered unexpectedly — {}",
                "detail ".repeat(30)
            ),
        });
        let rows = screen(&a, 120, 44);
        assert!(has(&rows, "LOG ─ 41 lines"), "{rows:#?}");
        assert!(
            has(&rows, "the endpoint answered unexpectedly"),
            "the newest entry is on screen:\n{rows:#?}"
        );
        assert!(!has(&rows, "entry-0 "), "the oldest is not:\n{rows:#?}");
    }
}

/// The History pane, which reads the same frames back off disk.
mod history {
    use super::*;

    fn record(id: &str, kind: VerdictKind) -> RunRecord {
        RunRecord {
            schema: 1,
            run_id: format!("run-{id}"),
            benchmark_id: id.to_string(),
            benchmark_name: id.to_string(),
            recorded_at: 0,
            target_url: "http://127.0.0.1:8870".into(),
            target_model: "nvidia/Qwen3.6-27B-NVFP4".into(),
            params: Default::default(),
            source: RunSource::default(),
            atlas_version: "1.0.0".into(),
            frame: BenchmarkResult::completed("done", Duration::from_secs(6430))
                .with_summary(vec![Stat::new("wall", "6430.9", "s")])
                .with_table(table(9))
                .with_verdict(Verdict {
                    kind,
                    reason: "recorded".into(),
                }),
        }
    }

    fn with_history(records: Vec<RunRecord>, row: usize) -> App {
        let mut a = bench_app();
        a.bench_sub = BenchSub::History;
        a.bench.history = records;
        a.bench.history_row = row;
        a
    }

    #[test]
    fn a_stored_run_renders_with_the_same_tiles_and_table_the_live_pane_drew() {
        let a = with_history(vec![record("concurrency-sweep", VerdictKind::Pass)], 0);
        let rows = screen(&a, 160, 48);
        assert!(has(&rows, "RUNS ─ 1"), "{rows:#?}");
        assert!(has(&rows, "concurrency-sweep"));
        assert!(has(&rows, "✓"), "the verdict is marked in the list");
        assert!(has(&rows, "WALL"), "{rows:#?}");
        assert!(has(&rows, "CONCURRENCY ─ 9 rows"));
        assert!(has(&rows, "6430s"), "the elapsed time is on the header");
    }

    #[test]
    fn a_failed_run_is_marked_differently_from_one_that_passed() {
        let a = with_history(
            vec![
                record("pass-run", VerdictKind::Pass),
                record("fail-run", VerdictKind::Fail),
            ],
            1,
        );
        let rows = screen(&a, 160, 48);
        assert!(has(&rows, "✗"), "{rows:#?}");
        assert!(has(&rows, "RUNS ─ 2"));
        assert!(has(&rows, " FAIL "), "the detail pane agrees:\n{rows:#?}");
    }

    #[test]
    fn the_list_scrolls_to_keep_the_selected_run_on_screen() {
        let records: Vec<RunRecord> = (0..60)
            .map(|i| record(&format!("run-{i:02}"), VerdictKind::Info))
            .collect();
        let a = with_history(records, 59);
        let rows = screen(&a, 160, 48);
        assert!(has(&rows, "run-59"), "{rows:#?}");
        assert!(!has(&rows, "run-00"), "{rows:#?}");
    }

    #[test]
    fn the_history_pane_survives_narrow_and_short_terminals() {
        let a = with_history(vec![record("sweep", VerdictKind::Pass)], 0);
        for (w, h) in [(20u16, 4u16), (20, 8), (40, 12), (200, 3)] {
            assert_eq!(screen(&a, w, h).len(), h as usize, "{w}x{h}");
        }
    }
}

#[test]
fn every_benchmarks_pane_survives_narrow_and_short_terminals() {
    for view in [View::List, View::Params, View::Run] {
        let mut a = bench_app();
        a.bench.view = view;
        a.bench.frame = Some(
            BenchmarkResult::completed("done", Duration::from_secs(1))
                .with_summary(vec![Stat::new("wall", "1.0", "s")])
                .with_table(table(20)),
        );
        a.bench
            .log
            .push_back(LogLine::error("something went wrong".repeat(20)));
        for (w, h) in [(20u16, 4u16), (20, 8), (20, 40), (40, 12), (200, 3)] {
            assert_eq!(screen(&a, w, h).len(), h as usize, "{w}x{h}");
        }
    }
}

/// ★ The Suite list must scroll the selection into view.
///
/// It rendered from index 0 unconditionally. Each entry occupies four rows, so
/// an 80x24 terminal showed about four of the seven benchmarks and `j` past
/// them moved a cursor nobody could see — the detail pane changing was the only
/// feedback. Both sibling lists (bench/history.rs, library/list.rs) already
/// computed an offset.
#[test]
fn the_suite_list_scrolls_the_selection_into_view() {
    let all = atlas_plugin::registry::all();
    assert!(
        all.len() > 4,
        "only meaningful once the suite outgrows one 80x24 screen"
    );
    let last = all.len() - 1;

    // 80x24 is the classic terminal size, and the one the bug was found on.
    let mut a = bench_app();
    a.bench.view = View::List;
    a.bench.selected = last;
    let rows = screen(&a, 80, 24);
    assert!(
        has(&rows, all[last].name),
        "the selected benchmark must be on screen:\n{rows:#?}"
    );

    // And selecting the top scrolls back, rather than leaving the list parked.
    a.bench.selected = 0;
    let rows = screen(&a, 80, 24);
    assert!(
        has(&rows, all[0].name),
        "selecting the first entry must scroll back to it:\n{rows:#?}"
    );
}
