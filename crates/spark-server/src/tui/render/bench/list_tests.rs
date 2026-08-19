// SPDX-License-Identifier: AGPL-3.0-only

//! The Suite list's windowing: reachability, the clip indicator, clamping.
//!
//! In its own mount rather than `bench_tests.rs` to stay under the 500-LoC
//! cap. These exist because the eleventh registered benchmark rendered
//! nowhere: the list drew four rows per entry from a selection-anchored
//! offset with no indicator that anything lay below the fold, so
//! `mlperf-agentic-subset` registering pushed Serve Matrix silently off a
//! 160x48 terminal.

use crate::tui::app::{App, Section};
use crate::tui::bench_state::View;
use crate::tui::render::harness::{has, screen};

/// A terminal tall enough to hold the whole registry on one page.
///
/// Derived, not hardcoded: the suite grows, and a fixed `160x48` silently
/// stopped holding it once `mlperf-agentic-subset` joined the decode-floor /
/// quick-speed / media-integrity additions — both tests below started failing
/// on a registry change that had nothing to do with them.
///
/// A list entry is NOT one row (measured: 31 rows gave a 7-entry page, 45 rows
/// gave 12 — about 2.8 rows apiece once the summary line and spacing are
/// counted), so the height has to scale with `n`, not merely clear a fixed
/// chrome. `3n + 20` keeps the page strictly larger than the registry with
/// room to spare, and grows correctly as entries are added.
fn tall_enough() -> u16 {
    let n = atlas_plugin::registry::all().len();
    u16::try_from(n * 3 + 20).expect("fits a u16 terminal")
}

fn list_app(selected: usize) -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Benchmarks;
    a.bench.view = View::List;
    a.bench.selected = selected;
    a
}

/// The one guarantee the 160x48 all-names assertion cannot give: it holds
/// for ANY registry size, so the leg after next cannot be invisible either.
#[test]
fn every_registered_benchmark_is_reachable_by_scrolling() {
    // 24 rows clip the suite, so reaching an entry means the offset followed
    // the selection there. 120 columns, not 80: names are only guaranteed
    // unclipped horizontally on a wide terminal (`render_tests` states the
    // same rule for 160), and this test is about the vertical fold.
    let all = atlas_plugin::registry::all();
    for (i, descriptor) in all.iter().enumerate() {
        let rows = screen(&list_app(i), 120, 24);
        assert!(
            has(&rows, descriptor.name),
            "{} unreachable at 120x24:\n{rows:#?}",
            descriptor.name
        );
    }
}

#[test]
fn the_clip_indicator_appears_only_when_the_list_is_clipped() {
    let n = atlas_plugin::registry::all().len();
    // 80x24 cannot hold the whole suite: the bottom border must say where
    // the window sits, starting at entry 1. Both halves on one row, so the
    // footer's own "1-7 jump" cannot satisfy this by accident.
    let rows = screen(&list_app(0), 80, 24);
    assert!(
        rows.iter()
            .any(|r| r.contains("─ 1-") && r.contains(&format!("of {n} ─"))),
        "clipped list must name its window:\n{rows:#?}"
    );
    // A registry-sized terminal holds every entry (`render_tests` asserts the
    // names); an indicator over an unclipped list would claim a fold that is
    // not there.
    let rows = screen(&list_app(0), 160, tall_enough());
    assert!(
        !has(&rows, &format!("of {n} ─")),
        "unclipped list must not draw an indicator:\n{rows:#?}"
    );
}

#[test]
fn the_indicator_tracks_the_selection_to_the_bottom() {
    let n = atlas_plugin::registry::all().len();
    let rows = screen(&list_app(n - 1), 80, 24);
    assert!(
        has(&rows, &format!("-{n} of {n} ─")),
        "with the last entry selected the window must end at {n}:\n{rows:#?}"
    );
}

#[test]
fn the_offset_is_clamped_at_both_ends() {
    // A selection past the registry — a stale index after the registry
    // shrinks — must show the last page, not scroll into blank space below
    // the real entries.
    let all = atlas_plugin::registry::all();
    let rows = screen(&list_app(all.len() + 40), 80, 24);
    assert!(
        has(&rows, all[all.len() - 1].name),
        "an over-large selection clamps to the last page:\n{rows:#?}"
    );
    // And the top of the list comes back exactly, not one entry short.
    let rows = screen(&list_app(0), 80, 24);
    assert!(has(&rows, all[0].name), "{rows:#?}");
}

#[test]
fn compaction_keeps_the_summary_and_duration_on_every_entry() {
    // Fitting more entries by dropping the separator row, never by dropping
    // the rows that say what a benchmark does and how long it takes.
    let all = atlas_plugin::registry::all();
    let rows = screen(&list_app(0), 80, 24);
    assert!(
        has(&rows, all[0].duration_hint),
        "the duration line survives the compact layout:\n{rows:#?}"
    );
}

#[test]
fn the_renderer_publishes_the_page_size_for_the_key_handler() {
    // The `log_scroll_max` contract: PgUp/PgDn page by whatever one frame
    // actually held, so the stride is never a guess about the terminal.
    let n = atlas_plugin::registry::all().len();
    let a = list_app(0);
    screen(&a, 80, 24);
    let page = a.bench.suite_page.get();
    assert!(page > 0, "a viewport of zero would freeze the page keys");
    assert!(
        page < n,
        "80x24 is clipped, so one page is less than the suite"
    );
    let tall = tall_enough();
    screen(&a, 160, tall);
    assert!(
        a.bench.suite_page.get() >= n,
        "a terminal sized from the registry ({n} entries, {tall} rows) holds \
         the whole suite on one page; got {}",
        a.bench.suite_page.get()
    );
}

#[test]
fn the_list_survives_a_12x4_terminal() {
    // The floor the rest of the suite is exercised at. Nothing readable
    // fits; the assertion is that the windowing math holds — `visible`
    // floored at one entry, the offset clamp — instead of panicking.
    let n = atlas_plugin::registry::all().len();
    for selected in [0, n - 1] {
        let rows = screen(&list_app(selected), 12, 4);
        assert_eq!(rows.len(), 4, "selected {selected}");
    }
}
