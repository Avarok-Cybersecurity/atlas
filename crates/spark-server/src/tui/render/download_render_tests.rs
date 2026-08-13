// SPDX-License-Identifier: AGPL-3.0-only

//! Render tests for the download surfaces: the per-row progress line, the
//! header chip, and the footer's stop hint.
//!
//! Split from `render_tests.rs` at the 500-LoC cap; the fixtures it shares
//! (`app`, `render`) stay in `render_tests.rs`, which is the SSOT for them.

use crate::tui::app::{App, Section};

use super::tests::{app, render};

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

/// The chip must be visible from EVERY section, not just the Library — the
/// whole point of the element. Reported as: switch away from the Library and a
/// 20 GB pull becomes invisible.
#[test]
fn a_download_is_visible_from_every_section() {
    for section in crate::tui::section::Section::ALL {
        let mut a = app_with_row("nvidia/Qwen3-80B-NVFP4");
        a.section = section;
        let root = std::env::temp_dir().join("atlas-render-chip");
        std::fs::create_dir_all(&root).ok();
        a.download.start("nvidia/Qwen3-80B-NVFP4", root);
        {
            let job = a.download.job.as_mut().expect("a job");
            job.total = 19_800_000_000;
            job.done = 8_300_000_000;
            job.rate_bps = 96_000_000.0;
        }
        let out = render(&a, 120, 32);
        assert!(
            out.contains("42%"),
            "{section:?}: the chip must carry the percentage:\n{out}"
        );
        assert!(
            out.contains("Qwen3-80B-NVFP4"),
            "{section:?}: the chip must name the model:\n{out}"
        );
    }
}

/// A percentage we cannot compute must never be faked. The Hub sometimes
/// reports no sizes; bytes moved is the honest substitute.
#[test]
fn an_unknown_total_shows_bytes_not_a_fake_percentage() {
    let mut a = app_with_row("org/big");
    a.section = crate::tui::section::Section::Stats;
    let root = std::env::temp_dir().join("atlas-render-chip2");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/big", root);
    {
        let job = a.download.job.as_mut().expect("a job");
        job.total = 0; // Hub reported nothing
        job.done = 3_400_000_000;
    }
    let out = render(&a, 120, 32);
    assert!(out.contains("3.2 GB"), "bytes moved:\n{out}");
    assert!(!out.contains("0%"), "never a fabricated percentage:\n{out}");
}

/// Stopping is not progress: the word appears and the pulse stops.
#[test]
fn a_cancelling_download_says_stopping_in_the_chip() {
    let mut a = app_with_row("org/big");
    a.section = crate::tui::section::Section::Network;
    let root = std::env::temp_dir().join("atlas-render-chip3");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/big", root);
    {
        let job = a.download.job.as_mut().expect("a job");
        job.total = 1_000;
        job.done = 500;
        job.cancelling = true;
    }
    let out = render(&a, 120, 32);
    assert!(out.contains("stopping"), "the chip must say so:\n{out}");
}

/// With nothing running the header must be byte-identical to before, and the
/// footer must not advertise a key that cannot act.
#[test]
fn no_download_means_no_chip_and_no_stop_hint() {
    let mut a = app_with_row("org/big");
    a.section = crate::tui::section::Section::Library;
    let out = render(&a, 120, 32);
    assert!(
        !out.contains("x stop"),
        "the stop hint is a false claim with nothing to stop:\n{out}"
    );
}
