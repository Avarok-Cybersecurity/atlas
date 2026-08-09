// SPDX-License-Identifier: AGPL-3.0-only

//! The Main tab: startup checklist, weight-load hero, READY strip, chips, the
//! log pane's line wrapping, and the Kernels table.
//!
//! Split from `main_tab.rs` only to stay under the repository's per-file cap.

use super::*;
use crate::tui::app::{MainSub, Section};
use crate::tui::data::kernels::{KernelRow, KernelTableModel, MissingKernel};
use crate::tui::progress::PhaseState;
use crate::tui::render::harness::{has, screen};

fn logline(msg: &str) -> Line<'static> {
    Line::from(vec![
        Span::raw("12:35:04 ".to_string()),
        Span::raw("INFO  ".to_string()),
        Span::raw("spark_model    ".to_string()),
        Span::raw(msg.to_string()),
    ])
}

fn text(line: &Line<'_>) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect()
}

#[test]
fn a_line_that_fits_is_left_alone() {
    let rows = wrap_line(logline("short"), 120);
    assert_eq!(rows.len(), 1);
    assert!(text(&rows[0]).ends_with("short"));
}

#[test]
fn a_long_line_wraps_instead_of_losing_its_tail() {
    // The reported bug: the pane cut at the panel edge and the rest of the
    // message was simply gone.
    let msg = "SSM snapshot pool: Marconi 16 slots (2424 MB), decode-rollback 8 slots x 8 seqs (9696 MB), 48 layers";
    let rows = wrap_line(logline(msg), 80);
    assert!(rows.len() > 1, "it wrapped");
    // Join on the WORDS, not the rendered rows: a wrap boundary puts the
    // continuation indent between "48" and "layers", so searching the
    // concatenated rows for the literal phrase would fail on correct
    // output. (It did.)
    let joined = rows.iter().map(|r| text(r)).collect::<Vec<_>>().join(" ");
    let words: Vec<&str> = joined.split_whitespace().collect();
    assert!(
        words.windows(2).any(|w| w == ["48", "layers"]),
        "the tail survives: {joined}"
    );
    for r in &rows {
        assert!(
            text(r).chars().count() <= 80,
            "no row exceeds the width: {:?}",
            text(r)
        );
    }
}

#[test]
fn continuations_are_indented_under_the_message() {
    let rows = wrap_line(logline(&"word ".repeat(40)), 60);
    assert!(rows.len() > 1);
    // The prefix is 9 + 6 + 15 = 30 characters wide.
    assert!(
        text(&rows[1]).starts_with(&" ".repeat(30)),
        "continuation lines up under the message, not under the timestamp"
    );
}

#[test]
fn a_zero_width_pane_does_not_panic_or_loop() {
    let rows = wrap_line(logline("anything"), 0);
    assert_eq!(rows.len(), 1);
}

fn main_app() -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Main;
    a
}

#[test]
fn a_server_with_no_model_refuses_to_report_a_load_that_is_not_happening() {
    // The listener binds before any model exists and reports itself ready, so
    // reading `progress.ready` alone drew a finished checklist and a full bar
    // beside an empty model name.
    let mut a = main_app();
    a.awaiting_model = true;
    a.progress.ready = true;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "STARTUP ─ awaiting a model"), "{rows:#?}");
    assert!(
        has(
            &rows,
            "no model loaded — open the Library (4) to choose one"
        ),
        "{rows:#?}"
    );
    assert!(!has(&rows, "READY"), "nothing is ready:\n{rows:#?}");
    assert!(!has(&rows, "OVERALL"), "and nothing is loading:\n{rows:#?}");
}

#[test]
fn a_loading_server_shows_the_checklist_and_the_weight_load_hero() {
    let mut a = main_app();
    a.progress.phases[0].state = PhaseState::Done;
    a.progress.phases[0].secs = 1.25;
    a.progress.phases[1].state = PhaseState::Running;
    a.progress.disk_gb = 34.9;
    a.progress.shard = 3;
    a.progress.shard_total = 10;
    a.progress.shard_name = "model-00003-of-00010.safetensors".into();
    a.progress.layer = 12;
    a.progress.layer_total = 40;
    a.progress.gpu_used_gb = 57.2;
    a.progress.gpu_free_gb = 62.5;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "STARTUP ─ 1/12"), "{rows:#?}");
    assert!(has(&rows, "✓ banner"), "a finished phase is ticked");
    assert!(has(&rows, "1.2s"), "and timed:\n{rows:#?}");
    assert!(has(&rows, "WEIGHT LOAD ─ model-00003-of-00010.safetensors"));
    assert!(has(&rows, "OVERALL"));
    assert!(has(&rows, "/ 34.9 GB preflight"));
    assert!(has(&rows, "SHARD  3/10"), "{rows:#?}");
    assert!(has(&rows, "LAYERS 12/40  GPU used 57.2 · free 62.5 GB"));
}

#[test]
fn a_serving_server_swaps_the_checklist_for_a_one_line_strip() {
    let mut a = main_app();
    a.progress.ready = true;
    a.progress.ready_in_secs = 41.5;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "READY ─"), "{rows:#?}");
    assert!(has(&rows, "nvidia/Qwen3.6-27B-NVFP4"));
    assert!(has(&rows, "loaded in 41.5s · listening"));
    assert!(has(&rows, &format!(":{}", a.args.port)));
    assert!(!has(&rows, "STARTUP"), "the checklist is gone:\n{rows:#?}");
}

#[test]
fn the_chip_strip_describes_the_configuration_that_is_running() {
    let rows = screen(&main_app(), 160, 48);
    assert!(has(&rows, "nvidia/Qwen3.6-27B-NVFP4"), "{rows:#?}");
    assert!(has(&rows, "kv "), "the KV dtype is a chip:\n{rows:#?}");
}

#[test]
fn the_log_pane_says_whether_it_is_following_or_holding_station() {
    let mut a = main_app();
    assert!(has(&screen(&a, 160, 48), "LOGS ── ⏵ follow"));

    a.log_scroll = Some(7);
    assert!(has(&screen(&a, 160, 48), "LOGS ── ⏸ 7↑"));
}

#[test]
fn the_log_filter_is_visible_while_it_is_being_typed_and_after() {
    let mut a = main_app();
    a.log_filter = "weight".into();
    a.log_filter_editing = true;
    assert!(has(&screen(&a, 160, 48), "filter: weight▏"));

    // Committed, the cursor goes but the filter stays on the title — a pane
    // silently hiding lines is how "the log stopped" gets reported.
    a.log_filter_editing = false;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "filter: weight"), "{rows:#?}");
    assert!(!has(&rows, "filter: weight▏"));
}

#[test]
fn a_long_shard_name_is_truncated_in_the_middle_so_both_ends_stay_readable() {
    let name = "models--nvidia--Qwen3.6-27B-NVFP4/snapshots/abc/model-00042-of-00085.safetensors";
    let out = middle_truncate(name, 34);
    assert!(out.chars().count() <= 34, "{out}");
    assert!(out.starts_with("models--nvidia"), "{out}");
    assert!(out.ends_with("safetensors"), "{out}");
    assert!(out.contains('…'));
    // A name that fits is left exactly as it is.
    assert_eq!(
        middle_truncate("short.safetensors", 34),
        "short.safetensors"
    );
}

#[test]
fn timestamps_are_wall_clock_hours_minutes_seconds() {
    use std::time::{Duration, UNIX_EPOCH};
    assert_eq!(chrono_lite(UNIX_EPOCH), "00:00:00");
    assert_eq!(
        chrono_lite(UNIX_EPOCH + Duration::from_secs(3661)),
        "01:01:01"
    );
    // Wraps at a day rather than counting hours forever.
    assert_eq!(
        chrono_lite(UNIX_EPOCH + Duration::from_secs(86_400)),
        "00:00:00"
    );
}

/// Main ▸ Kernels — the audit table, which is the instrument the whole
/// kernel-resolution story is read through.
mod kernels {
    use super::*;

    fn with_kernels(model: KernelTableModel) -> App {
        let mut a = main_app();
        a.main_sub = MainSub::Kernels;
        a.kernels = Some(model);
        a
    }

    fn row(module: &str, resolution: Option<bool>) -> KernelRow {
        KernelRow {
            module: module.to_string(),
            ptx_hash: "0123abcd".into(),
            resolution,
        }
    }

    #[test]
    fn the_pane_waits_rather_than_claiming_an_empty_audit() {
        let mut a = main_app();
        a.main_sub = MainSub::Kernels;
        let rows = screen(&a, 160, 48);
        assert!(has(&rows, "KERNELS ─ waiting for startup"), "{rows:#?}");
        assert!(has(&rows, "kernel audit runs at model load"));
    }

    #[test]
    fn a_module_reports_whether_it_was_used_never_looked_up_or_failed() {
        let a = with_kernels(KernelTableModel {
            rows: vec![
                row("atlas_gdn_decode", Some(true)),
                row("atlas_ssm_tail", Some(false)),
                row("atlas_moe_bf16", None),
            ],
            ..Default::default()
        });
        let rows = screen(&a, 160, 48);
        assert!(has(&rows, "KERNELS ─ 3 modules"), "{rows:#?}");
        assert!(has(&rows, "atlas_gdn_decode"));
        assert!(has(&rows, "used"));
        assert!(has(&rows, "** lookup FAILED **"), "{rows:#?}");
        assert!(has(&rows, "PTX-HASH"), "the header stays put");
    }

    #[test]
    fn only_the_actionable_failures_get_a_banner() {
        // Expected-absent kernels are declared with a reason; alarming on them
        // trains people to ignore the one time it matters.
        let a = with_kernels(KernelTableModel {
            rows: vec![row("atlas_gdn_decode", Some(true))],
            missing_required: vec![MissingKernel {
                module: "atlas_ssm_tail".into(),
                func: "tail_midchunk".into(),
                site: "ops/ssm.rs:214".into(),
            }],
            missing_expected: vec![MissingKernel {
                module: "atlas_fp4_mma".into(),
                func: "mma_e2m1".into(),
                site: "ops/fp4.rs:31".into(),
            }],
        });
        let rows = screen(&a, 160, 48);
        assert!(
            has(&rows, "⚠ 1 UNRESOLVED ─ 1 EXPECTED-ABSENT"),
            "{rows:#?}"
        );
        assert!(
            has(&rows, "atlas_ssm_tail::tail_midchunk  at ops/ssm.rs:214"),
            "the banner points at the dispatch site:\n{rows:#?}"
        );
        assert!(
            !has(&rows, "atlas_fp4_mma"),
            "a declared absence is not an alarm:\n{rows:#?}"
        );
    }

    #[test]
    fn a_filter_narrows_the_table_and_its_own_count() {
        let mut a = with_kernels(KernelTableModel {
            rows: vec![
                row("atlas_gdn_decode", Some(true)),
                row("atlas_moe_bf16", Some(true)),
            ],
            ..Default::default()
        });
        a.kernel_filter = "gdn".into();
        let rows = screen(&a, 160, 48);
        assert!(has(&rows, "KERNELS ─ 1 modules"), "{rows:#?}");
        assert!(!has(&rows, "atlas_moe_bf16"));
    }

    #[test]
    fn the_scroll_ceiling_is_published_from_what_was_actually_drawn() {
        // The wheel clamps to this; derived from the rows that survived the
        // filter, not from the audit's total.
        let a = with_kernels(KernelTableModel {
            rows: (0..80).map(|i| row(&format!("m{i}"), Some(true))).collect(),
            ..Default::default()
        });
        let _ = screen(&a, 160, 48);
        assert!(a.kernel_scroll_max.get() > 0, "80 rows do not fit in 48");

        let small = with_kernels(KernelTableModel {
            rows: vec![row("only", Some(true))],
            ..Default::default()
        });
        let _ = screen(&small, 160, 48);
        assert_eq!(small.kernel_scroll_max.get(), 0);
    }

    #[test]
    fn the_kernels_table_survives_narrow_and_short_terminals() {
        let a = with_kernels(KernelTableModel {
            rows: (0..30).map(|i| row(&format!("m{i}"), Some(true))).collect(),
            missing_required: vec![MissingKernel {
                module: "m".into(),
                func: "f".into(),
                site: "s:1".into(),
            }],
            missing_expected: Vec::new(),
        });
        for (w, h) in [(20u16, 4u16), (20, 8), (40, 12), (200, 3)] {
            assert_eq!(screen(&a, w, h).len(), h as usize, "{w}x{h}");
        }
    }
}
