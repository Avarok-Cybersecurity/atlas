// SPDX-License-Identifier: AGPL-3.0-only

//! The Help section's screens over `TestBackend`. Layout code is where a TUI
//! actually crashes, so every phase renders at the 80×24 floor and at a size
//! small enough to exercise the saturating math — plus content assertions for
//! the claims that must be on screen (public destination, preview counts,
//! the user code, the redaction caveat).

use super::super::harness::{has, screen};
use crate::tui::app::{App, Section};
use crate::tui::help_state::{HelpSub, ReportPhase};
use crate::tui::report::Composed;

fn help(sub: HelpSub) -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Help;
    a.help.sub = sub;
    a
}

fn composed() -> Composed {
    let body = format!(
        "steps\n\n## Environment\n\nAtlas test\n\n## Server log (last 2 of 9 lines, redacted best-effort)\n\n```text\nline a\nline b\n```\n\n{}\n",
        crate::tui::report::MARKER
    );
    Composed {
        chars: body.chars().count(),
        body,
        logs_included: 2,
        logs_total: 9,
    }
}

#[test]
fn every_phase_renders_at_the_floor_sizes() {
    let phases: Vec<ReportPhase> = vec![
        ReportPhase::Compose,
        ReportPhase::Preview,
        ReportPhase::RequestingCode,
        ReportPhase::WaitingAuth {
            user_code: "WDJB-MJHT".into(),
            verification_uri: "https://github.com/login/device".into(),
            expires_at: std::time::Instant::now() + std::time::Duration::from_secs(872),
        },
        ReportPhase::Submitting,
        ReportPhase::Done {
            number: 214,
            url: "https://github.com/o/r/issues/214".into(),
        },
        ReportPhase::Failed {
            message: "GitHub returned 503 — try again shortly".into(),
        },
    ];
    for phase in phases {
        for (w, h) in [(160, 48), (80, 24), (40, 12)] {
            let mut a = help(HelpSub::Report);
            a.help.preview = Some(composed());
            a.help.phase = match &phase {
                ReportPhase::WaitingAuth {
                    user_code,
                    verification_uri,
                    expires_at,
                } => ReportPhase::WaitingAuth {
                    user_code: user_code.clone(),
                    verification_uri: verification_uri.clone(),
                    expires_at: *expires_at,
                },
                ReportPhase::Done { number, url } => ReportPhase::Done {
                    number: *number,
                    url: url.clone(),
                },
                ReportPhase::Failed { message } => ReportPhase::Failed {
                    message: message.clone(),
                },
                ReportPhase::Compose => ReportPhase::Compose,
                ReportPhase::Preview => ReportPhase::Preview,
                ReportPhase::RequestingCode => ReportPhase::RequestingCode,
                ReportPhase::Submitting => ReportPhase::Submitting,
            };
            let rows = screen(&a, w, h);
            assert!(
                rows.iter().any(|r| !r.is_empty()),
                "phase drew nothing at {w}x{h}"
            );
        }
    }
}

#[test]
fn the_guide_names_the_version_the_exits_and_the_memory_only_promise() {
    let rows = screen(&help(HelpSub::Guide), 120, 40);
    assert!(has(&rows, crate::cli::ATLAS_VERSION), "{rows:#?}");
    assert!(has(&rows, "stops the SERVER"));
    assert!(has(&rows, "/detach"));
    assert!(has(&rows, "in memory only"));
}

#[test]
fn the_composer_names_the_public_destination_and_the_checkbox_state() {
    let mut a = help(HelpSub::Report);
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "posts publicly to github.com/"), "{rows:#?}");
    assert!(
        has(&rows, "[x] attach server logs"),
        "checkbox defaults ON, as glyphs"
    );
    assert!(
        has(&rows, "review & submit"),
        "with logs on, s reviews — it does not send"
    );
    a.help.attach_logs = false;
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "[ ] attach server logs"), "{rows:#?}");
    assert!(has(&rows, "s submit (no logs attached)"));
}

#[test]
fn the_preview_shows_counts_caveat_fence_and_marker() {
    let mut a = help(HelpSub::Report);
    a.help.phase = ReportPhase::Preview;
    a.help.preview = Some(composed());
    let rows = screen(&a, 120, 44);
    assert!(has(&rows, "exactly what will be posted"), "{rows:#?}");
    assert!(has(&rows, "/ 65536 chars"), "the budget is stated");
    assert!(has(&rows, "last 2 of 9 lines"));
    assert!(has(&rows, "redaction is best-effort"), "no over-claiming");
    assert!(
        has(&rows, "```text"),
        "the fence is visible, so what ships is inspectable"
    );
    assert!(
        has(&rows, "atlas-tui-report"),
        "the marker ships in the open"
    );
}

#[test]
fn preview_scroll_publishes_a_ceiling_and_windows_the_rows() {
    let mut a = help(HelpSub::Report);
    a.help.phase = ReportPhase::Preview;
    let mut c = composed();
    c.body = (0..200)
        .map(|i| format!("row {i}"))
        .collect::<Vec<_>>()
        .join("\n");
    a.help.preview = Some(c);
    let rows = screen(&a, 80, 24);
    assert!(has(&rows, "row 0"), "{rows:#?}");
    assert!(!has(&rows, "row 150"), "below the fold");
    let max = a.help.preview_scroll_max.get();
    assert!(max > 100, "ceiling published for the clamp, got {max}");
    a.help.preview_scroll = 150;
    let rows = screen(&a, 80, 24);
    assert!(has(&rows, "row 150"), "{rows:#?}");
    assert!(!has(&rows, "row 0"));
}

#[test]
fn the_authorize_screen_shows_code_url_countdown_and_cancel() {
    let mut a = help(HelpSub::Report);
    a.help.phase = ReportPhase::WaitingAuth {
        user_code: "WDJB-MJHT".into(),
        verification_uri: "https://github.com/login/device".into(),
        expires_at: std::time::Instant::now() + std::time::Duration::from_secs(872),
    };
    let rows = screen(&a, 80, 24);
    assert!(has(&rows, "WDJB-MJHT"), "{rows:#?}");
    assert!(
        has(&rows, "https://github.com/login/device"),
        "the URL is printed, never launched"
    );
    assert!(has(&rows, "code valid 14m"), "countdown in words");
    assert!(has(&rows, "c"), "copy affordance");
    assert!(has(&rows, "Esc"), "a way out");
    assert!(has(&rows, "in memory only") || has(&rows, "kept in memory only"));
}

#[test]
fn done_and_failed_carry_their_glyph_twins() {
    // Under NO_COLOR the green/red accents flatten; the ✓/✗ glyphs are what
    // separate the outcomes.
    let mut a = help(HelpSub::Report);
    a.help.phase = ReportPhase::Done {
        number: 214,
        url: "https://github.com/o/r/issues/214".into(),
    };
    let rows = screen(&a, 100, 30);
    assert!(has(&rows, "✓ issue #214 opened"), "{rows:#?}");
    assert!(has(&rows, "https://github.com/o/r/issues/214"));

    a.help.phase = ReportPhase::Failed {
        message: "GitHub is rate-limiting — retry in 30s".into(),
    };
    let rows = screen(&a, 100, 30);
    assert!(has(&rows, "✗ sending failed"), "{rows:#?}");
    assert!(has(&rows, "rate-limiting"));
    assert!(
        has(&rows, "the draft is intact"),
        "the survival promise is stated where it matters"
    );
}

#[test]
fn the_footer_answers_for_the_phase() {
    let mut a = help(HelpSub::Report);
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "s review & submit"), "{rows:#?}");
    a.help.phase = ReportPhase::Preview;
    a.help.preview = Some(composed());
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "y send"), "{rows:#?}");
}

#[test]
fn the_sidebar_and_key_overlay_teach_the_section() {
    let mut a = help(HelpSub::Guide);
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "Help"), "{rows:#?}");
    assert!(has(&rows, "Guide"));
    assert!(has(&rows, "Report Issue"));
    a.help_open = true;
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "1-7"), "the jump hint counts to seven now");
    assert!(has(&rows, "report an issue to GitHub"));
}
