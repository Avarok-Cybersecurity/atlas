// SPDX-License-Identifier: AGPL-3.0-only

//! The footer's per-section key hints — the two sections whose footer changes
//! with their inner state. Split from `render/mod.rs` at the 500-LoC cap.
//!
//! The rule both functions follow: the footer answers for whoever OWNS the
//! keyboard right now. A hint for a key that cannot act is not a hint, it is
//! a false claim about state — see the download `x stop` note below.

use crate::tui::app::App;

/// The Library's footer, which depends on which pane and mode it is in.
pub(super) fn library_hints(app: &App) -> &'static str {
    use crate::tui::lib_state::View;
    if app.lib.filter_editing {
        return "type to search · ⏎ keep · Esc clear";
    }
    match (app.lib.view, app.lib.editing) {
        (View::Cards, _) => "j/k move · ⏎ configure · d download · u updates · Esc back",
        // A picker owns the keyboard, so the footer answers for IT, not for
        // the form underneath it — and the borrow preview answers to a
        // different Enter (apply, not select), so it gets its own line.
        (View::Config, _)
            if matches!(
                app.lib.modal,
                Some(crate::tui::lib_modal::ConfigModal::Preview { .. })
            ) =>
        {
            "j/k scroll · ⏎ apply changes · Esc back to recipes"
        }
        // The add-picker's side panel scrolls on the SHIFTED pair; unnamed
        // here, J/K would be the one binding in the picker a user cannot
        // guess from the footer.
        (View::Config, _)
            if matches!(
                app.lib.modal,
                Some(crate::tui::lib_modal::ConfigModal::Add { .. })
            ) =>
        {
            "j/k move · J/K scroll help · ⏎ add · Esc cancel"
        }
        (View::Config, _) if app.lib.modal.is_some() => "j/k move · ⏎ select · Esc cancel",
        (View::Config, true) => "⏎ commit · Esc cancel",
        // `s` cannot start a model whose weights are absent, so the footer
        // says so BEFORE it is pressed rather than leaving the user to find
        // out from a refusal. Naming the way out matters more than naming the
        // key that will not work.
        (View::Config, false) if !app.lib.selected_has_weights() => {
            "⚠ weights not downloaded · Esc then d to download · ⏎ edit"
        }
        (View::Config, false) => {
            "j/k move · ⏎ edit · a add · x remove · b borrow · d recipe defaults · s START · Esc back"
        }
        // `x stop` ONLY while something is running. It used to be hardcoded,
        // which made it the one download-ish thing on screen at all times —
        // and a user with a download that never started read it as proof one
        // WAS running: "I don't see ANY indication of downloading except an
        // 'x to cancel'-like tag at the bottom". A key that cannot act is not
        // a hint, it is a false claim about state.
        // `u` taught here as well as on the Cards footer: it works in both
        // places, and a key that works but is named on only one screen reads
        // as absent on the other.
        (View::List, _) if app.download.job.is_some() => {
            "j/k move · ⏎ configure · d download · x stop · u updates · / search · r refresh"
        }
        (View::List, _) => {
            "j/k move · ⏎ configure · d download · u updates · / search · r refresh · ? help"
        }
    }
}

/// The Help footer answers for the report pipeline's current screen — five
/// phases with disjoint key sets, so one generic hint would be wrong on four
/// of them.
pub(super) fn help_hints(app: &App) -> &'static str {
    use crate::tui::help_state::{HelpSub, ReportPhase};
    if app.help.sub == HelpSub::Guide {
        return "⇥ cycle · 7 Report Issue · 1-7 jump · ? help · q quit";
    }
    if app.help.is_editing() {
        return "type · Esc done editing";
    }
    match app.help.phase {
        ReportPhase::Compose => "j/k field · ⏎ edit/toggle · s review & submit · 1-7 jump",
        ReportPhase::Preview => "j/k scroll · y send · a toggle logs · Esc back",
        ReportPhase::RequestingCode => "Esc cancel",
        ReportPhase::WaitingAuth { .. } => "c copy code · Esc cancel",
        ReportPhase::Submitting => "posting…",
        ReportPhase::Done { .. } => "c copy link · Esc compose another",
        ReportPhase::Failed { .. } => "s retry · Esc back",
    }
}

/// The Benchmarks footer changes with the step you are on — the form and the
/// live run answer to different keys, and a single generic hint would be wrong
/// in both.
pub(super) fn bench_hints(app: &App) -> &'static str {
    use crate::tui::app::BenchSub;
    use crate::tui::bench_state::View;
    if app.bench_sub == BenchSub::History {
        // PgUp/PgDn named or it does not exist: it is the one binding here a
        // user cannot guess from the run list.
        return "j/k run · PgUp/PgDn table · c card · ⇥ Suite↔History · ? help";
    }
    match (app.bench.view, app.bench.editing) {
        // PgUp/PgDn named for the same reason the History footer names it:
        // it is the one binding here a user cannot guess.
        (View::List, _) if app.bench.frame.is_some() => {
            "j/k select · PgUp/PgDn page · ⏎ configure · v last run · ⇥ Suite↔History · ? help"
        }
        (View::List, _) => {
            "j/k select · PgUp/PgDn page · ⏎ configure · ⇥ Suite↔History · 1-7 jump · ? help"
        }
        (View::Variants, _) => "j/k variant · ⏎ choose model · Esc back · ? help",
        (View::Params, true) => "⏎ commit · Esc cancel",
        (View::Params, false) => "j/k move · ⏎ edit · d defaults · p probe · s START · Esc back",
        (View::Run, _) => "c cancel · j/k scroll · Esc back to suite",
    }
}
