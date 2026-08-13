// SPDX-License-Identifier: AGPL-3.0-only

//! The two things drawn ON TOP of a finished frame: toasts and the key map.
//!
//! Split out of `render/mod.rs` at the 500-LoC cap. They belong together: both
//! are modal chrome that owns cells some other pane has already painted, so
//! both must `Clear` what they cover and both must decline to draw at all
//! rather than clip themselves into something unreadable.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};

use super::wrap;
use crate::tui::app::App;
use crate::tui::theme;

/// Toasts, drawn over whatever is underneath them.
///
/// They are boxed rather than being a single tinted line. A toast lands on top
/// of a dense pane — a table, a log, a progress bar — and a one-row strip with
/// only a background colour to separate it reads as part of that pane rather
/// than as a message about it. The border is in the toast's own accent colour
/// (green or red), which is also what tells you at a glance whether something
/// succeeded or failed.
pub(super) fn draw_toasts(f: &mut Frame, app: &App, content: Rect) {
    // +2 columns and +2 rows per toast for the border box.
    //
    // 56, not 44: `DownloadError::hint` writes the FIX for a failure — "…set
    // HF_TOKEN, or run `hf auth login`" — and at 44 columns the fix was cut
    // mid-sentence, which is the half of the message that matters. Errors also
    // wrap rather than ellipsise, for the same reason.
    let width = 56.min(content.width.saturating_sub(2));
    let inner_w = width.saturating_sub(2) as usize;
    let mut y = content.y + 1;
    for t in app.toasts.iter().rev().take(3) {
        // Errors WRAP (up to 3 lines) because `DownloadError::hint` carries the
        // fix and truncating it drops the actionable half. Successes are one
        // line: they have nothing to act on.
        let text_w = inner_w.saturating_sub(2);
        let body: Vec<Line> = if t.error {
            wrap(&t.text, text_w, theme::text())
                .into_iter()
                .take(3)
                .collect()
        } else {
            vec![Line::from(Span::styled(
                truncate_toast(&t.text, text_w),
                theme::text(),
            ))]
        };
        let height = body.len() as u16 + 2;
        let area = Rect {
            x: content.x + content.width.saturating_sub(width + 1),
            y,
            width,
            height,
        };
        if area.bottom() > content.bottom() || width < 6 {
            // Not enough room to draw a box honestly; skip rather than clip a
            // border into something unreadable.
            continue;
        }
        y += height + 1;
        let accent = if t.error {
            theme::error()
        } else {
            theme::brand_green()
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(accent.add_modifier(Modifier::BOLD))
            .style(Style::default().bg(theme::BG_RAISED.color()));
        let inner = block.inner(area);
        // `Clear` over the WHOLE box, including the border cells: without it
        // the rounded corners sit on top of whatever glyph was underneath.
        f.render_widget(Clear, area);
        f.render_widget(block, area);
        // ★ The outcome glyph, not just the border colour. Under NO_COLOR the
        // accent flattens to Reset and a success toast and a failure toast
        // were the SAME BOX — the only difference was hue, which was gone.
        // `✗` shares font coverage with the `✓` the list rows already use.
        let mark = if t.error {
            Span::styled("\u{2717} ", theme::error().add_modifier(Modifier::BOLD))
        } else {
            Span::styled(
                "\u{2713} ",
                theme::brand_green().add_modifier(Modifier::BOLD),
            )
        };
        let mut lines: Vec<Line> = Vec::with_capacity(body.len());
        for (n, mut l) in body.into_iter().enumerate() {
            // Continuation lines indent under the glyph so a wrapped error
            // reads as one message rather than three.
            l.spans.insert(
                0,
                if n == 0 {
                    mark.clone()
                } else {
                    Span::raw("  ")
                },
            );
            lines.push(l);
        }
        f.render_widget(
            Paragraph::new(lines).style(Style::default().bg(theme::BG_RAISED.color())),
            inner,
        );
    }
}

/// One line, ellipsised, so a long message cannot spill past the border it is
/// supposed to be inside.
pub(super) fn truncate_toast(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let n = text.chars().count();
    if n <= width {
        return text.to_string();
    }
    let head: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{head}…")
}

/// The key map, and the SSOT for how tall its modal has to be.
pub(super) const KEYS: [(&str, &str); 23] = [
    ("1-7", "jump to section (repeat cycles its subsections)"),
    (
        "Tab / Shift+Tab",
        "walk every sidebar row, subsections included",
    ),
    ("j/k ↑/↓", "move / scroll"),
    ("g / G", "top / bottom (follow)"),
    ("f", "log filter (Main)"),
    ("/", "search (Library)"),
    ("←/→ + Enter", "select node / detail (Network)"),
    ("Enter", "focus input (Terminal) / edit field (Benchmarks)"),
    ("s / c", "start / cancel the configured benchmark"),
    (
        "d",
        "Library: download / resume / update the selected model",
    ),
    (
        "a / x",
        "Config form: add a setting / remove it (server default applies)",
    ),
    (
        "b",
        "Config form: borrow parameters from another recipe (previewed first)",
    ),
    ("x", "Library: stop the running download"),
    ("u", "Library: check the selected model for updates"),
    ("t / Ctrl+T", "Chat: ask for thinking — auto / off / on"),
    ("T / Alt+T", "Chat: reasoning collapsed / expanded / hidden"),
    (
        "Ctrl+N",
        "Chat: clear the conversation (confirms if not empty)",
    ),
    ("Esc", "back / cancel"),
    ("Ctrl+C", "clean shutdown (drain + exit)"),
    // ★ NOT "quit TUI". `q` sets should_quit, and the loop then calls
    // shutdown::request -- it DRAINS AND STOPS THE SERVER, exactly like
    // Ctrl+C. Describing that as closing a window invites a stray keypress to
    // end a four-hour benchmark. The honest label is half the fix; the other
    // half is `App::work_in_flight`, which makes the press cost a confirmation
    // whenever there is something to lose.
    ("q", "shut down the server (drain + exit; confirms if busy)"),
    // ★ The one way to leave the dashboard WITHOUT stopping the server, and it
    // was reachable only by typing it into the Terminal tab and knowing it
    // existed. A user looking for "how do I get out of this" found `q` in this
    // list and nothing else -- so the destructive answer was the discoverable
    // one. It is a slash command rather than a key, and it is listed here
    // anyway, because this modal is where the question gets asked.
    (
        "/detach",
        "Terminal: leave the TUI, keep serving with plain logs",
    ),
    ("7", "Help: guide + report an issue to GitHub"),
    ("?", "this help"),
];

pub(super) fn draw_help(f: &mut Frame, app: &App, area: Rect) {
    let w = 64.min(area.width.saturating_sub(4));
    // Sized to the LIST when the terminal is tall enough — and SCROLLED, not
    // clipped, when it is not. Sizing to the list (the previous fix) held
    // only while the terminal was taller than the table: at the 80x24 floor
    // the modal gets 22 rows, the table wants KEYS.len()+2, and the last two
    // entries — `q` and `?` — sat silently below the bottom border. Third
    // recurrence of this clip; scrolling is the version that survives the
    // table growing again.
    let h = ((KEYS.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let visible = (h.saturating_sub(2) as usize).max(1);
    // The ceiling, published for `App::on_help_overlay_key` — the same renderer-owned
    // contract as every scroll ceiling in `app_scroll`.
    let max = KEYS.len().saturating_sub(visible);
    app.help_scroll_max.set(max);
    let off = app.help_scroll.min(max);
    let mut lines = Vec::with_capacity(visible);
    for (k, d) in KEYS.iter().skip(off).take(visible) {
        lines.push(Line::from(vec![
            Span::styled(format!("  {k:<16}"), theme::brand_cyan()),
            Span::styled(d.to_string(), theme::text2()),
        ]));
    }
    let mut block = Block::bordered()
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(theme::border(false))
        .title(Span::styled("─ KEYS ─", theme::text2()))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    // Position in the bottom border, only when the list is clipped — the
    // `library/modal.rs` idiom. Words and glyphs, so NO_COLOR changes nothing.
    if max > 0 {
        block = block.title_bottom(Span::styled(
            format!(
                "─ j/k scroll · {}-{} of {} ─",
                off + 1,
                (off + visible).min(KEYS.len()),
                KEYS.len()
            ),
            theme::text2(),
        ));
    }
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

/// Ask before `q` drains a server that is in the middle of something.
///
/// Deliberately modal and deliberately small: it names WHAT is in flight,
/// because "are you sure" answers nothing a user did not already know, and
/// the thing they need to weigh is whether the hours already spent matter
/// more than getting their prompt back.
/// The "one download at a time" question.
///
/// Grammar copied from [`draw_quit_confirm`] on purpose: the same rounded warn
/// border, the same two-line key list as the affordance, the same
/// affirmative-only rule. A user who has met one of these can read the other
/// without learning anything new.
///
/// The consequence line is the part that matters, and it is TRUE of this
/// downloader specifically: files land in a `.part` sibling and resume with a
/// `Range:` request from the existing length, so stopping loses nothing but
/// the current chunk.
pub(super) fn draw_download_switch(f: &mut Frame, app: &App, area: Rect) {
    let Some((running, wanted)) = app.download_switch.as_ref() else {
        return;
    };
    let job = app.download.job.as_ref();
    // Live values: the question sits on screen while bytes keep moving, and a
    // frozen percentage would be the one number on it that lies.
    let progress = match job {
        Some(j) if j.cancelling => " is stopping — waiting for the current chunk.".to_string(),
        Some(j) => match j.fraction() {
            Some(fr) if j.rate_bps > 0.0 => format!(
                " is still downloading — {}, {}.",
                crate::tui::format::percent(fr),
                crate::tui::format::rate(j.rate_bps)
            ),
            Some(fr) => format!(
                " is still downloading — {}.",
                crate::tui::format::percent(fr)
            ),
            None => format!(
                " is still downloading — {} so far.",
                crate::tui::format::bytes(j.done)
            ),
        },
        None => " has just finished.".to_string(),
    };
    let mut lines = vec![
        Line::from(vec![
            Span::styled(format!("  {running}"), theme::warn()),
            Span::styled(progress, theme::warn()),
        ]),
        Line::from(Span::styled(
            "  A second pull would share the same disk and halve both.",
            theme::text(),
        )),
    ];
    // Only claim the bytes are kept when there ARE bytes to keep.
    if let Some(j) = job.filter(|j| j.done > 0) {
        lines.push(Line::from(Span::styled(
            format!(
                "  Stopping keeps its {} on disk; d resumes it later.",
                crate::tui::format::bytes(j.done)
            ),
            theme::text2(),
        )));
    }
    lines.push(Line::from(""));
    // BOLD on the key column, not colour alone: under NO_COLOR the quit
    // modal's cyan keys go flat and become indistinguishable from their own
    // descriptions.
    let key = theme::brand_cyan().add_modifier(Modifier::BOLD);
    lines.push(Line::from(vec![
        Span::styled("  x / y", key),
        Span::styled(format!("  stop it, start {wanted}"), theme::text2()),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  any other key", key),
        Span::styled("  keep the current download", theme::text2()),
    ]));
    let w = 72.min(area.width.saturating_sub(4));
    let h = ((lines.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::warn())
        .title(Span::styled(
            "\u{2500} ONE DOWNLOAD AT A TIME \u{2500}",
            theme::warn(),
        ))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

/// Ask before `Ctrl+N` discards the conversation.
///
/// Grammar copied from [`draw_quit_confirm`]: the same rounded warn border,
/// the same two-line key list, the same affirmative-only rule — a user who
/// has met one of these prompts can read the others without learning
/// anything new. The first line names what is at stake in the user's own
/// units (turns, and whether a reply is still arriving), because "are you
/// sure" answers nothing they did not already know.
pub(super) fn draw_chat_clear_confirm(f: &mut Frame, app: &App, area: Rect) {
    if !app.confirm_chat_clear {
        return;
    }
    let turns = app.chat.transcript.len();
    let what = if app.chat.streaming {
        format!("  {turns} turns, one still streaming — it will be cancelled.")
    } else {
        format!("  {turns} turns will be discarded.")
    };
    let lines = vec![
        Line::from(Span::styled(what, theme::warn())),
        Line::from(Span::styled(
            "  The model keeps no memory of them once cleared.",
            theme::text(),
        )),
        Line::from(""),
        Line::from(vec![
            // BOLD keys, same NO_COLOR argument as the quit prompt.
            Span::styled(
                "  y / Ctrl+N",
                theme::brand_cyan().add_modifier(Modifier::BOLD),
            ),
            Span::styled("  clear it", theme::text2()),
            Span::styled(
                "     any other key",
                theme::brand_cyan().add_modifier(Modifier::BOLD),
            ),
            Span::styled("  keep it", theme::text2()),
        ]),
    ];
    let w = 62.min(area.width.saturating_sub(4));
    let h = ((lines.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::warn())
        .title(Span::styled("─ CLEAR THE CONVERSATION? ─", theme::warn()))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

pub(super) fn draw_quit_confirm(f: &mut Frame, app: &App, area: Rect) {
    let Some(what) = app.work_in_flight() else {
        return;
    };
    let lines = vec![
        Line::from(Span::styled(format!("  {what}."), theme::warn())),
        Line::from(Span::styled(
            "  Quitting drains it and stops the server.",
            theme::text(),
        )),
        Line::from(""),
        Line::from(vec![
            // BOLD so the keys survive NO_COLOR, where cyan flattens to
            // Reset and a key became indistinguishable from its description.
            Span::styled("  q / y", theme::brand_cyan().add_modifier(Modifier::BOLD)),
            Span::styled("  quit anyway", theme::text2()),
            Span::styled(
                "     any other key",
                theme::brand_cyan().add_modifier(Modifier::BOLD),
            ),
            Span::styled("  stay", theme::text2()),
        ]),
    ];
    let w = 62.min(area.width.saturating_sub(4));
    let h = ((lines.len() + 2) as u16).min(area.height.saturating_sub(2));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(theme::warn())
        .title(Span::styled(
            "\u{2500} STOP THE SERVER? \u{2500}",
            theme::warn(),
        ))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    f.render_widget(Paragraph::new(lines).block(block), modal);
}
