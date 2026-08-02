// SPDX-License-Identifier: AGPL-3.0-only

//! Frame layout: sticky header (logo + status), sidebar, per-section content,
//! sticky footer, toasts, help overlay. Pure `App` → `Frame`.

mod bench;
mod header;
mod library;
mod main_tab;
mod network_tab;
mod stats_tab;
mod terminal_tab;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};

use super::app::{App, Focus, MainSub, Section};
use super::theme;

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    // Reset every cell's SYMBOL first. The base block below sets a background
    // style, and `Block::render` does that with `set_style`, which repaints
    // colour but leaves the glyph that was already there. A `Block`'s inner
    // area is only overwritten where a child widget actually draws, so any
    // frame whose content shrank or shifted left the previous frame's
    // characters on screen — a stale "MODELS ─ 0 ─ recipes never fetched"
    // header sat two rows above a live list of 25, updating in place while the
    // ghost above it never changed. Clearing costs one buffer pass; the
    // terminal diff still only emits cells that actually changed.
    f.render_widget(ratatui::widgets::Clear, area);
    // Paint the base surface.
    f.render_widget(
        Block::default().style(Style::default().bg(theme::BG_BASE.color())),
        area,
    );
    let tall = area.height >= 28;
    let header_h = if tall { 3 } else { 1 };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header_h),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .split(area);
    header::draw_header(f, app, rows[0], tall);

    let sidebar_w = if area.width >= 96 { 18 } else { 4 };
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(sidebar_w), Constraint::Min(20)])
        .split(rows[1]);
    draw_sidebar(f, app, cols[0], sidebar_w >= 18);

    // The content area always wears a 1-cell ring so nothing shifts when a
    // benchmark starts; the ring is dim while idle and pulses brand cyan while
    // a benchmark is running. It lives here rather than in the Benchmarks tab
    // so the signal follows you to Stats or Terminal mid-run.
    let content = draw_glow_ring(f, app, cols[1]);

    match app.section {
        Section::Main => match app.main_sub {
            MainSub::Overview => main_tab::draw(f, app, content),
            MainSub::Kernels => main_tab::draw_kernels(f, app, content),
        },
        Section::Stats => stats_tab::draw(f, app, content),
        Section::Network => network_tab::draw(f, app, content),
        Section::Library => library::draw(f, app, content),
        Section::Benchmarks => bench::draw(f, app, content),
        Section::Terminal => terminal_tab::draw(f, app, content),
    }

    draw_footer(f, app, rows[2]);
    draw_toasts(f, app, content);
    if app.help_open {
        draw_help(f, area);
    }
    // LAST, over everything including the help overlay: the highlight has to
    // show what will actually be copied, and what is copied is read back out
    // of this finished frame.
    draw_selection(f, app);
}

/// Paint the drag highlight onto the finished frame.
///
/// Reverses the cells rather than setting a colour, so it stays legible over
/// every panel background, the selected-row tint and the log pane's per-level
/// colours — a fixed highlight colour is invisible on at least one of them.
fn draw_selection(f: &mut Frame, app: &App) {
    let Some(sel) = app.selection.filter(|s| s.is_drag()) else {
        return;
    };
    let area = f.area();
    let buf = f.buffer_mut();
    let ((_, sy), (_, ey)) = sel.ordered();
    for y in sy..=ey.min(area.height.saturating_sub(1)) {
        for x in area.x..area.x.saturating_add(area.width) {
            if sel.contains(x, y) {
                buf[(x, y)].modifier |= Modifier::REVERSED;
            }
        }
    }
}

/// Paint the content ring and return the area inside it.
fn draw_glow_ring(f: &mut Frame, app: &App, area: Rect) -> Rect {
    let style = if app.bench.glow {
        Style::default().fg(theme::glow(app.tick))
    } else {
        theme::border(false)
    };
    let mut block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(style);
    if app.bench.glow {
        block = block.title(Span::styled(
            format!(
                "─ ⏱ {} ─",
                app.bench
                    .descriptor()
                    .map(|d| d.name)
                    .unwrap_or("benchmark")
            ),
            Style::default()
                .fg(theme::glow(app.tick))
                .add_modifier(Modifier::BOLD),
        ));
    }
    let inner = block.inner(area);
    f.render_widget(block, area);
    inner
}

fn draw_sidebar(f: &mut Frame, app: &App, area: Rect, full: bool) {
    let mut lines: Vec<Line> = Vec::new();
    for s in Section::ALL {
        let selected = app.section == s;
        let bar = if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        let icon_style = if selected {
            theme::text()
        } else {
            theme::text2()
        };
        let mut spans = vec![bar, Span::styled(format!("{} ", s.icon()), icon_style)];
        if full {
            let label_style = if selected {
                theme::text().add_modifier(Modifier::BOLD)
            } else {
                theme::text2()
            };
            spans.push(Span::styled(s.label().to_string(), label_style));
            // Main's dot is the startup lamp, and only that: amber while the engine
            // is coming up, green once it is serving. It used to mean "unresolved
            // kernel lookups" and only ever rendered amber, which read as a load
            // that never finished. Unresolved kernels are not duplicated here —
            // the Kernels tab banners them and a startup toast points at it.
            if s == Section::Main {
                let lamp = if app.progress.ready {
                    theme::brand_green()
                } else {
                    theme::warn()
                };
                spans.push(Span::styled("  ●", lamp));
            }
        }
        let mut line = Line::from(spans);
        if selected {
            line = line.style(Style::default().bg(theme::BG_SELECTION.color()));
        }
        lines.push(line);
        // Subsections under the active section (full mode).
        if full && selected {
            let subs = s.subs();
            let active_sub = app.sub_index(s);
            for (i, name) in subs.iter().enumerate() {
                let active = i == active_sub;
                let glyph = if i + 1 == subs.len() { "└" } else { "├" };
                let style = if active {
                    theme::brand_cyan()
                } else {
                    theme::dim()
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("   {glyph} "), theme::dim()),
                    Span::styled(name.to_string(), style),
                ]));
            }
        }
    }
    f.render_widget(Paragraph::new(lines), area);
    // 1-col rule on the right edge. `Layout` hands back a zero-width rect when
    // the terminal is narrower than the constraints ask for, and `area.width - 1`
    // then underflows and panics — taking the dashboard, and with it the
    // server's foreground, down on a resize nobody expected to matter.
    if area.width == 0 {
        return;
    }
    for y in area.y..area.y + area.height {
        f.render_widget(
            Paragraph::new(Span::styled("│", theme::dim())),
            Rect {
                x: area.x + area.width - 1,
                y,
                width: 1,
                height: 1,
            },
        );
    }
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let mode = if app.help_open {
        (" HELP ", theme::TEXT_2)
    } else if app.focus == Focus::Input || app.log_filter_editing || app.lib.is_editing() {
        (" INPUT ", theme::CYAN)
    } else {
        (" NORMAL ", theme::BORDER_DIM)
    };
    let hints = match app.section {
        Section::Main => "j/k scroll · f filter · ⇥ Overview↔Kernels · 1-6 jump · ? help · q quit",
        Section::Stats => "⇥ cycle · 1-6 jump · ? help · q quit",
        Section::Network => "←/→ node · ⏎ detail · ⇥ cycle · 1-6 jump · ? help",
        Section::Library => library_hints(app),
        Section::Benchmarks => bench_hints(app),
        Section::Terminal => "⏎ input · Esc back · ↑/↓ scroll · End follow · ⇥ Ops↔Chat · ? help",
    };
    let line = Line::from(vec![
        Span::styled(
            mode.0,
            Style::default()
                .bg(mode.1.color())
                .fg(theme::BG_BASE.color()),
        ),
        Span::styled(format!("  {hints}"), theme::dim()),
    ]);
    f.render_widget(
        Paragraph::new(line).style(Style::default().bg(theme::BG_PANEL.color())),
        area,
    );
}

/// The Benchmarks footer changes with the step you are on — the form and the
/// live run answer to different keys, and a single generic hint would be wrong
/// in both.
fn bench_hints(app: &App) -> &'static str {
    use crate::tui::app::BenchSub;
    use crate::tui::bench_state::View;
    if app.bench_sub == BenchSub::History {
        return "j/k run · ⇥ Suite↔History · 1-6 jump · ? help";
    }
    match (app.bench.view, app.bench.editing) {
        (View::List, _) if app.bench.frame.is_some() => {
            "j/k select · ⏎ configure · v last run · ⇥ Suite↔History · ? help"
        }
        (View::List, _) => "j/k select · ⏎ configure · ⇥ Suite↔History · 1-6 jump · ? help",
        (View::Params, true) => "⏎ commit · Esc cancel",
        (View::Params, false) => "j/k move · ⏎ edit · d defaults · p probe · s START · Esc back",
        (View::Run, _) => "c cancel · j/k scroll · Esc back to suite",
    }
}

/// Toasts, drawn over whatever is underneath them.
///
/// They are boxed rather than being a single tinted line. A toast lands on top
/// of a dense pane — a table, a log, a progress bar — and a one-row strip with
/// only a background colour to separate it reads as part of that pane rather
/// than as a message about it. The border is in the toast's own accent colour
/// (green or red), which is also what tells you at a glance whether something
/// succeeded or failed.
fn draw_toasts(f: &mut Frame, app: &App, content: Rect) {
    // +2 columns and +2 rows per toast for the border box.
    let width = 44.min(content.width.saturating_sub(2));
    let inner_w = width.saturating_sub(2) as usize;
    for (i, t) in app.toasts.iter().rev().take(3).enumerate() {
        // Three rows each now (box + a blank), so they stack without touching.
        let area = Rect {
            x: content.x + content.width.saturating_sub(width + 1),
            y: content.y + 1 + (i as u16) * 4,
            width,
            height: 3,
        };
        if area.bottom() > content.bottom() || width < 6 {
            // Not enough room to draw a box honestly; skip rather than clip a
            // border into something unreadable.
            continue;
        }
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
        let text = truncate_toast(&t.text, inner_w);
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(text, theme::text())))
                .style(Style::default().bg(theme::BG_RAISED.color())),
            inner,
        );
    }
}

/// One line, ellipsised, so a long message cannot spill past the border it is
/// supposed to be inside.
fn truncate_toast(text: &str, width: usize) -> String {
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

fn draw_help(f: &mut Frame, area: Rect) {
    let w = 64.min(area.width.saturating_sub(4));
    let h = 18.min(area.height.saturating_sub(4));
    let modal = Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, modal);
    let keys = [
        ("1-6", "jump to section (repeat cycles its subsections)"),
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
        ("s", "start the configured benchmark"),
        ("c", "cancel the running benchmark"),
        (
            "d",
            "Library: download / resume / update the selected model",
        ),
        ("x", "Library: stop the running download"),
        ("u", "Library: check the selected model for updates"),
        ("Ctrl+Enter", "send chat message"),
        ("Esc", "back / cancel"),
        ("Ctrl+C", "clean shutdown (drain + exit)"),
        ("q", "quit TUI"),
        ("?", "this help"),
    ];
    let mut lines = vec![Line::default()];
    for (k, d) in keys {
        lines.push(Line::from(vec![
            Span::styled(format!("  {k:<16}"), theme::brand_cyan()),
            Span::styled(d.to_string(), theme::text2()),
        ]));
    }
    let block = Block::bordered()
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(theme::border(false))
        .title(Span::styled("─ KEYS ─", theme::text2()))
        .style(Style::default().bg(theme::BG_PANEL.color()));
    f.render_widget(Paragraph::new(lines).block(block), modal);
}

/// Shared rounded-panel block.
pub(super) fn panel(title: String, focused: bool) -> Block<'static> {
    Block::bordered()
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(theme::border(focused))
        .title(Span::styled(format!("─ {title} "), theme::title(focused)))
        .style(Style::default().bg(theme::BG_PANEL.color()))
}

/// The signature gradient bar as a styled line: `█▓░` with per-cell color.
pub(super) fn gradient_bar(frac: f64, width: u16) -> Line<'static> {
    let width = width.max(1) as usize;
    let filled = ((frac.clamp(0.0, 1.0)) * width as f64).round() as usize;
    let mut spans = Vec::with_capacity(width);
    for i in 0..width {
        if i < filled {
            let t = i as f64 / (width.saturating_sub(1)).max(1) as f64;
            let ch = if i + 1 == filled && filled < width {
                "▓"
            } else {
                "█"
            };
            spans.push(Span::styled(ch, Style::default().fg(theme::gradient_at(t))));
        } else {
            spans.push(Span::styled(
                "░",
                Style::default().fg(theme::GAUGE_TRACK.color()),
            ));
        }
    }
    Line::from(spans)
}

#[cfg(test)]
#[path = "render_tests.rs"]
mod tests;

/// Wrap `text` to `width` columns as owned lines.
/// The Library's footer, which depends on which pane and mode it is in.
fn library_hints(app: &App) -> &'static str {
    use crate::tui::lib_state::View;
    if app.lib.filter_editing {
        return "type to search · ⏎ keep · Esc clear";
    }
    match (app.lib.view, app.lib.editing) {
        (View::Cards, _) => "j/k move · ⏎ configure · d download · u updates · Esc back",
        (View::Config, true) => "⏎ commit · Esc cancel",
        // `s` cannot start a model whose weights are absent, so the footer
        // says so BEFORE it is pressed rather than leaving the user to find
        // out from a refusal. Naming the way out matters more than naming the
        // key that will not work.
        (View::Config, false) if !app.lib.selected_has_weights() => {
            "⚠ weights not downloaded · Esc then d to download · ⏎ edit"
        }
        (View::Config, false) => {
            "j/k move · ⏎ edit · d recipe defaults · s START · Esc back to recipes"
        }
        (View::List, _) => {
            "j/k move · ⏎ configure · d download · x stop · / search · r refresh · ? help"
        }
    }
}

/// The model actually being served, or the one the argv asked for.
///
/// `args` is the argv the dashboard STARTED with. It is empty for `spark serve`
/// with no model, so a Library launch rendered a blank name, and after a
/// request-triggered swap it would have gone on naming the model the process
/// booted with. Three panes asked the same question and all three asked the
/// wrong source; the host is the one that knows.
pub(crate) fn live_model_name(app: &App) -> String {
    app.host
        .as_ref()
        .and_then(|h| h.live_model())
        .or_else(|| app.args.model_name.clone())
        .or_else(|| app.args.model.clone())
        .unwrap_or_default()
}

pub(crate) fn wrap(text: &str, width: usize, style: ratatui::style::Style) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.len() + 1 + word.len() > width {
            lines.push(Line::from(Span::styled(
                std::mem::take(&mut current),
                style,
            )));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(Line::from(Span::styled(current, style)));
    }
    lines
}

#[cfg(test)]
#[path = "selection_render_tests.rs"]
mod selection_tests;
