// SPDX-License-Identifier: AGPL-3.0-only

//! Frame layout: sticky header (logo + status), sidebar, per-section content,
//! sticky footer, toasts, help overlay. Pure `App` → `Frame`.

mod bench;
mod chat_lines;
mod header;
mod help_tab;
mod hints;
mod library;
mod main_tab;
mod main_tab_kernels;
mod network_tab;
mod overlay;
mod stats_tab;
mod stats_thermal;
mod terminal_tab;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect, Size};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Paragraph};

use super::app::{App, Focus, MainSub, Section};
use super::theme;

/// Where the header ends and the sidebar ends, for a terminal of this size.
///
/// ★ **One definition, because the renderer and the hit-tester have to agree
/// exactly.** `events::on_mouse` maps a click back to a sidebar row by
/// subtracting the header height and testing the column against the sidebar
/// width — so it held its own copy of all four breakpoints, in another file,
/// with no test that could notice them drifting apart. The failure mode is
/// silent: nothing crashes and no row is out of range, the wrong section just
/// opens. It is the same class of defect as the subsection-offset bug
/// `events::sidebar_row` was extracted to fix, one layer out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chrome {
    /// Header rows above the sidebar's first row.
    pub header_h: u16,
    /// Sidebar columns.
    pub sidebar_w: u16,
}

impl Chrome {
    /// The chrome a terminal this size gets.
    pub fn of(size: Size) -> Self {
        Self {
            // The three-row header carries the logo block; below this the
            // content pane cannot spare the two rows, so it collapses to a
            // one-line strip.
            header_h: if size.height >= 28 { 3 } else { 1 },
            // The wide sidebar carries labels; the narrow one is icons only.
            sidebar_w: if size.width >= 96 { 18 } else { 4 },
        }
    }

    /// Is the header drawing the logo block rather than the one-line strip?
    pub fn tall_header(&self) -> bool {
        self.header_h > 1
    }

    /// Is the sidebar drawing labels — and therefore the active section's
    /// subsection rows, which shift every row below them?
    pub fn full_sidebar(&self) -> bool {
        self.sidebar_w >= 18
    }
}

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
    // Every published hit-target resets each frame: a rect from a frame that
    // is no longer on screen must not keep catching clicks.
    app.lib_search_click.set(None);
    let chrome = Chrome::of(area.as_size());
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(chrome.header_h),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .split(area);
    header::draw_header(f, app, rows[0], chrome.tall_header());

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(chrome.sidebar_w), Constraint::Min(20)])
        .split(rows[1]);
    draw_sidebar(f, app, cols[0], chrome.full_sidebar());

    // The content area always wears a 1-cell ring so nothing shifts when a
    // benchmark starts; the ring is dim while idle and pulses brand cyan while
    // a benchmark is running. It lives here rather than in the Benchmarks tab
    // so the signal follows you to Stats or Terminal mid-run.
    let content = draw_glow_ring(f, app, cols[1]);

    match app.section {
        Section::Main => match app.main_sub {
            MainSub::Overview => main_tab::draw(f, app, content),
            MainSub::Kernels => main_tab_kernels::draw(f, app, content),
        },
        Section::Stats => stats_tab::draw(f, app, content),
        Section::Network => network_tab::draw(f, app, content),
        Section::Library => library::draw(f, app, content),
        Section::Benchmarks => bench::draw(f, app, content),
        Section::Terminal => terminal_tab::draw(f, app, content),
        Section::Help => help_tab::draw(f, app, content),
    }

    draw_footer(f, app, rows[2]);
    overlay::draw_toasts(f, app, content);
    if app.help_open {
        overlay::draw_help(f, app, area);
    }
    // After the help modal: a question the user must answer outranks a
    // reference they were browsing.
    // Before the quit prompt: stopping the SERVER outranks a question about a
    // download, so if both are somehow up the server one is on top.
    overlay::draw_download_switch(f, app, area);
    // Below the quit prompt for the same reason the download question is:
    // stopping the server outranks a question about a transcript.
    overlay::draw_chat_clear_confirm(f, app, area);
    if app.confirm_quit {
        overlay::draw_quit_confirm(f, app, area);
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
            line = line.style(theme::selected());
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
        Section::Main => "j/k scroll · f filter · ⇥ Overview↔Kernels · 1-7 jump · ? help · q quit",
        Section::Stats => "⇥ cycle · 1-7 jump · ? help · q quit",
        // No "⏎ detail" here: Enter used to toggle a bool nothing rendered —
        // an advertised key with zero effect. The detail pane is always drawn.
        Section::Network => "←/→ node · ⇥ cycle · 1-7 jump · ? help",
        Section::Library => hints::library_hints(app),
        Section::Benchmarks => hints::bench_hints(app),
        // `/detach` named here and nowhere else on screen: it is the only way
        // out that leaves the server running, and this is the tab it is typed
        // into. Without it the only exit a user could find was `q`, which
        // stops the server.
        Section::Terminal => {
            "⏎ input · Esc back · ↑/↓ scroll · ⇥ Ops↔Chat · /detach leave · ? help"
        }
        Section::Help => hints::help_hints(app),
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
#[path = "harness.rs"]
mod harness;

#[cfg(test)]
#[path = "render_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "chrome_tests.rs"]
mod chrome_tests;

#[cfg(test)]
#[path = "download_render_tests.rs"]
mod download_tests;

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

/// Wrap `text` to `width` columns as styled lines.
///
/// The accumulation loop is `format::wrap_words` — one loop for the whole
/// dashboard, styled here. It measures BYTES, wrapping early rather than late,
/// and the `Paragraph`s downstream have no `Wrap` of their own; see the loop
/// for both arguments.
pub(crate) fn wrap(text: &str, width: usize, style: ratatui::style::Style) -> Vec<Line<'static>> {
    crate::tui::format::wrap_words(text, width)
        .into_iter()
        .map(|l| Line::from(Span::styled(l, style)))
        .collect()
}

#[cfg(test)]
#[path = "selection_render_tests.rs"]
mod selection_tests;
