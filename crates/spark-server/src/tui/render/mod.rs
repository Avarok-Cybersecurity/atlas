// SPDX-License-Identifier: AGPL-3.0-only

//! Frame layout: sticky header (logo + status), sidebar, per-section content,
//! sticky footer, toasts, help overlay. Pure `App` → `Frame`.

mod library_tab;
mod main_tab;
mod network_tab;
mod stats_tab;
mod terminal_tab;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph};

use super::app::{App, Focus, MainSub, Section};
use super::{logo, theme};

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
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
    draw_header(f, app, rows[0], tall);

    let sidebar_w = if area.width >= 96 { 18 } else { 4 };
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(sidebar_w), Constraint::Min(20)])
        .split(rows[1]);
    draw_sidebar(f, app, cols[0], sidebar_w >= 18);

    match app.section {
        Section::Main => match app.main_sub {
            MainSub::Overview => main_tab::draw(f, app, cols[1]),
            MainSub::Kernels => main_tab::draw_kernels(f, app, cols[1]),
        },
        Section::Stats => stats_tab::draw(f, app, cols[1]),
        Section::Network => network_tab::draw(f, app, cols[1]),
        Section::Library => library_tab::draw(f, app, cols[1]),
        Section::Terminal => terminal_tab::draw(f, app, cols[1]),
    }

    draw_footer(f, app, rows[2]);
    draw_toasts(f, app, cols[1]);
    if app.help_open {
        draw_help(f, area);
    }
}

fn status_pill(app: &App) -> Span<'static> {
    let (label, bg) = if app.progress.ready {
        (" ● SERVING ", theme::GREEN)
    } else {
        (" ● LOADING ", theme::WARN)
    };
    Span::styled(
        label,
        Style::default()
            .bg(bg.color())
            .fg(theme::BG_BASE.color())
            .add_modifier(Modifier::BOLD),
    )
}

fn draw_header(f: &mut Frame, app: &App, area: Rect, tall: bool) {
    // Chevron wave only during loading (motion restraint).
    let wave = if app.progress.ready {
        None
    } else {
        Some((app.tick / 3) as usize % 3)
    };
    let up = app.started.elapsed().as_secs();
    let uptime = format!("up {:02}:{:02}", up / 60 % 100, up % 60);
    let right = Line::from(vec![
        status_pill(app),
        Span::styled(format!("  {uptime} "), theme::text2()),
    ]);
    if tall {
        let lines = logo::three_line(wave);
        for (i, line) in lines.into_iter().enumerate() {
            let row = Rect {
                y: area.y + i as u16,
                height: 1,
                ..area
            };
            f.render_widget(Paragraph::new(line), row);
        }
        // Right cluster row 0; model·quant·port row 1.
        f.render_widget(
            Paragraph::new(right).alignment(ratatui::layout::Alignment::Right),
            Rect {
                y: area.y,
                height: 1,
                ..area
            },
        );
        let model = app
            .args
            .model_name
            .clone()
            .or_else(|| app.args.model.clone())
            .unwrap_or_default();
        let sub = Line::from(Span::styled(
            format!(
                "{model} · kv {} · :{} ",
                app.args.kv_cache_dtype, app.args.port
            ),
            theme::text2(),
        ));
        f.render_widget(
            Paragraph::new(sub).alignment(ratatui::layout::Alignment::Right),
            Rect {
                y: area.y + 1,
                height: 1,
                ..area
            },
        );
    } else {
        f.render_widget(Paragraph::new(logo::one_line(wave)), area);
        f.render_widget(
            Paragraph::new(right).alignment(ratatui::layout::Alignment::Right),
            area,
        );
    }
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
    // 1-col rule on the right edge.
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
    } else if app.focus == Focus::Input || app.log_filter_editing || app.lib_filter_editing {
        (" INPUT ", theme::CYAN)
    } else {
        (" NORMAL ", theme::BORDER_DIM)
    };
    let hints = match app.section {
        Section::Main => "j/k scroll · f filter · ⇥ Overview↔Kernels · 1-5 jump · ? help · q quit",
        Section::Stats => "⇥ cycle · 1-5 jump · ? help · q quit",
        Section::Network => "←/→ node · ⏎ detail · ⇥ cycle · 1-5 jump · ? help",
        Section::Library => "j/k move · / search · ⇥ cycle · 1-5 jump · ? help",
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

fn draw_toasts(f: &mut Frame, app: &App, content: Rect) {
    let width = 42.min(content.width.saturating_sub(2));
    for (i, t) in app.toasts.iter().rev().take(3).enumerate() {
        let area = Rect {
            x: content.x + content.width.saturating_sub(width + 1),
            y: content.y + 1 + (i as u16) * 2,
            width,
            height: 1,
        };
        let accent = if t.error {
            theme::error()
        } else {
            theme::brand_green()
        };
        let line = Line::from(vec![
            Span::styled("▌ ", accent),
            Span::styled(t.text.clone(), theme::text()),
        ]);
        f.render_widget(Clear, area);
        f.render_widget(
            Paragraph::new(line).style(Style::default().bg(theme::BG_RAISED.color())),
            area,
        );
    }
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
        ("1-5", "jump to section (repeat cycles its subsections)"),
        (
            "Tab / Shift+Tab",
            "walk every sidebar row, subsections included",
        ),
        ("j/k ↑/↓", "move / scroll"),
        ("g / G", "top / bottom (follow)"),
        ("f", "log filter (Main)"),
        ("/", "search (Library)"),
        ("←/→ + Enter", "select node / detail (Network)"),
        ("Enter", "focus input (Terminal)"),
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
