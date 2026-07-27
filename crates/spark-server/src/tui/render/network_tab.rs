// SPDX-License-Identifier: AGPL-3.0-only

//! Network tab: TP/EP topology as hand-drawn node cards. The single-node
//! case is a designed centerpiece, not an empty screen; 2+ nodes join with
//! labeled connectors. Health/latency shown honestly — never faked.

use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::panel;
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let world = app.args.world_size.max(1);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Min(4)])
        .split(area);
    if world == 1 {
        draw_single(f, app, rows[0]);
    } else {
        draw_multi(f, app, rows[0], world);
    }
    draw_detail(f, app, rows[1]);
}

fn card_lines(app: &App, rank: usize) -> Vec<Line<'static>> {
    let s = &app.stats;
    let is_head = rank == 0;
    let role = if is_head { "HEAD" } else { "WORKER" };
    let mut lines = vec![Line::from(Span::styled(
        format!("⬡ rank {rank} · {role}"),
        theme::text().add_modifier(Modifier::BOLD),
    ))];
    if is_head {
        lines.push(Line::from(Span::styled(
            "NVIDIA GB10 · local".to_string(),
            theme::text2(),
        )));
        let used = s.atlas_used_gb;
        let total = s.gpu_total_gb.max(0.001);
        let w = 18usize;
        let filled = ((used / total).clamp(0.0, 1.0) * w as f64) as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(w - filled);
        lines.push(Line::from(vec![
            Span::styled("mem ", theme::dim()),
            Span::styled(bar, theme::brand_cyan()),
            Span::styled(format!(" {used:.0}/{total:.0} GB"), theme::text2()),
        ]));
        lines.push(Line::from(Span::styled(
            format!("{}:{}", app.args.bind, app.args.port),
            theme::text2(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            format!("{}:{}", app.args.master_addr, app.args.master_port),
            theme::text2(),
        )));
        lines.push(Line::from(Span::styled(
            "state not visible from head (no worker HTTP)".to_string(),
            theme::dim(),
        )));
    }
    lines
}

fn draw_single(f: &mut Frame, app: &App, area: Rect) {
    let w = 44.min(area.width.saturating_sub(4));
    let h = 8.min(area.height.saturating_sub(2));
    let card = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    // Decorative dim chevrons flanking the card (the stage).
    if card.x >= area.x + 6 {
        let deco = |x: u16| Rect {
            x,
            y: card.y + h / 2,
            width: 4,
            height: 1,
        };
        let l = Line::from(Span::styled(
            "❯❯❯",
            theme::brand_purple().add_modifier(Modifier::DIM),
        ));
        f.render_widget(Paragraph::new(l.clone()), deco(area.x + 1));
        f.render_widget(Paragraph::new(l), deco(card.x + w + 2));
    }
    // Heartbeat: border alternates green/dim-green every 2s (2 frames at 10Hz).
    let beat = (app.tick / 20).is_multiple_of(2);
    let border = if beat {
        theme::brand_green()
    } else {
        theme::dim()
    };
    let block = panel("⬡ rank 0 · HEAD ─".into(), false).border_style(border);
    f.render_widget(Paragraph::new(card_lines(app, 0)).block(block), card);
    let caption = format!(
        "tp {} · ep {} · world 1 — all local",
        app.args.tp_size, app.args.ep_size
    );
    f.render_widget(
        Paragraph::new(Span::styled(caption, theme::text2())).alignment(Alignment::Center),
        Rect {
            y: (card.y + h).min(area.y + area.height - 1),
            height: 1,
            ..area
        },
    );
}

fn draw_multi(f: &mut Frame, app: &App, area: Rect, world: usize) {
    let n = world.min(4);
    let mut constraints = Vec::new();
    for _ in 0..n {
        constraints.push(Constraint::Ratio(1, n as u32));
    }
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);
    for rank in 0..n {
        let selected = app.network_selected == rank;
        let border = if selected {
            theme::brand_cyan()
        } else {
            theme::brand_green()
        };
        let title = format!("⬡ rank {rank} ─");
        let block = panel(title, selected).border_style(border);
        let cell = cols[rank];
        let card = Rect {
            x: cell.x + 1,
            y: cell.y + 1,
            width: cell.width.saturating_sub(2),
            height: cell.height.saturating_sub(3).min(8),
        };
        f.render_widget(Paragraph::new(card_lines(app, rank)).block(block), card);
        if rank + 1 < n {
            let link = Rect {
                x: cell.x + cell.width.saturating_sub(1),
                y: card.y + card.height / 2,
                width: 2.min(area.width),
                height: 1,
            };
            f.render_widget(
                Paragraph::new(Span::styled("══", theme::brand_cyan())),
                link,
            );
        }
    }
    let label = if app.args.ep_size > 1 {
        format!("═ NCCL ═ ep {} · experts sharded", app.args.ep_size)
    } else {
        format!("═ NCCL ═ tp {} · ring", app.args.tp_size)
    };
    f.render_widget(
        Paragraph::new(Span::styled(label, theme::brand_cyan())).alignment(Alignment::Center),
        Rect {
            y: area.y + area.height.saturating_sub(1),
            height: 1,
            ..area
        },
    );
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let block = panel(format!("NODE {} ─ detail ─", app.network_selected), false);
    let mut lines = vec![Line::from(vec![
        Span::styled(" role ", theme::dim()),
        Span::styled(
            if app.network_selected == 0 {
                "head (serves HTTP, owns scheduler)"
            } else {
                "worker (EP shard, no HTTP)"
            },
            theme::text(),
        ),
    ])];
    lines.push(Line::from(vec![
        Span::styled(" bootstrap ", theme::dim()),
        Span::styled(
            format!(
                "{}:{} (NCCL master)",
                app.args.master_addr, app.args.master_port
            ),
            theme::text(),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(" topology ", theme::dim()),
        Span::styled(
            format!(
                "world {} · tp {} · ep {} · this rank {}",
                app.args.world_size, app.args.tp_size, app.args.ep_size, app.args.rank
            ),
            theme::text(),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(" latency ", theme::dim()),
        Span::styled(
            "— (probes race the collectives; deliberately not measured live)",
            theme::dim(),
        ),
    ]));
    f.render_widget(Paragraph::new(lines).block(block), area);
}
