// SPDX-License-Identifier: AGPL-3.0-only

//! The joined recipe⋈local list, and its detail pane.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Modifier;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{panel, wrap};
use crate::tui::app::App;
use crate::tui::data::catalogue::Entry;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area);
    draw_list(f, app, cols[0]);
    super::list_detail::draw_detail(f, app, cols[1]);
}

/// The weights-state badge, then `▐recipe▌` and `▐optimized▌`. Recipe and
/// kernel target are two independent facts: a recipe with no compiled kernel
/// target still serves, on generic kernels.
///
/// The weights state is a WORD, first on the line, not only the mark glyph in
/// the left column. A user pressed `d` on a model that was already fully on
/// disk, the no-op finished inside one frame, and they concluded downloads
/// were broken — the `✓` was on screen the whole time and read as decoration.
/// A word needs no decoding. A badge rather than re-sorting (the sort already
/// puts runnable rows first, but a filtered or scrolled list gives no visible
/// boundary to read the grouping from) and rather than a column (which spends
/// fixed width on every row at every terminal size). First on the line so the
/// `Paragraph`'s clipping drops the subtitle before it ever drops the state.
///
/// Under `NO_COLOR` the states still differ twice over: present/partial/
/// downloading are REVERSED blocks (a modifier, which survives), absent is
/// plain dim text — and the words themselves are four different words.
fn badges(app: &App, entry: &Entry) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    // Deliberately static while downloading: the row already pulses on the
    // `(tick/4)%2` cadence and the third line carries the moving bar — a
    // second animation here would compete with that cadence, not add to it.
    let (label, style) = if app.download.is_downloading(&entry.model) {
        (
            " downloading ",
            theme::brand_cyan().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    } else if entry.has_weights() {
        (
            " on disk ",
            theme::brand_green().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    } else if entry.local.is_some() {
        // A started, unfinished download: `d` resumes it.
        (
            " partial ",
            theme::warn().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )
    } else {
        // Plain dim text, never a block: absence drawn as a solid badge reads
        // as a state that was ACHIEVED, and the asymmetry is itself the
        // signal once colour is gone.
        (" not downloaded ", theme::dim())
    };
    out.push(Span::styled(label, style));
    out.push(Span::raw(" "));
    if let Some(r) = entry.primary() {
        let (label, style) = if r.is_atlas() {
            (" recipe ", theme::brand_purple())
        } else {
            // Listed, never hidden — but it cannot be launched from here.
            (" vllm ", theme::dim())
        };
        out.push(Span::styled(
            label,
            style.add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ));
        out.push(Span::raw(" "));
    }
    if entry.optimized() {
        out.push(Span::styled(
            " optimized ",
            theme::brand_cyan().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ));
        out.push(Span::raw(" "));
    }
    // Last, so it is the first thing dropped when the pane is narrow, and only
    // for a CONFIRMED mismatch: an unreachable Hub reports `Unknown`, which
    // draws nothing. A stale badge shown because the network was down would be
    // a lie about the user's disk.
    match app.download.freshness.get(&entry.model) {
        Some(f) if f.is_stale() => out.push(Span::styled(
            " update ",
            theme::warn().add_modifier(Modifier::REVERSED | Modifier::BOLD),
        )),
        _ if app.download.checking.as_deref() == Some(entry.model.as_str()) => {
            // Skeleton in the place the badge will occupy, so the row does not
            // reflow when the answer lands.
            out.push(Span::styled(" ░░░░░░ ", theme::dim()));
        }
        _ => {}
    }
    out
}

/// The persistent search field: one row, always drawn, at the top of the list.
///
/// The filter has existed for as long as the Library — `/` opens it — and
/// users did not find it, because until pressed it appeared NOWHERE: an
/// affordance that only exists while active is not an affordance. The field
/// costs one list row and is drawn in all three states (empty, set, editing),
/// so "can I search this?" is answered by looking, not by knowing.
///
/// Publishes its rect through `App::lib_search_click` so a mouse click can
/// focus it — published from what was actually DRAWN rather than recomputed in
/// the hit-tester, for the same reason `render::Chrome` exists: two copies of
/// layout math drift, and the failure is silent.
///
/// Under `NO_COLOR`: the `⌕` glyph and the bracketed hint carry "this is a
/// search box"; while editing the row is REVERSED (`theme::selected`) and the
/// footer mode chip flips to INPUT, both of which survive without colour.
fn draw_search_field(f: &mut Frame, app: &App, row: Rect, total: usize, shown: usize) {
    app.lib_search_click.set(Some(row));
    let mut spans = vec![Span::styled(" ⌕ ", theme::brand_cyan())];
    if app.lib.filter_editing {
        spans.push(Span::styled(app.lib.filter.clone(), theme::text()));
        spans.push(Span::styled("▏", theme::brand_cyan()));
    } else if !app.lib.filter.is_empty() {
        spans.push(Span::styled(app.lib.filter.clone(), theme::text()));
        spans.push(Span::styled(
            format!("  — {shown} of {total} · / edits"),
            theme::dim(),
        ));
    } else {
        // The hint names both ways in. "click" is honest: the rect published
        // above is exactly what `events::on_mouse` tests.
        spans.push(Span::styled("search models — / or click", theme::dim()));
    }
    let mut line = Line::from(spans);
    if app.lib.filter_editing {
        line = line.style(theme::selected());
    }
    f.render_widget(Paragraph::new(line), row);
}

fn draw_list(f: &mut Frame, app: &App, area: Rect) {
    let rows = app.lib.visible();
    // The freshness of the recipe list belongs in the title: it is context for
    // every row, not a property of any one of them.
    let status = if app.lib.fetching {
        format!(
            " {} fetching ─",
            theme::SPINNER[(app.tick as usize / 2) % theme::SPINNER.len()]
        )
    } else {
        format!(" recipes {} ─", app.lib.index.status_text())
    };
    // Kept short: a long title is the first thing to overflow on a narrow
    // terminal, and the separators are decoration, not information. The
    // filter moved out of the title and into the persistent field below.
    let block = panel(format!("MODELS ─ {} ─{status} ─", rows.len()), true);
    let mut inner = block.inner(area);
    f.render_widget(block, area);

    // The search field owns the first inner row; the list gets the rest.
    if inner.height >= 1 {
        let field = Rect { height: 1, ..inner };
        draw_search_field(f, app, field, app.lib.rows.len(), rows.len());
        inner.y += 1;
        inner.height -= 1;
    }

    // Why the recipe list is stale belongs on screen, not only in a log. The
    // title has room for "offline"; the cause is what the reader has to act on,
    // and it differs completely — no route needs a proxy, a 403 needs a wait.
    let mut header: Vec<Line> = Vec::new();
    if let Some(detail) = app.lib.index.offline_detail() {
        header.extend(wrap(
            &detail,
            inner.width.saturating_sub(2) as usize,
            theme::warn(),
        ));
        header.push(Line::from(""));
    }

    if rows.is_empty() {
        let hint = if app.lib.filter.is_empty() {
            "no models or recipes yet — press r to fetch recipes"
        } else {
            "nothing matches this search"
        };
        header.push(Line::from(Span::styled(format!(" {hint}"), theme::dim())));
        f.render_widget(Paragraph::new(header), inner);
        return;
    }

    // Three lines per row; keep the selection on screen.
    let per_row = 3usize;
    let visible = (inner.height as usize / per_row).max(1);
    let first = app.lib.selected.saturating_sub(visible.saturating_sub(1));

    let mut lines: Vec<Line> = header;
    for (i, entry) in rows.iter().enumerate().skip(first).take(visible) {
        let selected = i == app.lib.selected;
        let bar = if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        // A checkmark means "the weights are here", nothing else. A download
        // in flight owns the glyph while it runs.
        let mark = if app.download.is_downloading(&entry.model) {
            Span::styled("↓ ", theme::brand_cyan())
        } else if entry.has_weights() {
            Span::styled("✓ ", theme::brand_green())
        } else if entry.local.is_some() {
            Span::styled("◐ ", theme::warn())
        } else {
            Span::styled("· ", theme::dim())
        };
        let name_style = if selected {
            theme::text().add_modifier(Modifier::BOLD)
        } else {
            theme::text()
        };
        let mut head_spans = vec![bar, mark, Span::styled(entry.model.clone(), name_style)];
        // A live download marks its row inline, right after the name: the row
        // is what the eye tracks in a list, and the glyph column alone is easy
        // to miss. It glows — bold and dim alternating on the tick — because a
        // static dot next to a stalled job and one next to a moving job would
        // look identical.
        if let Some(job) = app.download.job.as_ref().filter(|j| j.repo == entry.model) {
            // CYAN, not amber. Amber in this palette means "something needs
            // your attention" — the update badge, the loading lamp, the word
            // `stopping` — and a healthy transfer wearing it invites the
            // question "is something wrong?". Cyan is the momentum colour and
            // is what the header chip uses, so the two beat as one system.
            //
            // While stopping the pulse STOPS and the colour becomes amber:
            // that is a state needing attention, and the halted pulse is the
            // only signal that survives NO_COLOR.
            let glow = if job.cancelling {
                theme::warn().add_modifier(Modifier::DIM)
            } else if (app.tick / 4).is_multiple_of(2) {
                theme::brand_cyan().add_modifier(Modifier::BOLD)
            } else {
                theme::brand_cyan().add_modifier(Modifier::DIM)
            };
            head_spans.push(Span::styled(" ●", glow));
        }
        let mut head = Line::from(head_spans);
        if selected {
            head = head.style(theme::selected());
        }
        lines.push(head);

        let mut second = vec![Span::raw("   ")];
        second.extend(badges(app, entry));
        let subtitle = entry.subtitle();
        if !subtitle.is_empty() {
            second.push(Span::styled(format!(" {subtitle}"), theme::dim()));
        }
        lines.push(Line::from(second));
        // Line three is either the model's size, or — while it is being
        // fetched — that same line turned into a progress bar. Reusing the
        // line is deliberate: `per_row` stays 3, so no scroll arithmetic
        // changes, and the progress stays attached to the model it belongs to
        // instead of floating in a modal.
        lines.push(match progress_line(app, &entry.model, inner.width) {
            Some(l) => l,
            None => Line::from(vec![
                Span::raw("   "),
                Span::styled(entry.size_text(), theme::text2()),
                Span::styled(
                    match entry.recipes.len() {
                        // Not a dead end any more: Enter opens synthesized
                        // starting points (see `lib_start`), and the row is
                        // where that has to be discoverable.
                        0 => "  ·  no recipe — ⏎ starting points".to_string(),
                        // One recipe still names itself; several become a count,
                        // because listing three stems is what the card view is for.
                        1 => format!("  ·  {}", entry.recipes[0].id),
                        n => format!("  ·  {n} recipes"),
                    },
                    theme::dim(),
                ),
            ]),
        });
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// The progress line for a model, if one is being downloaded.
///
/// Degrades right-to-left as the pane narrows — rate first, then the file
/// counter, then the byte pair — so the bar and percentage, which are the
/// parts that answer "is this still moving", survive longest.
fn progress_line(app: &App, model: &str, width: u16) -> Option<Line<'static>> {
    let job = app.download.job.as_ref().filter(|j| j.repo == model)?;
    let mut spans = vec![Span::raw("   ")];

    // A spinner ALWAYS, in front of whatever else this line says. On a 20 GB
    // model the first several minutes are under one percent, and a bar that
    // renders as twelve empty cells next to "0%" is indistinguishable from a
    // download that never started — which is exactly how this was first
    // reported. The spinner is the part that says "still moving" when the
    // numbers cannot yet.
    let phase = (app.tick as usize / 2) % theme::SPINNER.len();
    spans.push(Span::styled(theme::SPINNER[phase], theme::brand_cyan()));
    spans.push(Span::raw(" "));

    match job.fraction() {
        Some(f) => {
            const CELLS: usize = 12;
            // At least one cell once ANY bytes have moved: rounding 0.75% of
            // twelve cells to zero drew an empty bar for a download that was
            // running fine.
            let exact = (f * CELLS as f64).round() as usize;
            let filled = if job.done > 0 { exact.max(1) } else { exact };
            spans.push(Span::styled(
                "▓".repeat(filled.min(CELLS)),
                theme::brand_cyan(),
            ));
            spans.push(Span::styled(
                "░".repeat(CELLS.saturating_sub(filled)),
                theme::dim(),
            ));
            // One decimal below 10%, because "0%" for twenty minutes of real
            // progress reads as a stall. Whole numbers above that. The RULE
            // lives in `format::percent` so the header chip cannot drift from
            // this row; the width here is just column alignment.
            spans.push(Span::styled(
                format!("  {:>5}", crate::tui::format::percent(f)),
                theme::text(),
            ));
        }
        // The Hub did not report sizes, so there is no fraction to draw.
        None => spans.push(Span::styled("fetching", theme::text())),
    }

    if job.cancelling {
        // Cancellation is honoured within a chunk, but saying so beats a bar
        // that keeps moving after the user asked it to stop.
        spans.push(Span::styled("  stopping…", theme::warn()));
        return Some(Line::from(spans));
    }
    if width >= 60 && job.total > 0 {
        spans.push(Span::styled(
            format!("  {} / {}", gb(job.done), gb(job.total)),
            theme::text2(),
        ));
    }
    // Rate BEFORE the file counter, and at a much lower threshold. The list
    // pane is roughly half the content area, so at a 200-column terminal this
    // line gets ~85 columns — the old 96 threshold meant the rate, the one
    // field that proves bytes are moving, never appeared at any realistic
    // window size.
    if width >= 70 && job.rate_bps > 0.0 {
        spans.push(Span::styled(
            format!("  {}", crate::tui::format::rate(job.rate_bps)),
            theme::dim(),
        ));
    }
    if width >= 88
        && let Some((i, of, _)) = &job.file
    {
        spans.push(Span::styled(format!("  file {i}/{of}"), theme::dim()));
    }
    Some(Line::from(spans))
}

/// ★ This used to divide by 10⁹ while the Library card that replaces this row
/// on completion divided by 1024³ — so a checkpoint downloaded as "20.0 GB"
/// became "18.6 GB" the moment it finished, for no reason the user could see.
fn gb(bytes: u64) -> String {
    crate::tui::format::bytes(bytes)
}
