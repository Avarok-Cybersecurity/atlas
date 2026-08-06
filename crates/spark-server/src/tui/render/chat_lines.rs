// SPDX-License-Identifier: AGPL-3.0-only

//! One transcript entry → display rows.
//!
//! The chat pane slices its viewport on the rows this returns, so they must be
//! exactly what renders — measured in display columns, not bytes.
//!
//! The visual grammar, all of it borrowed from the existing design system
//! (`tui/theme.rs`) rather than invented here:
//!
//! ```text
//! ❯ what the user asked                     purple chevron, full-strength text
//! ⬢ ┆ ⠹ thinking 4.2s                       cyan spinner, dim dashed rule
//!   ┆ muted reasoning, streaming live       TEXT_2 behind the dashed rule
//!   ▏the answer                             solid cyan rule, full-strength text
//!   ttft 412ms · 247 think + 312 tok        dim footer
//! ```
//!
//! Reasoning is subordinate chrome: dimmer text, a *dashed* rule against the
//! answer's solid one, and — once the answer lands — one quiet summary line.
//! The two are distinguishable at a glance without reading a word, which is
//! the property that matters.

use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::tui::chat::{ChatMessage, Role};
use crate::tui::chat_thinking::{ThinkingView, dur};
use crate::tui::theme;

/// How much of a live reasoning trace the collapsed view keeps on screen.
///
/// Enough to see that it is moving and roughly what it is chewing on; not so
/// much that a thousand-token ramble owns the pane. The expanded view is the
/// place to read all of it.
const PREVIEW_ROWS: usize = 6;

/// The answer's rule: solid, brand cyan, unchanged from before reasoning
/// existed — a reply with no reasoning must render exactly as it always did.
fn rule_answer() -> Span<'static> {
    Span::styled("▏", theme::brand_cyan())
}

/// The reasoning rule: dashed and dim. Deliberately NOT cyan — cyan is the
/// activity/focus role, and the answer owns it here.
fn rule_think() -> Span<'static> {
    Span::styled("┆", theme::dim())
}

fn model_body() -> Style {
    theme::text().bg(theme::BG_PANEL.color())
}

/// Word-wrap `text` into display rows of at most `width` columns.
///
/// Measured in display columns via `unicode-width`, not `str::len`, or CJK and
/// emoji replies would compute a tail that is short by a row per line. A word
/// longer than the pane is hard-split rather than allowed to overhang.
pub(super) fn wrap_rows(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let mut rows = Vec::new();
    for logical in text.split('\n') {
        let (mut cur, mut cur_w) = (String::new(), 0usize);
        for word in logical.split_inclusive(' ') {
            let w = UnicodeWidthStr::width(word);
            if cur_w + w > width && !cur.is_empty() {
                rows.push(std::mem::take(&mut cur));
                cur_w = 0;
            }
            if w > width {
                for ch in word.chars() {
                    let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
                    if cur_w + cw > width {
                        rows.push(std::mem::take(&mut cur));
                        cur_w = 0;
                    }
                    cur.push(ch);
                    cur_w += cw;
                }
            } else {
                cur.push_str(word);
                cur_w += w;
            }
        }
        rows.push(cur);
    }
    rows
}

/// Truncate to `width` display columns, marking the cut.
///
/// The header is a single unwrappable row, so it is the one place in the pane
/// that must clip rather than reflow.
fn clip(s: &str, width: usize) -> String {
    if UnicodeWidthStr::width(s) <= width {
        return s.to_string();
    }
    let mut out = String::new();
    let mut w = 0usize;
    for ch in s.chars() {
        let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
        if w + cw + 1 > width {
            break;
        }
        out.push(ch);
        w += cw;
    }
    out.push('…');
    out
}

/// The reply footer, honestly labelled.
///
/// `ttft` keeps meaning time-to-first-token and now actually measures it;
/// `answer` is the separate, genuinely different number that explains a pane
/// which sat blank while the model thought. Reasoning tokens are named as
/// such rather than folded into the answer's count, because the decode rate is
/// computed over both and a footer whose numbers do not multiply out is worse
/// than no footer.
fn footer(m: &ChatMessage, wide: bool) -> Option<String> {
    if m.ttft_ms.is_none() && m.tok_per_s.is_none() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    match m.ttft_ms {
        Some(v) => parts.push(format!("ttft {}", dur(v / 1000.0))),
        None => parts.push("ttft —".into()),
    }
    // Only worth a column when it says something `ttft` did not.
    if wide
        && !m.reasoning.is_empty()
        && let Some(v) = m.answer_ttft_ms
    {
        parts.push(format!("answer {}", dur(v / 1000.0)));
    }
    if m.reasoning.tokens > 0 {
        parts.push(format!("{} think + {} tok", m.reasoning.tokens, m.tokens));
    } else {
        parts.push(format!("{} tok", m.tokens));
    }
    if let Some(tps) = m.tok_per_s.filter(|t| *t > 0.0) {
        parts.push(format!("{:.1} ms/tok", 1000.0 / tps));
    }
    Some(parts.join(" · "))
}

/// The reasoning block: header plus whatever body the view calls for.
fn thinking_rows(
    rows: &mut Vec<Vec<Span<'static>>>,
    m: &ChatMessage,
    live: bool,
    view: ThinkingView,
    tick: u64,
    body_w: usize,
) {
    // A model that emitted no reasoning gets no header, no summary and no
    // wasted row — it renders exactly as it did before any of this existed.
    if m.reasoning.is_empty() || view == ThinkingView::Hidden {
        return;
    }
    let secs = m.reasoning.seconds().unwrap_or(0.0);
    // The one spinner on screen: only the streaming tip can be live, and it
    // stops the moment the first answer token lands.
    let (lead, lead_style, rest) = if live {
        (
            format!(" {} ", theme::SPINNER[(tick % 10) as usize]),
            theme::brand_cyan(),
            format!("thinking {}", dur(secs)),
        )
    } else {
        let caret = if view == ThinkingView::Expanded {
            " ▾ "
        } else {
            " ▸ "
        };
        let n = m.reasoning.tokens;
        // The long form is the one that reads well; the short one exists so
        // the 80x24 floor still gets a sentence instead of an ellipsis.
        let rest = if body_w >= 34 {
            format!("thought for {} · {n} tokens", dur(secs))
        } else {
            format!("thought {} · {n} tok", dur(secs))
        };
        (caret.to_string(), theme::dim(), rest)
    };
    let room = body_w.saturating_sub(UnicodeWidthStr::width(lead.as_str()) + 1);
    rows.push(vec![
        rule_think(),
        Span::styled(lead, lead_style),
        Span::styled(clip(&rest, room), theme::text2()),
    ]);
    // Expanded shows all of it; collapsed shows a live tail while it streams
    // and nothing at all once the answer has arrived to take the stage.
    let limit = match (view, live) {
        (ThinkingView::Expanded, _) => usize::MAX,
        (_, true) => PREVIEW_ROWS,
        (_, false) => return,
    };
    // One column of the rule's own, so `┆` never abuts the text.
    let wrapped = wrap_rows(&m.reasoning.text, body_w.saturating_sub(1));
    let skip = wrapped.len().saturating_sub(limit);
    for r in wrapped.into_iter().skip(skip) {
        rows.push(vec![
            rule_think(),
            Span::styled(format!(" {r}"), theme::text2()),
        ]);
    }
}

fn answer_rows(rows: &mut Vec<Vec<Span<'static>>>, m: &ChatMessage, is_tip: bool, body_w: usize) {
    if m.is_answerless() {
        // Observed with `response_format` + thinking on: 2 of 4 requests
        // returned reasoning and no content at all. Say so — a pane that just
        // stops looks like the bug is here.
        let msg = if m.reasoning.is_empty() {
            "(no answer — the reply produced nothing)"
        } else {
            "(no answer — the model stopped after thinking)"
        };
        rows.push(vec![rule_answer(), Span::styled(msg, theme::warn())]);
        return;
    }
    // While reasoning streams there is no answer yet, and an empty cyan rule
    // under the thinking block would just be a stray glyph. The cursor rides
    // the last reasoning row instead.
    if m.text.is_empty() && is_tip && !rows.is_empty() {
        return;
    }
    for r in wrap_rows(&m.text, body_w) {
        rows.push(vec![rule_answer(), Span::styled(r, model_body())]);
    }
}

/// One transcript entry as display rows, gutter included.
///
/// `is_tip` marks the live streaming message — the only one that may carry a
/// spinner or a cursor.
pub(super) fn message_lines(
    m: &ChatMessage,
    is_tip: bool,
    view: ThinkingView,
    tick: u64,
    body_w: usize,
) -> Vec<Line<'static>> {
    let mut rows: Vec<Vec<Span<'static>>> = Vec::new();
    match m.role {
        Role::User => {
            for r in wrap_rows(&m.text, body_w) {
                rows.push(vec![Span::raw(""), Span::styled(r, theme::text())]);
            }
        }
        Role::Model => {
            // Still thinking = the tip, with nothing readable emitted yet.
            // Reasoning that keeps arriving AFTER the answer started just
            // accumulates into the block; the header stops spinning, because
            // what the user is now watching is the answer.
            let live = is_tip && m.text.is_empty();
            thinking_rows(&mut rows, m, live, view, tick, body_w);
            answer_rows(&mut rows, m, is_tip, body_w);
            if is_tip && let Some(last) = rows.last_mut() {
                last.push(Span::styled("▍", theme::brand_cyan()));
            }
            if let Some(f) = footer(m, body_w >= 60) {
                for r in wrap_rows(&f, body_w) {
                    rows.push(vec![Span::raw(""), Span::styled(r, theme::dim())]);
                }
            }
        }
    }
    let (glyph, gstyle) = match m.role {
        Role::User => ("❯ ", theme::brand_purple().add_modifier(Modifier::BOLD)),
        Role::Model => ("⬢ ", theme::brand_cyan()),
    };
    rows.into_iter()
        .enumerate()
        .map(|(i, mut spans)| {
            let mut line = vec![if i == 0 {
                Span::styled(glyph, gstyle)
            } else {
                Span::styled("  ", Style::default())
            }];
            line.append(&mut spans);
            Line::from(line)
        })
        .collect()
}

#[cfg(test)]
#[path = "chat_lines_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "chat_wide_tests.rs"]
mod wide_tests;
