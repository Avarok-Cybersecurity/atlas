// SPDX-License-Identifier: AGPL-3.0-only

//! How a contamination run is presented. Split from the state machine the way
//! `bfcl/report.rs` is: the run logic and the rendering read separately, and
//! everything here is a pure function of the [`Score`] — table-testable with
//! no server.

use std::collections::BTreeMap;

use super::prompts;
use super::score::{Class, Score};
use crate::result::{Cell, CellStyle, Column, ResultTable, Stat};

/// One class, rendered. `(what, style, detail)` — the detail column carries
/// the part of the finding that does not fit a fixed-width cell.
fn render_class(class: &Class) -> (String, CellStyle, String) {
    match class {
        Class::Identical => ("identical".into(), CellStyle::Good, String::new()),
        Class::Diverged { at, detail } => (
            format!("DIVERGED @ char {at}"),
            CellStyle::Bad,
            detail.clone(),
        ),
        Class::Contaminated { foreign } => (
            "CONTAMINATED".into(),
            CellStyle::Bad,
            format!("carries foreign canary {foreign}"),
        ),
        Class::Persistent { at } => (
            format!("PERSISTENT @ char {at}"),
            CellStyle::Bad,
            "state survived into solo execution".into(),
        ),
        Class::AloneUnstable => (
            "alone-unstable".into(),
            CellStyle::Warn,
            "two solo runs disagree (#435) — contamination unattributable".into(),
        ),
        Class::Unmeasured { why } => ("unmeasured".into(), CellStyle::Warn, why.clone()),
    }
}

/// One row per `(probe, leg)` cell, in the scorer's own (BTreeMap) order.
pub(super) fn table(s: &Score) -> ResultTable {
    let mut t = ResultTable::new(
        "PROBE × LEG",
        vec![
            Column::left("Probe", 6),
            Column::left("Leg", 6),
            Column::left("Result", 20),
            Column::left("Detail", 44),
        ],
    );
    for ((probe_idx, leg), class) in &s.cells {
        let name = prompts::PROBES
            .get(*probe_idx)
            .map(|p| p.name)
            .unwrap_or("?");
        let (what, style, detail) = render_class(class);
        t.push(vec![
            Cell::new(name),
            Cell::styled(leg.clone(), CellStyle::Dim),
            Cell::styled(what, style),
            Cell::styled(detail, CellStyle::Dim),
        ]);
    }
    t
}

fn count_stat(label: &str, n: usize) -> Stat {
    Stat::new(label, n.to_string(), "").with_style(if n == 0 {
        CellStyle::Good
    } else {
        CellStyle::Bad
    })
}

pub(super) fn summary(s: &Score) -> Vec<Stat> {
    vec![
        Stat::new("Identical", format!("{}/{}", s.identical, s.compared), "").with_style(
            if s.compared > 0 && s.identical == s.compared {
                CellStyle::Good
            } else {
                CellStyle::Bad
            },
        ),
        count_stat("Contaminated", s.contaminated),
        count_stat("Persistent", s.persistent),
        count_stat("Diverged", s.diverged),
        Stat::new("Tokens compared", s.tokens_compared.to_string(), "")
            .with_style(CellStyle::Accent),
    ]
}

/// Raw gate numbers, same source the summary tiles read from. Every class is a
/// key even when zero: a missing key and a zero must stay distinguishable to
/// whatever compares records later.
pub(super) fn metrics(s: &Score) -> BTreeMap<String, f64> {
    [
        ("compared", s.compared),
        ("identical", s.identical),
        ("diverged", s.diverged),
        ("contaminated", s.contaminated),
        ("persistent", s.persistent),
        ("alone_unstable", s.alone_unstable),
        ("unmeasured", s.unmeasured),
        ("tokens_compared", s.tokens_compared),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v as f64))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::super::score::{Class, Score};
    use super::*;
    use crate::result::CellStyle;

    fn score_with(cells: Vec<((usize, &str), Class)>) -> Score {
        let mut s = Score::default();
        for ((i, leg), c) in cells {
            match &c {
                Class::Identical => s.identical += 1,
                Class::Diverged { .. } => s.diverged += 1,
                Class::Contaminated { .. } => s.contaminated += 1,
                Class::Persistent { .. } => s.persistent += 1,
                Class::AloneUnstable => s.alone_unstable += 1,
                Class::Unmeasured { .. } => s.unmeasured += 1,
            }
            s.compared += 1;
            s.cells.insert((i, leg.to_string()), c);
        }
        s
    }

    /// Positive: every cell becomes a row, the contaminated one names the
    /// foreign canary and is styled Bad. Negative: the clean cell is NOT
    /// rendered as a finding.
    #[test]
    fn every_class_renders_its_evidence_and_severity() {
        for (class, expected) in [
            (Class::Identical, ("identical", CellStyle::Good, "")),
            (
                Class::Diverged {
                    at: 4,
                    detail: "streams differ".into(),
                },
                ("DIVERGED @ char 4", CellStyle::Bad, "streams differ"),
            ),
            (
                Class::Contaminated {
                    foreign: "FOREIGN".into(),
                },
                (
                    "CONTAMINATED",
                    CellStyle::Bad,
                    "carries foreign canary FOREIGN",
                ),
            ),
            (
                Class::Persistent { at: 7 },
                (
                    "PERSISTENT @ char 7",
                    CellStyle::Bad,
                    "state survived into solo execution",
                ),
            ),
            (
                Class::AloneUnstable,
                (
                    "alone-unstable",
                    CellStyle::Warn,
                    "two solo runs disagree (#435) — contamination unattributable",
                ),
            ),
            (
                Class::Unmeasured {
                    why: "timeout".into(),
                },
                ("unmeasured", CellStyle::Warn, "timeout"),
            ),
        ] {
            let rendered = render_class(&class);
            assert_eq!(
                (rendered.0.as_str(), rendered.1, rendered.2.as_str()),
                expected
            );
        }

        let s = score_with(vec![
            ((0, "c2"), Class::Identical),
            (
                (0, "c4"),
                Class::Contaminated {
                    foreign: "QJ-CRIMSON-OTTER-77".into(),
                },
            ),
        ]);
        let t = table(&s);
        assert_eq!(t.rows.len(), 2, "one row per cell");
        let bad: Vec<_> = t
            .rows
            .iter()
            .filter(|r| r[2].style == CellStyle::Bad)
            .collect();
        assert_eq!(bad.len(), 1, "exactly the contaminated cell is Bad");
        assert!(bad[0][2].text.contains("CONTAMINATED"));
        assert!(
            bad[0][3].text.contains("QJ-CRIMSON-OTTER-77"),
            "the operator's first question is WHOSE state leaked; the row must \
             answer it: {:?}",
            bad[0][3].text
        );
        let clean = t
            .rows
            .iter()
            .find(|r| r[1].text == "c2")
            .expect("the c2 row");
        assert_eq!(clean[2].text, "identical");
        assert_eq!(clean[2].style, CellStyle::Good);
    }

    #[test]
    fn probe_rows_carry_the_probe_name_not_an_index() {
        let s = score_with(vec![((1, "post"), Class::Persistent { at: 7 })]);
        let t = table(&s);
        assert_eq!(t.rows[0][0].text, prompts::PROBES[1].name);
        assert!(t.rows[0][2].text.contains("PERSISTENT @ char 7"));
    }

    /// Positive: a clean score summarises green. Negative: one leak flips the
    /// Identical tile to Bad — a red run must not render a green headline.
    #[test]
    fn summary_headline_tracks_the_verdict_direction() {
        let clean = score_with(vec![((0, "c2"), Class::Identical)]);
        assert_eq!(summary(&clean)[0].style, CellStyle::Good);

        let dirty = score_with(vec![
            ((0, "c2"), Class::Identical),
            (
                (1, "c2"),
                Class::Diverged {
                    at: 3,
                    detail: "streams differ".into(),
                },
            ),
            (
                (0, "c4"),
                Class::Contaminated {
                    foreign: "foreign".into(),
                },
            ),
            ((1, "post"), Class::Persistent { at: 9 }),
        ]);
        let mut dirty = dirty;
        dirty.tokens_compared = 42;
        let tiles: Vec<_> = summary(&dirty)
            .into_iter()
            .map(|tile| (tile.label, tile.value, tile.style))
            .collect();
        assert_eq!(
            tiles,
            vec![
                ("Identical".into(), "1/4".into(), CellStyle::Bad),
                ("Contaminated".into(), "1".into(), CellStyle::Bad),
                ("Persistent".into(), "1".into(), CellStyle::Bad),
                ("Diverged".into(), "1".into(), CellStyle::Bad),
                ("Tokens compared".into(), "42".into(), CellStyle::Accent),
            ]
        );
    }

    /// Every class is a metric key even at zero — absence and zero must stay
    /// distinguishable to a record comparator.
    #[test]
    fn metrics_carry_every_class_even_when_zero() {
        let s = score_with(vec![((0, "c2"), Class::Identical)]);
        let m = metrics(&s);
        for key in [
            "compared",
            "identical",
            "diverged",
            "contaminated",
            "persistent",
            "alone_unstable",
            "unmeasured",
            "tokens_compared",
        ] {
            assert!(m.contains_key(key), "missing metric {key}");
        }
        assert_eq!(m["identical"], 1.0);
        assert_eq!(m["contaminated"], 0.0, "zero, not absent");

        // And the values TRACK the score, rather than being present-but-frozen
        // — a metric that always reads 0 is indistinguishable from a clean run.
        let dirty = Score {
            compared: 8,
            identical: 7,
            diverged: 6,
            contaminated: 5,
            persistent: 4,
            alone_unstable: 3,
            unmeasured: 2,
            tokens_compared: 1,
            ..Default::default()
        };
        assert_eq!(
            metrics(&dirty),
            [
                ("alone_unstable".to_string(), 3.0),
                ("compared".to_string(), 8.0),
                ("contaminated".to_string(), 5.0),
                ("diverged".to_string(), 6.0),
                ("identical".to_string(), 7.0),
                ("persistent".to_string(), 4.0),
                ("tokens_compared".to_string(), 1.0),
                ("unmeasured".to_string(), 2.0),
            ]
            .into_iter()
            .collect()
        );
    }
}
