// SPDX-License-Identifier: AGPL-3.0-only

//! What a benchmark emits on every `next()` — the frame the TUI renders.
//!
//! The types here are deliberately presentation-shaped but style-free: a cell
//! carries a semantic [`CellStyle`] (`Good` / `Warn` / `Bad`), never a color.
//! Colors stay in the TUI's `theme`, so the palette is changed in one place and
//! a benchmark cannot introduce an off-brand shade.
//!
//! Frames are serializable: the last frame of a run is persisted under
//! `~/.atlas/runs/` and re-rendered by the History pane with the same code.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Where a run is. The `run()` stream ends after the first terminal status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    /// More frames are coming; `next()` will be called again.
    Running,
    /// The benchmark finished its work. Terminal.
    Completed,
    /// The benchmark stopped early and the result is not trustworthy. Terminal.
    Failed,
}

impl RunStatus {
    pub fn is_terminal(self) -> bool {
        !matches!(self, RunStatus::Running)
    }
}

/// Semantic emphasis. The TUI maps these onto the brand palette.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellStyle {
    #[default]
    Neutral,
    Dim,
    /// Brand cyan — the value this row exists to show.
    Accent,
    Good,
    Warn,
    Bad,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Align {
    #[default]
    Left,
    Right,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Column {
    pub title: String,
    pub align: Align,
    /// Preferred width in cells. The renderer may shrink it to fit.
    pub width: u16,
}

impl Column {
    pub fn left(title: impl Into<String>, width: u16) -> Self {
        Self {
            title: title.into(),
            align: Align::Left,
            width,
        }
    }
    /// Numeric columns right-align so digits line up down the column.
    pub fn right(title: impl Into<String>, width: u16) -> Self {
        Self {
            title: title.into(),
            align: Align::Right,
            width,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Cell {
    pub text: String,
    pub style: CellStyle,
}

impl Cell {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            style: CellStyle::Neutral,
        }
    }
    pub fn styled(text: impl Into<String>, style: CellStyle) -> Self {
        Self {
            text: text.into(),
            style,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResultTable {
    pub title: String,
    pub columns: Vec<Column>,
    pub rows: Vec<Vec<Cell>>,
}

impl ResultTable {
    pub fn new(title: impl Into<String>, columns: Vec<Column>) -> Self {
        Self {
            title: title.into(),
            columns,
            rows: Vec::new(),
        }
    }
    pub fn push(&mut self, row: Vec<Cell>) {
        self.rows.push(row);
    }
}

/// A headline number for the tile row above the table.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stat {
    pub label: String,
    pub value: String,
    pub unit: String,
    pub style: CellStyle,
}

impl Stat {
    pub fn new(
        label: impl Into<String>,
        value: impl Into<String>,
        unit: impl Into<String>,
    ) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            unit: unit.into(),
            style: CellStyle::Neutral,
        }
    }
    pub fn with_style(mut self, style: CellStyle) -> Self {
        self.style = style;
        self
    }
}

/// A gate outcome. `Info` is for benchmarks that measure without gating, so a
/// sweep never renders a green PASS it did not actually earn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerdictKind {
    Pass,
    Fail,
    Info,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Verdict {
    pub kind: VerdictKind,
    /// Why. Always state the measured value against the threshold — a bare
    /// "FAIL" sends the reader back to the raw logs.
    pub reason: String,
}

impl Verdict {
    pub fn pass(reason: impl Into<String>) -> Self {
        Self {
            kind: VerdictKind::Pass,
            reason: reason.into(),
        }
    }
    pub fn fail(reason: impl Into<String>) -> Self {
        Self {
            kind: VerdictKind::Fail,
            reason: reason.into(),
        }
    }
    pub fn info(reason: impl Into<String>) -> Self {
        Self {
            kind: VerdictKind::Info,
            reason: reason.into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogLine {
    pub level: LogLevel,
    pub text: String,
}

impl LogLine {
    pub fn info(text: impl Into<String>) -> Self {
        Self {
            level: LogLevel::Info,
            text: text.into(),
        }
    }
    pub fn warn(text: impl Into<String>) -> Self {
        Self {
            level: LogLevel::Warn,
            text: text.into(),
        }
    }
    pub fn error(text: impl Into<String>) -> Self {
        Self {
            level: LogLevel::Error,
            text: text.into(),
        }
    }
}

/// One frame of a benchmark run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub status: RunStatus,
    /// What is happening right now, e.g. `"warmup"` or `"isl 1024 · conc 8"`.
    pub phase: String,
    /// `(done, total)` for the progress bar; `None` when the total is unknown.
    pub progress: Option<(u64, u64)>,
    pub summary: Vec<Stat>,
    pub table: Option<ResultTable>,
    pub verdict: Option<Verdict>,
    /// Raw headline numbers, keyed by stable metric name. The PR gate compares
    /// these against `baseline.json` — parsing the human-formatted `summary`
    /// strings would couple two presentation layers, so the numbers live here
    /// once and both layers read them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metrics: BTreeMap<String, f64>,
    /// Lines appended to the run log since the previous frame (not cumulative).
    pub log: Vec<LogLine>,
    pub elapsed: Duration,
    /// What state the box was in before and after this run.
    ///
    /// Stamped by [`crate::executor`] onto the TERMINAL frame only, which is
    /// the frame [`crate::RunRecord`] keeps and the gate record is built from.
    /// Absent on every intermediate frame, and absent from the JSON entirely
    /// for a record written before this existed — a run with no hardware state
    /// is unmeasured, which is not the same as a run measured on a healthy
    /// box, and no reader should be able to confuse the two.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_state: Option<crate::hardware::HardwareStateReport>,
    /// Content identity of the dataset the run scored against, when the
    /// benchmark has one (e.g. `file-sha256:…;draw-sha256:…` for the MLPerf
    /// agentic leg). Carried into the gate record: the metrics map is
    /// f64-only, so without this a record can pin a draw's SIZE but not its
    /// CONTENT — and two same-size draws of different content are exactly the
    /// incomparable pair the BFCL notes warn about. `None` for benchmarks
    /// without a dataset; absent from older persisted frames.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_fingerprint: Option<String>,
}

impl BenchmarkResult {
    pub fn running(phase: impl Into<String>, elapsed: Duration) -> Self {
        Self {
            status: RunStatus::Running,
            phase: phase.into(),
            progress: None,
            summary: Vec::new(),
            table: None,
            verdict: None,
            metrics: BTreeMap::new(),
            log: Vec::new(),
            elapsed,
            hardware_state: None,
            dataset_fingerprint: None,
        }
    }

    pub fn completed(phase: impl Into<String>, elapsed: Duration) -> Self {
        Self {
            status: RunStatus::Completed,
            ..Self::running(phase, elapsed)
        }
    }

    pub fn failed(phase: impl Into<String>, reason: impl Into<String>, elapsed: Duration) -> Self {
        let reason = reason.into();
        Self {
            status: RunStatus::Failed,
            verdict: Some(Verdict::fail(reason.clone())),
            log: vec![LogLine::error(reason)],
            ..Self::running(phase, elapsed)
        }
    }

    pub fn with_progress(mut self, done: u64, total: u64) -> Self {
        self.progress = Some((done, total));
        self
    }
    pub fn with_summary(mut self, summary: Vec<Stat>) -> Self {
        self.summary = summary;
        self
    }
    pub fn with_metrics(mut self, metrics: BTreeMap<String, f64>) -> Self {
        self.metrics = metrics;
        self
    }
    pub fn with_table(mut self, table: ResultTable) -> Self {
        self.table = Some(table);
        self
    }
    pub fn with_verdict(mut self, verdict: Verdict) -> Self {
        self.verdict = Some(verdict);
        self
    }
    pub fn with_log(mut self, log: Vec<LogLine>) -> Self {
        self.log = log;
        self
    }
    pub fn log_line(mut self, line: LogLine) -> Self {
        self.log.push(line);
        self
    }
    pub fn with_hardware_state(mut self, report: crate::hardware::HardwareStateReport) -> Self {
        self.hardware_state = Some(report);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_running_is_non_terminal() {
        assert!(!RunStatus::Running.is_terminal());
        assert!(RunStatus::Completed.is_terminal());
        assert!(RunStatus::Failed.is_terminal());
    }

    #[test]
    fn failed_carries_the_reason_into_both_verdict_and_log() {
        let r = BenchmarkResult::failed("scoring", "endpoint refused", Duration::from_secs(1));
        assert_eq!(r.status, RunStatus::Failed);
        assert_eq!(r.phase, "scoring");
        assert_eq!(r.elapsed, Duration::from_secs(1));
        let verdict = r.verdict.as_ref().expect("failure verdict");
        assert_eq!(verdict.kind, VerdictKind::Fail);
        assert_eq!(verdict.reason, "endpoint refused");
        assert_eq!(r.log.len(), 1);
        assert_eq!(r.log[0].level, LogLevel::Error);
        assert_eq!(r.log[0].text, "endpoint refused");
    }

    #[test]
    fn frames_round_trip_through_json_for_the_history_pane() {
        let mut table = ResultTable::new("Samples", vec![Column::right("Rate", 8)]);
        table.push(vec![Cell::styled("119.3", CellStyle::Good)]);
        let mut r = BenchmarkResult::completed("done", Duration::from_millis(1500))
            .with_progress(2, 3)
            .with_summary(vec![
                Stat::new("Throughput", "119.3", "tok/s").with_style(CellStyle::Accent),
            ])
            .with_table(table)
            .with_metrics(BTreeMap::from([("median_tps".into(), 119.3)]))
            .with_verdict(Verdict::pass("median +0.4% (limit 3%)"))
            .with_log(vec![LogLine::warn("clock dipped")])
            .with_hardware_state(crate::hardware::HardwareStateReport::opened(
                crate::hardware::Sensitivity::Correctness,
                crate::hardware::HardwareState::default(),
            ));
        r.dataset_fingerprint = Some("file-sha256:abc".into());
        let json = serde_json::to_string(&r).unwrap();
        let back: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, RunStatus::Completed);
        assert_eq!(back.phase, "done");
        assert_eq!(back.progress, Some((2, 3)));
        assert_eq!(back.elapsed, Duration::from_millis(1500));
        assert_eq!(back.summary.len(), 1);
        assert_eq!(back.summary[0].label, "Throughput");
        assert_eq!(back.summary[0].value, "119.3");
        assert_eq!(back.summary[0].unit, "tok/s");
        assert_eq!(back.summary[0].style, CellStyle::Accent);
        let table = back.table.as_ref().expect("table");
        assert_eq!(table.title, "Samples");
        assert_eq!(table.columns.len(), 1);
        assert_eq!(table.columns[0].title, "Rate");
        assert_eq!(table.columns[0].align, Align::Right);
        assert_eq!(table.columns[0].width, 8);
        assert_eq!(table.rows.len(), 1);
        assert_eq!(table.rows[0][0].text, "119.3");
        assert_eq!(table.rows[0][0].style, CellStyle::Good);
        let verdict = back.verdict.as_ref().expect("verdict");
        assert_eq!(verdict.kind, VerdictKind::Pass);
        assert_eq!(verdict.reason, "median +0.4% (limit 3%)");
        assert_eq!(back.metrics, BTreeMap::from([("median_tps".into(), 119.3)]));
        assert_eq!(back.log.len(), 1);
        assert_eq!(back.log[0].level, LogLevel::Warn);
        assert_eq!(back.log[0].text, "clock dipped");
        assert!(back.hardware_state.is_some());
        assert_eq!(back.dataset_fingerprint.as_deref(), Some("file-sha256:abc"));
    }
}
