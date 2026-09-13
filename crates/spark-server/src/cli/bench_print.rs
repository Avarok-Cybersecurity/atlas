// SPDX-License-Identifier: AGPL-3.0-only

//! Rendering for `spark benchmark`.
//!
//! **stdout carries the payload, stderr carries the commentary.** The plain
//! subscriber writes its log lines to stdout and that format is a grep
//! contract, so progress and banners go to stderr and only the report, list or
//! JSON goes to stdout. `spark benchmark run … --format json > run.json` is
//! then a clean file rather than a log with JSON in it.

use anyhow::Result;
use atlas_plugin::headless::{RunReporter, RunRequest};
use atlas_plugin::{
    BenchmarkResult, PluginEvent, RunRecord, VerdictKind, params::ParamSpec, registry,
};

use super::bench_args::OutputFormat;

/// The whole suite.
pub fn print_suite(format: OutputFormat) -> Result<()> {
    let all = registry::all();
    if format == OutputFormat::Json {
        let rows: Vec<_> = all
            .iter()
            .map(|d| {
                serde_json::json!({
                    "id": d.id, "name": d.name, "summary": d.summary,
                    "duration_hint": d.duration_hint,
                    "needs_confirmation": d.needs_confirmation,
                    "intended_for": d.intended_for.map(|e| e.families),
                    // Explicit, so a consumer never has to infer the structure
                    // from an id suffix. `members` is present only on a group;
                    // `member_of` only on a shard.
                    "members": atlas_plugin::gate::group::find(d.id).map(|g| g.members),
                    "member_of": atlas_plugin::gate::group::member_of(d.id).map(|g| g.id),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    // Members are printed UNDER their group, not as eleven more top-level rows.
    // Two reasons, and the second is not cosmetic: a reader scanning the suite
    // should see eleven gates, not nineteen entries of which eight are quarters
    // of two others; and `render/bench/list.rs` already records that the
    // ELEVENTH registered benchmark rendered nowhere at 160x48, so eight more
    // flat rows would push real gates off the screen.
    let width = all.iter().map(|d| d.id.len() + 2).max().unwrap_or(0);
    for d in all {
        if atlas_plugin::gate::group::member_of(d.id).is_some() {
            continue;
        }
        let mark = if d.needs_confirmation { " (--yes)" } else { "" };
        println!(
            "{:width$}  {:<10}  {}{}",
            d.id, d.duration_hint, d.summary, mark
        );
        let Some(group) = atlas_plugin::gate::group::find(d.id) else {
            continue;
        };
        for m in group.members {
            let Some(md) = registry::find(m) else {
                continue;
            };
            println!(
                "{:width$}  {:<10}  one shard; run all four, or run the group id above",
                format!("  {m}"),
                md.duration_hint
            );
        }
    }
    Ok(())
}

/// One benchmark's parameter schema, so `--param` can be written without guessing.
pub fn print_schema(id: &str, format: OutputFormat) -> Result<()> {
    let descriptor = super::bench_run::find(id)?;
    let specs = descriptor.build().parameters();
    if format == OutputFormat::Json {
        let rows: Vec<_> = specs
            .iter()
            .map(|s| {
                serde_json::json!({
                    "key": s.key, "label": s.label, "help": s.help,
                    "default": s.default.to_edit_string(),
                    "domain": s.kind.domain_hint(),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "id": descriptor.id, "name": descriptor.name,
                "detail": descriptor.detail, "parameters": rows,
            }))?
        );
        return Ok(());
    }
    println!("{}  —  {}", descriptor.id, descriptor.name);
    println!("{}\n", descriptor.detail);
    // Which checkpoint the numbers mean something for. Printed before the
    // parameters because it decides whether the run is worth configuring.
    if let Some(expect) = descriptor.intended_for {
        println!("  defined on   {}", expect.families.join(" | "));
        println!("  {}\n", expect.note);
    }
    print_variants(descriptor.id);
    print_specs(&specs);
    Ok(())
}

/// The benchmark's model variants, from its assembled baseline — the values
/// `--checkpoint` accepts, so the reader is not left to grep BENCH.toml for
/// them. Prints nothing outside a checkout or for a benchmark with no baseline
/// entries; the `defined on` families above still describe those.
fn print_variants(benchmark_id: &str) {
    let Ok(root) = super::bench_run::repo_root() else {
        return;
    };
    let Ok(baseline) = atlas_plugin::gate::read_baseline(&root, benchmark_id) else {
        return;
    };
    for (hardware, hw) in &baseline.hardware {
        for (checkpoint, entry) in &hw.models {
            let default = if *checkpoint == hw.default {
                "  (default — runs when no --checkpoint is passed)"
            } else {
                "  (select with --checkpoint under --pull-request-gate)"
            };
            let label = if entry.label.is_empty() {
                String::new()
            } else {
                format!("  — {}", entry.label)
            };
            println!("  variant [{hardware}]  {checkpoint}{label}{default}");
        }
    }
    println!();
}

fn print_specs(specs: &[ParamSpec]) {
    let width = specs.iter().map(|s| s.key.len()).max().unwrap_or(0);
    for s in specs {
        println!(
            "  --param {:width$} = {:<14} {}  [{}]",
            s.key,
            s.default.to_edit_string(),
            s.help,
            s.kind.domain_hint()
        );
    }
}

/// The run list.
pub fn print_history(records: &[RunRecord], format: OutputFormat) -> Result<()> {
    if format == OutputFormat::Json {
        println!("{}", serde_json::to_string_pretty(records)?);
        return Ok(());
    }
    if records.is_empty() {
        println!("no runs recorded yet");
        return Ok(());
    }
    for r in records {
        let verdict = match r.verdict_kind() {
            Some(VerdictKind::Pass) => "PASS",
            Some(VerdictKind::Fail) => "FAIL",
            Some(VerdictKind::Info) => "info",
            None => "—",
        };
        let source = if r.is_legacy() {
            "legacy".to_string()
        } else {
            format!("{:?}", r.source).to_lowercase()
        };
        println!(
            "{}  {:<20} {:<6} {:<6} {}",
            r.run_id,
            r.benchmark_id,
            verdict,
            source,
            r.age_text()
        );
    }
    Ok(())
}

/// One whole record.
pub fn print_record(record: &RunRecord, format: OutputFormat) -> Result<()> {
    if format == OutputFormat::Json {
        println!("{}", serde_json::to_string_pretty(record)?);
        return Ok(());
    }
    println!("{}  {}", record.run_id, record.benchmark_name);
    println!("  when     {} ({})", record.recorded_at, record.age_text());
    println!("  target   {} · {}", record.target_url, record.target_model);
    println!(
        "  source   {:?} · atlas {}",
        record.source, record.atlas_version
    );
    if !record.params.is_empty() {
        println!("  params");
        for (k, v) in &record.params {
            println!("    {k} = {v}");
        }
    }
    println!();
    print_frame(&record.frame);
    Ok(())
}

/// The measurement itself: stats, table, verdict.
pub fn print_frame(frame: &BenchmarkResult) {
    for stat in &frame.summary {
        let unit = &stat.unit;
        println!("  {:<24} {}{}", stat.label, stat.value, unit);
    }
    if let Some(table) = &frame.table {
        println!();
        let headers: Vec<&str> = table.columns.iter().map(|c| c.title.as_str()).collect();
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in &table.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.text.len());
                }
            }
        }
        let line = |cells: Vec<String>| {
            let padded: Vec<String> = cells
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{:width$}", c, width = widths.get(i).copied().unwrap_or(0)))
                .collect();
            println!("  {}", padded.join("  "));
        };
        line(headers.iter().map(|h| h.to_string()).collect());
        for row in &table.rows {
            line(row.iter().map(|c| c.text.clone()).collect());
        }
    }
    if let Some(v) = &frame.verdict {
        println!("\n  {:?}: {}", v.kind, v.reason);
    }
}

/// Progress to stderr, deduplicated.
///
/// At a 250 ms poll a long sweep would otherwise emit thousands of identical
/// lines, so a phase prints only when it changes.
pub struct StdoutReporter {
    pub quiet: bool,
    last_phase: String,
    last_status: String,
    /// The last frame log line printed, so a repeated frame does not repeat it.
    last_frame_log: String,
}

impl StdoutReporter {
    pub fn new(quiet: bool) -> Self {
        Self {
            quiet,
            last_phase: String::new(),
            last_status: String::new(),
            last_frame_log: String::new(),
        }
    }
}

impl RunReporter for StdoutReporter {
    fn started(&mut self, request: &RunRequest) {
        eprintln!(
            "{} → {} · {}",
            request.descriptor.name, request.target.base_url, request.target.model
        );
    }

    fn event(&mut self, event: &PluginEvent) {
        match event {
            // Warnings and errors always surface; info is progress noise.
            PluginEvent::Log(line)
                if !self.quiet
                    || matches!(
                        line.level,
                        atlas_plugin::LogLevel::Warn | atlas_plugin::LogLevel::Error
                    ) =>
            {
                eprintln!("  {:?}: {}", line.level, line.text);
            }
            PluginEvent::Status(s) if !self.quiet && *s != self.last_status => {
                self.last_status = s.clone();
                eprintln!("  {s}");
            }
            _ => {}
        }
    }

    fn frame(&mut self, frame: &BenchmarkResult) {
        // Frame log lines, BEFORE the phase early-return.
        //
        // ★ These used to be dropped entirely: `PluginEvent::Log` was printed
        // but a `BenchmarkResult`'s own `log` was not, so every diagnostic a
        // benchmark attaches to a frame was visible in the TUI and invisible
        // on the CLI -- the mode the PR gate runs in. That is how a BFCL run
        // drew n=972 instead of its pinned 1004 while the guard written to
        // catch precisely that printed its warning into nowhere.
        //
        // Emitted before the phase check because a warning must not depend on
        // whether its frame happened to also change phase.
        for line in &frame.log {
            if !self.quiet
                || matches!(
                    line.level,
                    atlas_plugin::LogLevel::Warn | atlas_plugin::LogLevel::Error
                )
            {
                let text = format!("  {:?}: {}", line.level, line.text);
                if text != self.last_frame_log {
                    self.last_frame_log = text.clone();
                    eprintln!("{text}");
                }
            }
        }
        if self.quiet || frame.phase == self.last_phase {
            return;
        }
        self.last_phase = frame.phase.clone();
        let progress = match frame.progress {
            Some((done, total)) => format!(" [{done}/{total}]"),
            None => String::new(),
        };
        eprintln!(
            "  [{:>6.1}s] {}{progress}",
            frame.elapsed.as_secs_f64(),
            frame.phase
        );
    }
}
