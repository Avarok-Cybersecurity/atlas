// SPDX-License-Identifier: AGPL-3.0-only

//! Ops-tab slash commands. Every command maps to an existing SAFE mechanism —
//! read-only globals or already-atomic setters. Nothing here races the
//! scheduler thread's locals; commands that would (kill-seq, flush-KV, pause)
//! are deliberately absent.

use super::app::App;

pub const COMMANDS: &[(&str, &str)] = &[
    ("/help", "list commands"),
    ("/status", "scheduler snapshot + request counters"),
    (
        "/metrics [substr]",
        "in-process prometheus dump, optionally filtered",
    ),
    (
        "/kernels [substr]",
        "kernel resolution rows, optionally filtered",
    ),
    ("/gpu", "GPU + host memory"),
    ("/cache", "prefix-cache hit statistics"),
    ("/watchdog on|off", "toggle the loop watchdog"),
    ("/detach", "leave the TUI, keep serving with plain logs"),
    ("/quit", "clean shutdown (drain in-flight, then exit)"),
];

/// Ghost-text completion for the input line.
pub fn complete(input: &str) -> Option<&'static str> {
    if !input.starts_with('/') || input.contains(' ') {
        return None;
    }
    COMMANDS
        .iter()
        .map(|(c, _)| c.split(' ').next().unwrap_or(c))
        .find(|c| c.starts_with(input) && *c != input)
}

/// Execute one line. Output is appended to the ops pane; chat-style bare text
/// is routed to the Chat tab's engine instead.
pub fn execute(line: &str, app: &mut App) {
    let line = line.trim();
    app.ops.output.push(format!("❯ {line}"));
    if !line.starts_with('/') {
        app.ops
            .output
            .push("(bare text goes to the Chat tab — press 6 twice)".into());
        return;
    }
    let mut parts = line.splitn(2, ' ');
    let cmd = parts.next().unwrap_or("");
    let arg = parts.next().unwrap_or("").trim();
    match cmd {
        "/help" => {
            for (c, d) in COMMANDS {
                app.ops.output.push(format!("  {c:<22} {d}"));
            }
        }
        "/status" => cmd_status(app),
        "/metrics" => cmd_metrics(app, arg),
        "/kernels" => cmd_kernels(app, arg),
        "/gpu" => cmd_gpu(app),
        "/cache" => cmd_cache(app),
        "/watchdog" => {
            let on = match arg {
                "on" => true,
                "off" => false,
                _ => {
                    app.ops.output.push("usage: /watchdog on|off".into());
                    return;
                }
            };
            match app.run.as_ref().map(|r| &r.levers) {
                Some(l) => {
                    l.set_loop_watchdog(on);
                    app.ops
                        .output
                        .push(format!("loop watchdog: {}", if on { "ON" } else { "OFF" }));
                }
                None => app
                    .ops
                    .output
                    .push("loop watchdog: no run yet — the scheduler has not started".into()),
            }
        }
        "/detach" => app.detach = true,
        "/quit" => {
            super::shutdown::request("/quit");
            app.should_quit = true;
        }
        other => app
            .ops
            .output
            .push(format!("unknown command {other} — /help")),
    }
}

fn cmd_status(app: &mut App) {
    match app.run.as_ref().and_then(|r| r.snapshot.read()) {
        Some(s) => {
            app.ops.output.push(format!(
                "  seqs: active {} · prefilling {} · swapped {} · pending {}",
                s.active_seqs, s.prefilling_seqs, s.swapped_seqs, s.pending_len
            ));
            app.ops.output.push(format!(
                "  kv blocks {}/{} free · ssm slots {}/{} used",
                s.kv_blocks_free, s.kv_blocks_total, s.ssm_slots_used, s.ssm_slots_total
            ));
            app.ops.output.push(format!(
                "  mtp {} · delivered {:.1} tok/s · {} steps",
                crate::tui::format::mtp_mode_label(s.mtp_mode),
                s.delivered_tps,
                s.steps_total
            ));
        }
        None => app
            .ops
            .output
            .push("  scheduler snapshot not yet published".into()),
    }
    app.ops.output.push(format!(
        "  requests: {} total · {} active · {} gen tok · {} prompt tok",
        crate::metrics::REQUESTS_TOTAL.get(),
        crate::metrics::REQUESTS_ACTIVE.get(),
        crate::metrics::GENERATION_TOKENS_TOTAL.get(),
        crate::metrics::PROMPT_TOKENS_TOTAL.get(),
    ));
}

fn cmd_metrics(app: &mut App, filter: &str) {
    let mut n = 0;
    for mf in prometheus::gather() {
        if !filter.is_empty() && !mf.name().contains(filter) {
            continue;
        }
        let kind = mf.get_field_type();
        for m in mf.get_metric() {
            let v = match kind {
                prometheus::proto::MetricType::COUNTER => m.get_counter().get_value(),
                prometheus::proto::MetricType::GAUGE => m.get_gauge().get_value(),
                prometheus::proto::MetricType::HISTOGRAM => {
                    m.get_histogram().get_sample_count() as f64
                }
                _ => continue,
            };
            let labels: Vec<String> = m
                .get_label()
                .iter()
                .map(|l| format!("{}={}", l.name(), l.value()))
                .collect();
            let suffix = if labels.is_empty() {
                String::new()
            } else {
                format!("{{{}}}", labels.join(","))
            };
            app.ops
                .output
                .push(format!("  {}{suffix} = {v}", mf.name()));
            n += 1;
            if n >= 40 {
                app.ops
                    .output
                    .push("  … (narrow with /metrics <substr>)".into());
                return;
            }
        }
    }
    if n == 0 {
        app.ops.output.push("  (no metrics matched)".into());
    }
}

fn cmd_kernels(app: &mut App, filter: &str) {
    let rows = spark_runtime::kernel_audit::audit_rows();
    let mut n = 0;
    for r in rows {
        let (m, f, ok) = (&r.module, &r.func, r.loaded);
        if !filter.is_empty() && !m.contains(filter) && !f.contains(filter) {
            continue;
        }
        // A failed lookup without its dispatch site is a name, not a work item.
        let site = if ok {
            String::new()
        } else {
            format!("  at {}:{}", r.site.file(), r.site.line())
        };
        app.ops
            .output
            .push(format!("  {} {m}::{f}{site}", if ok { "✓" } else { "✗" }));
        n += 1;
        if n >= 40 {
            app.ops
                .output
                .push("  … (narrow with /kernels <substr>)".into());
            return;
        }
    }
    if n == 0 {
        app.ops.output.push("  (no kernel lookups matched)".into());
    }
}

fn cmd_gpu(app: &mut App) {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let free = super::data::gpu_free_bytes().map(|b| b as f64 / GIB);
    let baseline = spark_runtime::gpu::baseline_free_bytes()
        .map(|b| b as f64 / GIB)
        .unwrap_or(0.0);
    match free {
        Some(f) => {
            app.ops.output.push(format!(
                "  gpu free {f:.1} GB · baseline {baseline:.1} GB · atlas ≈ {:.1} GB",
                (baseline - f).max(0.0)
            ));
        }
        None => app.ops.output.push("  gpu memory query unavailable".into()),
    }
}

fn cmd_cache(app: &mut App) {
    let hits = spark_runtime::prefix_cache::cache_hit_count();
    let misses = spark_runtime::prefix_cache::cache_miss_count();
    let toks = spark_runtime::prefix_cache::cache_hit_tokens_total();
    let rate = if hits + misses > 0 {
        format!("{:.1}%", hits as f64 * 100.0 / (hits + misses) as f64)
    } else {
        "—".into()
    };
    app.ops.output.push(format!(
        "  prefix cache: {hits} hits / {misses} misses ({rate}) · {toks} tokens served warm"
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_finds_unique_prefixes() {
        assert_eq!(complete("/ker"), Some("/kernels"));
        assert_eq!(complete("/kernels"), None); // already complete
        assert_eq!(complete("hello"), None); // not a slash command
    }
}

// The dispatch cases live in their own mount; the completion case above stays
// beside the function it describes.
#[cfg(test)]
#[path = "commands_tests.rs"]
mod line_tests;
