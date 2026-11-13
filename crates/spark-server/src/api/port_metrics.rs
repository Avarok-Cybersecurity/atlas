// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the port sprint's (`port/consumer-opt`) `/metrics` series,
//! appended to the Prometheus text by `misc_handlers::metrics_handler`.
//!
//! Each lane keeps its series in its own atomics and its own
//! `render_prometheus(out: &mut String)`, which formats them with the helpers
//! below (passed by function argument to crates below spark-server, which
//! format their own text with the same `# HELP` / `# TYPE` lines). The lane
//! adds one call line at the marker in [`render`]. No prometheus statics are
//! registered and `metrics.rs` is untouched. A row-off lane's series render as
//! zeros. Lines are inserted immediately BEFORE the marker below by the
//! coordinator's fragment apply pass.

use std::fmt::{Display, Write};

/// Append every port series to the `/metrics` text. Empty until a lane adds a
/// line.
///
/// `clippy::ptr_arg` is allowed because the body is empty until a lane adds a
/// line: an unused `&mut String` reads as "take `&mut str`", and clippy 0.1.93
/// panics while rendering that suggestion here.
#[allow(unused_variables, clippy::ptr_arg)]
pub fn render(out: &mut String) {
    spark_model::port::w4_stats::render_prometheus(out);
    spark_model::port::embed_host::render_prometheus(out);
    spark_runtime::kv_cache::block_io::swap::render_prometheus(out);
    spark_runtime::pinned_pool::render_prometheus(out);
    spark_runtime::kv_host_tier::render_prometheus(out);
    crate::scheduler::restore_queue::render_prometheus(out);
    spark_runtime::kv_host_tier::disk::render_prometheus(out);
    spark_runtime::radix_tree::render_prometheus(out);
    spark_model::layers::qwen3_attention::sparse_seam::render_prometheus(out);
    // @port:render
}

/// One counter: `# HELP`, `# TYPE <name> counter`, `<name> <v>`.
pub fn counter(out: &mut String, name: &str, help: &str, v: impl Display) {
    let _ = writeln!(
        out,
        "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}"
    );
}

/// One gauge: `# HELP`, `# TYPE <name> gauge`, `<name> <v>`.
pub fn gauge(out: &mut String, name: &str, help: &str, v: impl Display) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}");
}

/// A labeled gauge family: one `# HELP` / `# TYPE <name> gauge` header, then
/// `<name>{<labels>} <v>` per entry. `labels` is the text inside the braces,
/// for example `pos="1"`; an empty string writes the bare name.
pub fn labeled<V: Display>(out: &mut String, name: &str, help: &str, series: &[(&str, V)]) {
    labeled_kind(out, name, help, "gauge", series);
}

/// A labeled counter family, formatted like [`labeled`] with
/// `# TYPE <name> counter`.
pub fn labeled_counter<V: Display>(out: &mut String, name: &str, help: &str, series: &[(&str, V)]) {
    labeled_kind(out, name, help, "counter", series);
}

/// A summary written as its `_sum` and `_count` series only (no quantiles):
/// `# TYPE <name> summary`, `<name>_sum <sum>`, `<name>_count <count>`.
pub fn hist_sum_count(out: &mut String, name: &str, help: &str, sum: impl Display, count: u64) {
    let _ = writeln!(
        out,
        "# HELP {name} {help}\n# TYPE {name} summary\n{name}_sum {sum}\n{name}_count {count}"
    );
}

fn labeled_kind<V: Display>(
    out: &mut String,
    name: &str,
    help: &str,
    kind: &str,
    series: &[(&str, V)],
) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
    for (labels, v) in series {
        if labels.is_empty() {
            let _ = writeln!(out, "{name} {v}");
        } else {
            let _ = writeln!(out, "{name}{{{labels}}} {v}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_write_prometheus_text() {
        let mut s = String::new();
        counter(&mut s, "a_total", "A.", 3u64);
        gauge(&mut s, "b", "B.", 0.5f64);
        labeled(&mut s, "c", "C.", &[("pos=\"1\"", 2u64), ("", 4u64)]);
        hist_sum_count(&mut s, "d", "D.", 1.5f64, 2);
        assert_eq!(
            s,
            "# HELP a_total A.\n# TYPE a_total counter\na_total 3\n\
             # HELP b B.\n# TYPE b gauge\nb 0.5\n\
             # HELP c C.\n# TYPE c gauge\nc{pos=\"1\"} 2\nc 4\n\
             # HELP d D.\n# TYPE d summary\nd_sum 1.5\nd_count 2\n"
        );
    }

    #[test]
    fn render_only_appends() {
        let mut s = String::from("x 1\n");
        render(&mut s);
        assert!(s.starts_with("x 1\n"));
    }
}
