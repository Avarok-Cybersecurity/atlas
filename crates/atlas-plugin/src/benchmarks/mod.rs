// SPDX-License-Identifier: AGPL-3.0-only

//! The benchmark suite.
//!
//! Everything except BFCL is native: it drives the served endpoint over HTTP
//! and needs nothing installed on the box. BFCL keeps Python for dataset
//! materialization and AST scoring, provisioned into `~/.atlas/artifacts`
//! during `load()`.

pub mod agentic;
pub mod baseline;
pub mod bfcl;
pub mod concurrency;
pub mod serve_matrix;
pub mod stats;
pub mod ttft;

/// Collapse a message onto one line and bound its length.
///
/// Log lines land in a fixed-height pane; a model reply or a `pip` traceback
/// pasted in raw scrolls everything else off the screen.
pub fn one_line(text: impl AsRef<str>) -> String {
    const MAX: usize = 300;
    let mut s: String = text
        .as_ref()
        .chars()
        .map(|c| {
            if c == '\n' || c == '\r' || c == '\t' {
                ' '
            } else {
                c
            }
        })
        .collect();
    let squashed = s.split_whitespace().collect::<Vec<_>>().join(" ");
    s = squashed;
    if s.chars().count() > MAX {
        s = s.chars().take(MAX - 1).collect::<String>() + "…";
    }
    s
}

/// A prompt prefix no other request in this process will use.
///
/// `run_id` comes from the run's [`crate::PluginHandle`]; the caller supplies a
/// `prefix` that is unique within the run. Together they are unique across the
/// process without a process-global counter — which matters because the
/// cold-TTFT gate is exactly the measurement a shared prefix would corrupt.
pub fn unique_prefix_tag(prefix: &str, run_id: u64) -> String {
    format!("{prefix}-{}-{run_id}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_line_squashes_and_truncates() {
        assert_eq!(one_line("a\n b\t\tc "), "a b c");
        let long = one_line("x".repeat(1000));
        assert_eq!(long.chars().count(), 300);
        assert!(long.ends_with('…'));
    }

    #[test]
    fn prefix_tags_never_repeat() {
        let a = unique_prefix_tag("cold", 1);
        let b = unique_prefix_tag("cold", 2);
        assert_ne!(a, b);
    }
}
