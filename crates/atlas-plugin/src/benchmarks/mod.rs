// SPDX-License-Identifier: AGPL-3.0-only

//! The benchmark suite.
//!
//! Everything except BFCL is native: it drives the served endpoint over HTTP
//! and needs nothing installed on the box. BFCL keeps Python for dataset
//! materialization and AST scoring, provisioned into `~/.atlas/artifacts`
//! during `load()`.

use std::sync::atomic::{AtomicU64, Ordering};

pub mod agentic;
pub mod baseline;
pub mod bfcl;
pub mod concurrency;
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

/// A salt no other request in this process will use.
///
/// The cold-TTFT gate depends on this: two requests sharing a prefix means the
/// second one hits the cache and the "cold" number is warm.
pub fn unique_salt(prefix: &str) -> String {
    // STATIC, DELIBERATELY — process lifecycle. Uniqueness must hold across
    // EVERY request this process issues, which is the whole guarantee: two
    // requests sharing a prefix means the second hits the cache and the
    // "cold" number is warm. A per-run counter would let two benchmark runs
    // in one process collide and silently warm each other's cold leg.
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{}-{n}", std::process::id())
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
    fn salts_never_repeat() {
        let a = unique_salt("cold");
        let b = unique_salt("cold");
        assert_ne!(a, b);
    }
}
