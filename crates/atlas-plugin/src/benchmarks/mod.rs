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
pub mod contamination;
pub mod decode_floor;
pub mod expert_categories;
pub mod expert_fidelity;
pub mod media_integrity;
pub mod mlperf_agentic;
pub mod quick_speed;
pub mod serve_matrix;
pub mod ssm_poison;
pub mod stats;
pub mod transcript;
pub mod ttft;
pub mod video;
pub mod vision;

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

/// Content-derived identity for a named set of committed benchmark assets.
pub(crate) fn content_stamp<'a>(
    prefix: &str,
    assets: impl IntoIterator<Item = (&'a str, &'a [u8])>,
) -> String {
    let mut acc: u64 = 1469598103934665603; // FNV-1a offset basis
    for (name, bytes) in assets {
        for byte in name.as_bytes().iter().chain(bytes) {
            acc ^= *byte as u64;
            acc = acc.wrapping_mul(1099511628211);
        }
    }
    format!("{prefix}-{acc:016x}")
}

/// First occurrence of a lowercased term outside an identifier-like word.
pub(crate) fn first_standalone_term(haystack: &str, term: &str) -> Option<usize> {
    let first = term.chars().next()?;
    let last = term.chars().next_back()?;
    let continues = |edge: char, adjacent: char| {
        adjacent == '_'
            || if edge.is_numeric() {
                adjacent.is_numeric()
            } else {
                adjacent.is_alphanumeric()
            }
    };
    haystack.match_indices(term).find_map(|(at, _)| {
        let before = haystack[..at].chars().next_back();
        let after = haystack[at + term.len()..].chars().next();
        (!before.is_some_and(|adjacent| continues(first, adjacent))
            && !after.is_some_and(|adjacent| continues(last, adjacent)))
        .then_some(at)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_line_squashes_and_truncates() {
        assert_eq!(one_line(" a\n b\r\t\tc \u{a0} d "), "a b c d");
        assert_eq!(one_line("é".repeat(299)), "é".repeat(299));
        assert_eq!(one_line("é".repeat(300)), "é".repeat(300));
        assert_eq!(
            one_line("é".repeat(301)),
            "é".repeat(299) + "…",
            "the ellipsis is the 300th character"
        );
    }

    #[test]
    fn prefix_tags_never_repeat() {
        let a = unique_prefix_tag("cold", 1);
        let b = unique_prefix_tag("cold", 2);
        let c = unique_prefix_tag("warm", 1);
        assert_eq!(a, format!("cold-{}-1", std::process::id()));
        assert_ne!(a, b);
        assert_ne!(a, c);
    }
}
