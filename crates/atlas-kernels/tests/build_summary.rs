// SPDX-License-Identifier: AGPL-3.0-only

//! The one line a build prints per kernel target, graded as TEXT.
//!
//! H100 round 11 asked for the resolved kernel count to be visible in a normal
//! build; round 13 (2026-09-11) found it still absent from `build-r13.log` and
//! recovered `196` from `target/release/build/atlas-kernels-*/output`, then
//! confirmed it a second way by running the boot preflight
//! (`kernel_check.modules_embedded: 196`) against the finished binary. This
//! file is why that is now one `grep` of the build output.
//!
//! An integration test rather than a `#[cfg(test)]` module inside the build
//! script, because cargo never runs a build script's own unit tests — the same
//! reason `tests/target_defaults.rs`, `tests/kernel_build_flags.rs` and
//! `tests/kernel_target_arch.rs` exist. It compiles `build_summary.rs`
//! directly, so there is no second formatter to drift.

#[path = "../build_summary.rs"]
mod build_summary;

use build_summary::summary;

/// THE LINE, as an operator reads it out of a build log. Pinned whole rather
/// than field by field: the value of this line is that it can be grepped from a
/// campaign's build output months later, and a format assembled from separately
/// asserted pieces can still be reordered without a test noticing.
#[test]
fn the_summary_is_the_round_thirteen_h100_line() {
    assert_eq!(
        summary(196, "hopper", "qwen3.8-27b", "nvfp4", 14),
        "atlas-kernels: 196 kernels (hopper, qwen3.8-27b, nvfp4), 14 declared overrides"
    );
}

/// ZERO overrides prints, rather than vanishing. The predecessor line omitted
/// the clause entirely at zero, so "this target declares none" and "this build
/// did not report" were the same bytes in a log — and the whole point of the
/// line is that its absence means something.
#[test]
fn zero_overrides_still_prints_a_count() {
    let line = summary(158, "gb10", "qwen3.6-27b", "nvfp4", 0);
    assert_eq!(
        line,
        "atlas-kernels: 158 kernels (gb10, qwen3.6-27b, nvfp4), 0 declared overrides"
    );
    assert!(line.contains("0 declared overrides"));
}

/// One line per TARGET, and the target is named by `(hw, model, quant)` — not
/// by the loop index the predecessor printed. A multi-target build emits one
/// line each and they must be distinguishable from each other by content: an
/// index is only meaningful next to the resolution order it came from, which a
/// build log does not carry.
#[test]
fn each_target_is_identified_by_its_own_triple() {
    let lines: Vec<String> = [
        (196usize, "hopper", "qwen3.8-27b", "nvfp4", 14usize),
        (196, "b200", "qwen3.8-27b", "nvfp4", 14),
        (158, "gb10", "qwen3.6-27b", "nvfp4", 2),
    ]
    .into_iter()
    .map(|(n, hw, model, quant, ov)| summary(n, hw, model, quant, ov))
    .collect();
    let mut unique = lines.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(
        unique.len(),
        lines.len(),
        "two targets must not print the same line:\n{}",
        lines.join("\n")
    );
    for line in &lines {
        assert!(line.starts_with("atlas-kernels: "), "{line}");
    }
}

/// The count is the FIRST number in the line, so `grep -o` and an operator's
/// eye find the same thing. Guards against a future edit that leads with the
/// target triple and leaves the count trailing.
#[test]
fn the_kernel_count_leads_the_line() {
    let line = summary(196, "hopper", "qwen3.8-27b", "nvfp4", 14);
    let first_number: String = line
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    assert_eq!(first_number, "196");
    assert!(line.contains("196 kernels ("), "{line}");
}
