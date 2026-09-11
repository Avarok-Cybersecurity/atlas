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
//! Round 15 then found the line's SECOND field mislabelled — it read
//! `14 declared overrides` while the target's `[kernels] overrides` list held
//! a different number of entries — and that is what the two-count tests below
//! pin.
//!
//! An integration test rather than a `#[cfg(test)]` module inside the build
//! script, because cargo never runs a build script's own unit tests — the same
//! reason `tests/target_defaults.rs`, `tests/kernel_build_flags.rs` and
//! `tests/kernel_target_arch.rs` exist. It compiles `build_summary.rs`
//! directly, so there is no second formatter to drift.

#[path = "../build_summary.rs"]
mod build_summary;

use build_summary::{count_declared_overrides, declared_overrides, summary};

/// The repository's `kernels/` directory, from this crate's manifest.
fn kernels_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
}

/// THE LINE, as an operator reads it out of a build log. Pinned whole rather
/// than field by field: the value of this line is that it can be grepped from a
/// campaign's build output months later, and a format assembled from separately
/// asserted pieces can still be reordered without a test noticing.
#[test]
fn the_summary_line_is_pinned_whole() {
    assert_eq!(
        summary(199, "hopper", "qwen3.8-27b", "nvfp4", 14, 15),
        "atlas-kernels: 199 kernels (hopper, qwen3.8-27b, nvfp4), \
         14 model-dir kernels, 15 declared overrides"
    );
}

/// THE round-15 defect, stated as a property: the two counts are INDEPENDENT
/// and the line must be able to show them disagreeing.
///
/// Round 14 printed `14 declared overrides` with eleven entries in the list;
/// round 15 printed `14` with fifteen. The field never tracked the list — a
/// hardware-owned kernel lives in `common/`, not in the per-model directory —
/// so an operator checking "did my override land" was reading a number that
/// could not answer. Both now print, and a formatter that collapsed them into
/// one field would fail here.
#[test]
fn the_model_dir_count_and_the_overrides_list_are_separate_fields() {
    let r14 = summary(196, "hopper", "qwen3.8-27b", "nvfp4", 14, 11);
    let r15 = summary(199, "hopper", "qwen3.8-27b", "nvfp4", 14, 15);
    assert!(
        r14.contains("14 model-dir kernels, 11 declared overrides"),
        "{r14}"
    );
    assert!(
        r15.contains("14 model-dir kernels, 15 declared overrides"),
        "{r15}"
    );
    assert_ne!(
        r14, r15,
        "the overrides list moved; the line must move with it"
    );
}

/// ZERO prints, rather than vanishing. The predecessor line omitted the
/// overrides clause entirely at zero, so "this target declares none" and "this
/// build did not report" were the same bytes in a log — and the whole point of
/// the line is that its absence means something.
///
/// It is also the state of every target in THIS tree: no hardware set declares
/// an override yet, so a line that printed nothing at zero would carry no
/// information at all today.
#[test]
fn zero_counts_still_print() {
    let line = summary(158, "gb10", "qwen3.6-27b", "nvfp4", 0, 0);
    assert_eq!(
        line,
        "atlas-kernels: 158 kernels (gb10, qwen3.6-27b, nvfp4), \
         0 model-dir kernels, 0 declared overrides"
    );
}

/// One line per TARGET, and the target is named by `(hw, model, quant)` — not
/// by the loop index the predecessor printed. A multi-target build emits one
/// line each and they must be distinguishable from each other by content: an
/// index is only meaningful next to the resolution order it came from, which a
/// build log does not carry.
#[test]
fn each_target_is_identified_by_its_own_triple() {
    let lines: Vec<String> = [
        (199usize, "hopper", "qwen3.8-27b", "nvfp4", 14usize, 15usize),
        (196, "b200", "qwen3.8-27b", "nvfp4", 14, 0),
        (158, "gb10", "qwen3.6-27b", "nvfp4", 2, 0),
    ]
    .into_iter()
    .map(|(n, hw, model, quant, md, ov)| summary(n, hw, model, quant, md, ov))
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
    let line = summary(199, "hopper", "qwen3.8-27b", "nvfp4", 14, 15);
    let first_number: String = line
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    assert_eq!(first_number, "199");
    assert!(line.contains("199 kernels ("), "{line}");
}

/// The counter reads the REAL `kernels/<hw>/HARDWARE.toml`, not a fixture, and
/// agrees entry for entry with the parser the tests and
/// `scripts/check_kernel_shadows.py` read.
///
/// Every declaration in the tree is checked rather than one target's, and NO
/// non-zero count is asserted: every hardware set inherits everything today,
/// so the honest statement is "whatever each file declares, the count is that
/// list's length and every entry is a kernel source". A PR that adds a tuned
/// kernel adds the declaration and this test grades it without being touched.
#[test]
fn the_overrides_count_is_the_real_declaration_entry_for_entry() {
    let root = kernels_root();
    for hw in ["gb10", "hopper", "b200"] {
        let n = count_declared_overrides(&root, hw);
        let text = std::fs::read_to_string(root.join(hw).join("HARDWARE.toml"))
            .unwrap_or_else(|e| panic!("kernels/{hw}/HARDWARE.toml: {e}"));
        let value: toml::Value = text.parse().expect("valid TOML");
        let entries = declared_overrides(&value);
        assert_eq!(entries.len(), n, "kernels/{hw}");
        for e in &entries {
            assert!(
                e.ends_with(".cu") || e.ends_with(".cuh"),
                "kernels/{hw}: override {e} is not a kernel source"
            );
        }
    }
}

/// A target with no `[kernels]` table counts zero and does not panic — which
/// is every target in this tree — and neither does a missing tree, which is
/// the `ATLAS_SKIP_BUILD` path.
#[test]
fn an_absent_declaration_counts_zero() {
    assert_eq!(count_declared_overrides(&kernels_root(), "gb10"), 0);
    assert_eq!(
        count_declared_overrides(std::path::Path::new("/nonexistent"), "hopper"),
        0
    );
    let empty: toml::Value = "[hardware]\narch = \"sm_90a\"\n".parse().unwrap();
    assert!(declared_overrides(&empty).is_empty());
    let bad: toml::Value = "[kernels]\noverrides = [1, \"ok.cu\"]\n".parse().unwrap();
    assert_eq!(declared_overrides(&bad), vec!["ok.cu".to_string()]);
}
