// SPDX-License-Identifier: AGPL-3.0-only

//! `Hardware::gate_key` — the string that decides WHICH baseline a committed
//! record is scored against.
//!
//! Split into its own file to keep `hardware.rs` under the 500-line cap.

use super::Hardware;

fn gpu(name: &str) -> Hardware {
    Hardware {
        gpu: name.to_string(),
        ..Hardware::default()
    }
}

#[test]
fn a_gb10_normalises_to_the_key_the_baselines_use() {
    // Every committed BASELINE.json keys its thresholds under "gb10"; if this
    // drifts, `resolve` reports "no baseline for hardware ..." and every gate
    // on the box stops, so it is worth pinning literally.
    assert_eq!(gpu("NVIDIA GB10").gate_key(), "gb10");
    // Vendor prefix and punctuation are noise, so the same part reported three
    // ways must land on one key rather than three.
    assert_eq!(gpu("GB10").gate_key(), "gb10");
    assert_eq!(gpu("nvidia gb10").gate_key(), "gb10");
}

#[test]
fn a_box_that_reports_nothing_is_named_unknown_not_empty() {
    // `fetch_hardware` degrades to an unknown Hardware WITHOUT surfacing an
    // error, so this path is reached in practice. It must produce a key that
    // no baseline defines — the lookup then fails loudly instead of matching
    // whichever entry happened to sort first.
    assert_eq!(Hardware::default().gate_key(), "unknown");
    assert_eq!(gpu("").gate_key(), "unknown");
    // A name made entirely of punctuation filters down to nothing; returning
    // "" would key every such box together.
    assert_eq!(gpu("- / .").gate_key(), "unknown");
}

#[test]
fn parts_that_differ_must_not_collapse_onto_one_key() {
    // ★ The failure that matters: two DIFFERENT accelerators sharing a key
    // means one box is silently scored against the other's thresholds, which
    // is not a lenient comparison but a meaningless one. Capacity and
    // generation both have to survive normalisation.
    let keys = [
        gpu("NVIDIA GB10").gate_key(),
        gpu("NVIDIA GB200").gate_key(),
        gpu("NVIDIA A100-SXM4-40GB").gate_key(),
        gpu("NVIDIA A100-SXM4-80GB").gate_key(),
        gpu("AMD Radeon 8060S (gfx1151)").gate_key(),
    ];
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b, "distinct parts must not share a gate key");
        }
        assert_ne!(a, "unknown", "a named part must not read as unknown");
    }
}
