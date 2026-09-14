// SPDX-License-Identifier: AGPL-3.0-only

//! The dispatch-side paged-decode route line, graded as TEXT.
//!
//! H100 round 15 anomaly 3: "there is no dispatch-side route line naming the
//! chosen split count at all". The kernel-selection table printed
//! `paged_decode_fp8_splitk_hopper … used`, which proves the entry RESOLVED —
//! not that eleven splits reached the launch. The `(24,11,1)` grid was only
//! visible in an nsys capture, which is not a reasonable standing cost for
//! confirming a shipped lever from a serve log (#928).
//!
//! Pinned WHOLE, for the reason `tests/build_summary.rs` pins its line whole: a
//! format assembled from separately asserted pieces can be reordered without a
//! test noticing, and the value of this line is that it is greppable out of a
//! campaign log months later.

use super::splitk_dispatch::{
    ROUTE_NONSPLIT_BF16, ROUTE_NONSPLIT_FP8, ROUTE_SPLITK_BF16, ROUTE_SPLITK_FP8, route_line,
};
use atlas_kernels::attn_splitk::SplitkPolicy;

/// THE line, as round 15 asked for it, for the FP8 arm on Hopper under `auto`.
///
/// `sm_count` is read from `atlas_kernels::TARGET_SM_COUNT` — the compiled
/// target's, the same number the policy divided by — so this test states the
/// expectation against that constant rather than hardcoding 132 and passing on
/// a gb10 build for the wrong reason.
#[test]
fn the_fp8_route_line_names_the_kernel_the_split_count_and_the_policy() {
    let sm = atlas_kernels::TARGET_SM_COUNT;
    assert_eq!(
        route_line(ROUTE_SPLITK_FP8, 11, SplitkPolicy::Auto),
        format!(
            "paged decode attention: paged_decode_attn_splitk_fp8_hopper num_splits=11 \
             sm_count={sm} policy=auto (ATLAS_ATTN_DECODE_SPLITK)"
        ),
    );
}

/// The BF16 twin says the same line about its own kernel. Qwen3.8-27B runs FP8
/// KV on 44 layers and BF16 on the four `--kv-high-precision-layers auto` ones,
/// and both took the split-K path in round 15 — nsys priced them separately at
/// 32.7 and 35.0 µs/launch — so a log that named only one arm would be as
/// incomplete as no line at all.
#[test]
fn the_bf16_twin_reports_its_own_arm() {
    let line = route_line(ROUTE_SPLITK_BF16, 11, SplitkPolicy::Auto);
    assert!(
        line.contains("paged_decode_attn_splitk_bf16_hopper"),
        "{line}"
    );
    assert!(line.contains("num_splits=11"), "{line}");
    assert_ne!(line, route_line(ROUTE_SPLITK_FP8, 11, SplitkPolicy::Auto));
}

/// `ATLAS_ATTN_DECODE_SPLITK=0` is the pre-#928 control, and the line must say
/// so honestly: the NON-split kernel, at one split, under a policy that renders
/// as `1`. Round 15's S0 cell booted `attn_decode_splitk=1 (env)` from a raw
/// `0`, which is correct (`0`/`off` resolves to one split) and confusing
/// without this second line naming the kernel that then ran.
#[test]
fn the_zero_control_reports_the_non_split_kernel_at_one_split() {
    assert_eq!(
        route_line(ROUTE_NONSPLIT_FP8, 1, SplitkPolicy::Pinned(1)),
        format!(
            "paged decode attention: paged_decode_attn_fp8 num_splits=1 sm_count={} \
             policy=1 (ATLAS_ATTN_DECODE_SPLITK)",
            atlas_kernels::TARGET_SM_COUNT,
        ),
    );
    // And the policy field round-trips the boot line's own spelling, so the two
    // lines in one log cannot disagree about the lever.
    for (policy, label) in [
        (SplitkPolicy::Legacy, "legacy"),
        (SplitkPolicy::Auto, "auto"),
        (SplitkPolicy::Pinned(6), "6"),
    ] {
        assert!(
            route_line(ROUTE_NONSPLIT_BF16, 1, policy).contains(&format!("policy={label} ")),
            "policy {policy:?} must render as {label}",
        );
    }
}

/// The environment variable that moves it is NAMED, the way every other route
/// line in this crate names its lever — an operator who reads the line has the
/// knob without grepping the source.
#[test]
fn the_line_names_its_lever_and_leads_with_the_kernel() {
    let line = route_line(ROUTE_SPLITK_FP8, 11, SplitkPolicy::Auto);
    assert!(line.ends_with("(ATLAS_ATTN_DECODE_SPLITK)"), "{line}");
    assert!(
        line.starts_with("paged decode attention: paged_decode_attn_splitk_fp8_hopper "),
        "{line}"
    );
}
