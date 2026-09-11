// SPDX-License-Identifier: AGPL-3.0-only

//! The `attn_decode_splitk` row of the target table (#928).
//!
//! Split from `target_defaults_tests.rs` when the merge that brought #928's
//! attention row alongside #917's `w8a8_prefill_max_m_*` pair and #928's
//! `ssm_ba_gates_hopper` row carried the parent past the 500-line cap. Neither
//! branch crossed it alone, which is the case the cap is least able to warn
//! about in advance and the reason the seam is drawn here rather than left for
//! the next row to force.
//!
//! The seam is by lever: this file grades the split-K policy end to end —
//! declaration, environment, and the spelling the serve line reports it in —
//! while the parent keeps the whole-table tests — the serve line that
//! enumerates every lever in particular — next to the fixtures they name.
//!
//! A child of `tests`, not a sibling, so `HOPPER`, `GB10`, `with`, `empty` and
//! `format_levers` come from the parent rather than being copied. A second copy
//! of those fixtures is how two files come to disagree about what Hopper
//! declares.

use super::*;

/// The split-K policy row (#928): declaration first, environment second, and
/// the resolved value printed in the spelling that reproduces it.
#[test]
fn the_split_k_policy_resolves_and_reports_like_every_other_lever() {
    use atlas_kernels::attn_splitk::SplitkPolicy;
    assert_eq!(
        empty(&HOPPER).attn_decode_splitk.value,
        SplitkPolicy::Auto,
        "an H100 serve with an empty environment must reach the split count \
         that fills 132 SMs — the whole content of #928"
    );
    assert_eq!(
        empty(&GB10).attn_decode_splitk.value,
        SplitkPolicy::Legacy,
        "GB10 is unchanged"
    );
    assert!(!empty(&HOPPER).attn_decode_splitk.from_env());

    // The A/B an H100 round runs against the new default.
    let off = with(&HOPPER, &[("ATLAS_ATTN_DECODE_SPLITK", "0")]);
    assert_eq!(off.attn_decode_splitk.value, SplitkPolicy::Pinned(1));
    assert!(off.attn_decode_splitk.from_env());
    assert!(
        format_levers(&off).contains("attn_decode_splitk=1 (env)"),
        "{}",
        format_levers(&off)
    );

    // …and the one that arms it on a target that declares `legacy`.
    let on = with(&GB10, &[("ATLAS_ATTN_DECODE_SPLITK", "auto")]);
    assert_eq!(on.attn_decode_splitk.value, SplitkPolicy::Auto);
    assert!(
        format_levers(&on).contains("attn_decode_splitk=auto (env)"),
        "{}",
        format_levers(&on)
    );
}
