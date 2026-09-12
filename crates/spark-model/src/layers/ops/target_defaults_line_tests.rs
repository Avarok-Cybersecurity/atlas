// SPDX-License-Identifier: AGPL-3.0-only

//! The serve log line, pinned (section 4 of the target-table tests).
//!
//! Split from `target_defaults_tests.rs` when #927's fused attention Q/K/V
//! row and #927's strided GDN decode row were merged onto the H100
//! integration branch together and carried the parent past the 500-line
//! cap. Neither branch crossed it alone — the same case
//! `target_defaults_splitk_tests.rs` was drawn for, and the same seam.
//!
//! The formatter these grade moved in the same commit, to
//! `target_defaults_line.rs`; this file is its opposite number. Fixtures
//! (`GB10`, `HOPPER`, `empty`, `with`) come from the parent rather than
//! being copied — a second copy is how the table and the line that reports
//! it come to disagree.

use super::*;

// ── (4) the serve log line ──

/// THE OPERATOR-FACING DELIVERABLE. A serve prints ONE line naming every
/// resolved value and marking the ones the environment supplied, so a number
/// reported from a run can be attributed to a configuration without also
/// having the launch script that produced it — which was the whole failure the
/// 2026-09-11 review named.
///
/// Graded through `format_levers` rather than `summary_line()`: the latter
/// reads (and seals) the process-wide `OnceLock`, which would make this test
/// order-dependent and would grade the environment of whoever ran it.
#[test]
fn the_serve_line_names_every_lever_and_marks_the_environment_ones() {
    let line = format_levers(&empty(&HOPPER));
    assert!(line.starts_with("target defaults (hopper): "), "{line}");
    for field in [
        "cublas_gemm_scope=ffn,attn,ssm",
        "ffn_batch16_tier=off",
        "ffn_m16_tc=off",
        "attn_m16_tc=on",
        "attn_ncol_gemv=off",
        "lm_head_m16_tc=on",
        "lm_head_batchm_max=16",
        "ssm_batched_recurrent=on",
        "gdn_decode_hopper=off",
        "gdn_decode_strided_hopper=on",
        "gdn_prefill_tc=on",
        "ssm_ba_gates_hopper=on",
        "decode_split_silu=on",
        "ssm_decode_ring_slots=auto",
        "w8a8_prefill_max_m=max/max",
        "attn_decode_splitk=auto",
    ] {
        assert!(line.contains(field), "missing `{field}` in:\n{line}");
    }
    assert!(
        !line.contains("(env)"),
        "an empty environment must mark nothing as an override:\n{line}"
    );

    // …and the override is VISIBLE. A line that reported the value without its
    // provenance would let a run be labelled with the target's recipe while it
    // actually ran the operator's.
    let overridden = with(
        &GB10,
        &[
            ("ATLAS_ATTN_M16_TC", "1"),
            ("ATLAS_LM_HEAD_BATCHM_MAX", "16"),
        ],
    );
    let line = format_levers(&overridden);
    assert!(line.contains("attn_m16_tc=on (env)"), "{line}");
    assert!(line.contains("lm_head_batchm_max=16 (env)"), "{line}");
    assert!(line.contains("lm_head_m16_tc=off"), "{line}");
    assert!(!line.contains("lm_head_m16_tc=off (env)"), "{line}");
    // GB10 keeps the pre-#928 split rule and says so.
    assert!(line.contains("attn_decode_splitk=legacy"), "{line}");
}

/// A target that names no hardware (a build that read no HARDWARE.toml) still
/// produces a readable line rather than `target defaults (): …`.
#[test]
fn an_unnamed_target_still_prints_a_readable_line() {
    let anon = TargetDefaults { hw: "", ..GB10 };
    assert!(
        format_levers(&empty(&anon)).starts_with("target defaults (unknown): "),
        "{}",
        format_levers(&empty(&anon))
    );
}
