// SPDX-License-Identifier: AGPL-3.0-only

//! The tensor-core GDN prefill spine's ENTRY NAME and its route line.
//!
//! # Why the name is a constant
//!
//! H100 round 12 (`h100-round12-report.md`, stage 2c and 4b): the serve logged
//!
//! ```text
//! GDN state spine: gated_delta_rule_chunk_delta_h_tcfuse (ATLAS_GDN_PREFILL_TC; …)
//! ```
//!
//! while the nsys trace of the same cell showed launches of
//! `gated_delta_rule_chunk_delta_h_tcfuse_x2` and **none** of the 1-limb entry.
//! The log named the FAMILY; the binary ran the `_x2` member. That is not a
//! cosmetic difference on this kernel — the 1-limb `…_tcfuse` entry misses the
//! spine's own accuracy contract (`h` rel_rms 2.0e-3 to 2.7e-3 against a 1e-3
//! budget, at every T the microtest runs) and the `_x2` arm is the one that
//! passes it (3.0e-6 to 3.9e-6). A reader diffing the serve log against the
//! microtest's arm names reads the shipped arm as the ungated one.
//!
//! So the name exists ONCE, here. `qwen3_ssm::init_kernels` binds the handle
//! with it and [`gdn_tc_spine_route_line`] prints it; neither spells a string
//! of its own, which is the only arrangement in which the two cannot drift
//! apart again.

/// The spine entry the `[defaults] gdn_prefill_tc` lever ships.
///
/// `_x2` = two bf16 limbs of `S_c` in Phase A. Both entries are ABI-, grid-,
/// block- and smem-identical, so the choice is invisible downstream and only
/// the accuracy contract separates them (see the module docs).
pub const GDN_TC_SPINE_ENTRY: &str = "gated_delta_rule_chunk_delta_h_tcfuse_x2";

/// The module the entry lives in — `kernels/gb10/common/
/// gated_delta_rule_chunk_tc.cu`, shared, not relocated to `kernels/hopper`.
pub const GDN_TC_SPINE_MODULE: &str = "gated_delta_rule_chunk_tc";

/// `GDN state spine: …` — the line the prefill prints when the tensor-core
/// spine is live, built from [`GDN_TC_SPINE_ENTRY`] so it can only ever name
/// the kernel the probe bound.
///
/// Pure and returning a `String` rather than logging: a route line nothing can
/// grade is how a log comes to describe a kernel the binary does not run,
/// which is the defect this file exists to close.
pub fn gdn_tc_spine_route_line(num_v_heads: u32, batch_size: u32, smem_bytes: u32) -> String {
    format!(
        "GDN state spine: {GDN_TC_SPINE_ENTRY} (ATLAS_GDN_PREFILL_TC; bf16 mma.sync \
         operands, f32 accumulator = the recurrent state, h stays f32) \
         grid=[{num_v_heads},{batch_size}] block=256 smem={smem_bytes}B"
    )
}

#[cfg(test)]
mod tests {
    use super::{GDN_TC_SPINE_ENTRY, gdn_tc_spine_route_line};

    /// THE ROUND-12 NIT, pinned: the line names the `_x2` entry, not the
    /// family. `…_tcfuse ` with a trailing space is what the old line printed
    /// and is what a reader would mistake for the ungated 1-limb arm.
    #[test]
    fn the_route_line_names_the_entry_that_is_launched() {
        let line = gdn_tc_spine_route_line(48, 1, 88_324);
        assert!(line.contains(GDN_TC_SPINE_ENTRY), "{line}");
        assert!(
            !line.contains("gated_delta_rule_chunk_delta_h_tcfuse "),
            "the line must not name the FAMILY where the binary launches the \
             `_x2` member — round 12 stage 4b:\n{line}"
        );
    }

    /// …and it still carries the launch geometry an operator reads it for.
    #[test]
    fn the_route_line_carries_the_geometry() {
        let line = gdn_tc_spine_route_line(48, 2, 88_324);
        for field in [
            "grid=[48,2]",
            "block=256",
            "smem=88324B",
            "ATLAS_GDN_PREFILL_TC",
        ] {
            assert!(line.contains(field), "missing `{field}` in:\n{line}");
        }
    }
}
