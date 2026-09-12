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

/// The SCALAR spine entries `qwen3_ssm::init` can bind — `ATLAS_GDN_PIPE=1`,
/// `ATLAS_GDN_VTILE=1`, and the default. Named here beside the tensor-core
/// entry for the same reason that one is: the init route line and the handle
/// are built from the same string or they drift apart.
pub const GDN_SCALAR_SPINE_PIPE: &str = "gated_delta_rule_chunk_delta_h_pipe";
/// SPLIT=4 / 512 threads. Reachable, never default — see `init_kernels`.
pub const GDN_SCALAR_SPINE_VTILE: &str = "gated_delta_rule_chunk_delta_h_vtile";
/// SPLIT=2 / 256 threads: the default scalar spine.
pub const GDN_SCALAR_SPINE_VFUSED: &str = "gated_delta_rule_chunk_delta_h_vfused";

/// `GDN state spine: …` — the line `qwen3_ssm::init` prints ONCE PER LAYER as
/// it binds the handles, before any prefill has run.
///
/// # Why it is not simply the scalar entry's name
///
/// H100 round 14 (`h100-round14-report.md`, anomaly 2). A serve with the
/// tensor-core spine live logged both of these:
///
/// ```text
///    48  qwen3_ssm::init: GDN state spine: gated_delta_rule_chunk_delta_h_vfused
/// 14400  GDN state spine: gated_delta_rule_chunk_delta_h_tcfuse_x2 (…)
/// ```
///
/// 48 init lines naming the scalar parent, one per layer, ahead of 14 400
/// dispatch lines naming the kernel that actually ran. `f9ae638` fixed the
/// dispatch line; the init line was not touched, and it is the FIRST GDN line
/// a reader meets in a log they opened to answer "did the lever engage?" — so
/// it read as "the TC spine is not engaged" on a serve where it was.
///
/// The line now reads the handle the probe resolved, which is the same bit the
/// dispatch reads: with the `[defaults] gdn_prefill_tc` handle bound, the spine
/// that will launch is [`GDN_TC_SPINE_ENTRY`] and the line says so. The scalar
/// entry stays bound underneath — the dispatch's shape guards fall back to it,
/// and the dispatch logs whichever one it launched — but it is no longer what
/// this line NAMES, which was the whole defect.
/// `vsplit_entry` is `Some` only when `[defaults] gdn_spine_vsplit` resolved to
/// a split AND that split's twin is bound — in which case it, not
/// [`GDN_TC_SPINE_ENTRY`], is the entry the prefill will launch, and naming the
/// parent here would re-create the round-14 defect one lever down.
pub fn gdn_init_spine_line(
    tc_spine_bound: bool,
    vsplit_entry: Option<&str>,
    scalar_entry: &str,
) -> String {
    match (tc_spine_bound, vsplit_entry) {
        (true, Some(entry)) => format!(
            "GDN state spine: {entry} ([defaults] gdn_prefill_tc + gdn_spine_vsplit; \
             the unsplit {GDN_TC_SPINE_ENTRY} and the scalar spine stay bound as the \
             fallbacks the prefill's shape guards drop to, and the prefill logs the \
             entry it launches)"
        ),
        (true, None) => format!(
            "GDN state spine: {GDN_TC_SPINE_ENTRY} ([defaults] gdn_prefill_tc; the \
             scalar spine stays bound as the fallback the prefill's shape guards \
             drop to, and the prefill logs the entry it launches)"
        ),
        (false, _) => format!("GDN state spine: {scalar_entry}"),
    }
}

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

/// The value-SPLIT twin's module — `kernels/hopper/common/
/// gdn_chunk_delta_h_vsplit_hopper.cu`, a Hopper-only ADDITION with new entry
/// names (declared in that target's `[kernels] overrides`), so the handle is
/// `KernelHandle(0)` on gb10, b200 and strix and the launcher stays on the
/// unsplit spine there without reading anything.
pub const GDN_SPINE_VSPLIT_MODULE: &str = "gdn_chunk_delta_h_vsplit_hopper";

/// 2-way value split: 2 CTAs per value head, 64 state columns each.
pub const GDN_SPINE_VSPLIT2_ENTRY: &str = "gated_delta_rule_chunk_delta_h_vsplit2_hopper";
/// 4-way value split: 4 CTAs per value head, 32 state columns each.
pub const GDN_SPINE_VSPLIT4_ENTRY: &str = "gated_delta_rule_chunk_delta_h_vsplit4_hopper";

/// `GDN state spine: …` — the line the prefill prints when the VALUE-SPLIT
/// twin is live, in place of [`gdn_tc_spine_route_line`].
///
/// A separate line and not a suffix on the other one, for the reason this
/// whole file exists: round 12 caught a serve log naming a FAMILY while the
/// binary launched a member, and the two are separated here by exactly the
/// fact a reader is checking — which entry, and on how many CTAs. The grid is
/// the whole claim of this lever (48 CTAs of 132 SMs at split 1; 96 at 2, 192
/// at 4), so it is in the line rather than inferable from it.
///
/// `entry` is passed in rather than derived so the caller cannot print one
/// split's name beside another split's geometry; `ops::gdn_spine_vsplit_entry`
/// is the only thing that maps a split to a name, and it returns these
/// constants.
pub fn gdn_vsplit_spine_route_line(
    entry: &str,
    num_v_heads: u32,
    grid_y: u32,
    split: u32,
    smem_bytes: u32,
) -> String {
    format!(
        "GDN state spine: {entry} ([defaults] gdn_spine_vsplit={split}; the value \
         dimension is split across CTAs — no cross-CTA reduction, bit-identical to \
         {GDN_TC_SPINE_ENTRY}) grid=[{num_v_heads},{grid_y}] block=256 smem={smem_bytes}B"
    )
}

#[cfg(test)]
mod tests {
    use super::{
        GDN_SCALAR_SPINE_PIPE, GDN_SCALAR_SPINE_VFUSED, GDN_SCALAR_SPINE_VTILE,
        GDN_SPINE_VSPLIT_MODULE, GDN_SPINE_VSPLIT2_ENTRY, GDN_SPINE_VSPLIT4_ENTRY,
        GDN_TC_SPINE_ENTRY, gdn_init_spine_line, gdn_tc_spine_route_line,
        gdn_vsplit_spine_route_line,
    };

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

    /// The value-split line names the ENTRY, carries the grid the lever exists
    /// to move, and says which parent it claims byte equality against.
    #[test]
    fn the_vsplit_route_line_names_the_entry_and_the_grid() {
        let line = gdn_vsplit_spine_route_line(GDN_SPINE_VSPLIT2_ENTRY, 48, 2, 2, 54_532);
        for field in [
            GDN_SPINE_VSPLIT2_ENTRY,
            GDN_TC_SPINE_ENTRY,
            "gdn_spine_vsplit=2",
            "grid=[48,2]",
            "block=256",
            "smem=54532B",
        ] {
            assert!(line.contains(field), "missing `{field}` in:\n{line}");
        }
    }

    /// The two splits are DIFFERENT entry points and must not share a name —
    /// the round-12 defect one level down, where a family name stood in for a
    /// member. They differ only in their split digit, which is the property
    /// that makes `ops::gdn_spine_vsplit_entry` a total function of the split.
    #[test]
    fn the_two_vsplit_entries_are_distinct_and_hopper_owned() {
        assert_ne!(GDN_SPINE_VSPLIT2_ENTRY, GDN_SPINE_VSPLIT4_ENTRY);
        for e in [GDN_SPINE_VSPLIT2_ENTRY, GDN_SPINE_VSPLIT4_ENTRY] {
            assert!(e.ends_with("_hopper"), "{e}");
            assert_ne!(e, GDN_TC_SPINE_ENTRY, "a twin must not shadow its parent");
        }
        assert_eq!(GDN_SPINE_VSPLIT_MODULE, "gdn_chunk_delta_h_vsplit_hopper");
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

    /// THE ROUND-14 NIT, pinned. With the tensor-core handle bound, the INIT
    /// line names the entry the dispatch will launch — not the scalar parent
    /// that merely stays bound behind it (round 14, anomaly 2: 48 of these
    /// lines said `…_vfused` while all 14 400 dispatches went to `…_x2`).
    #[test]
    fn the_init_line_names_the_tc_entry_when_its_handle_is_bound() {
        let line = gdn_init_spine_line(true, None, GDN_SCALAR_SPINE_VFUSED);
        assert!(line.contains(GDN_TC_SPINE_ENTRY), "{line}");
        assert!(
            !line.contains(GDN_SCALAR_SPINE_VFUSED),
            "the init line must not NAME the scalar spine where the probe bound \
             the tensor-core one:\n{line}"
        );
    }

    /// …and the init line and the dispatch line name the SAME entry, which is
    /// the property that makes 48 lines and 14 400 lines one answer.
    #[test]
    fn the_two_route_lines_agree_on_the_entry() {
        let init = gdn_init_spine_line(true, None, GDN_SCALAR_SPINE_VFUSED);
        let dispatch = gdn_tc_spine_route_line(48, 1, 88_324);
        for line in [&init, &dispatch] {
            assert!(line.contains(GDN_TC_SPINE_ENTRY), "{line}");
        }
    }

    /// THE ROUND-14 NIT, one lever down: with the value-split twin bound, the
    /// init line names the SPLIT entry. The parent stays bound underneath as
    /// the fallback the prefill's guards drop to — which is why it is still
    /// mentioned — but it is no longer what the line NAMES, because it is no
    /// longer what will launch.
    #[test]
    fn the_init_line_names_the_split_entry_when_the_twin_is_bound() {
        let line =
            gdn_init_spine_line(true, Some(GDN_SPINE_VSPLIT4_ENTRY), GDN_SCALAR_SPINE_VFUSED);
        assert!(line.contains(GDN_SPINE_VSPLIT4_ENTRY), "{line}");
        assert!(
            !line.starts_with(&format!("GDN state spine: {GDN_TC_SPINE_ENTRY}")),
            "the line must NAME the entry that launches, not the parent it \
             falls back to:\n{line}"
        );
        assert!(!line.contains(GDN_SCALAR_SPINE_VFUSED), "{line}");
    }

    /// With the probe OFF — every target but `kernels/hopper` today — the line
    /// is the scalar entry it has always been, in all three arms, and spells no
    /// string of its own.
    #[test]
    fn the_init_line_names_the_scalar_entry_when_the_probe_is_off() {
        for entry in [
            GDN_SCALAR_SPINE_PIPE,
            GDN_SCALAR_SPINE_VTILE,
            GDN_SCALAR_SPINE_VFUSED,
        ] {
            assert_eq!(
                gdn_init_spine_line(false, None, entry),
                format!("GDN state spine: {entry}"),
            );
        }
    }
}
