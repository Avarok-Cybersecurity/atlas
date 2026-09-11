// SPDX-License-Identifier: AGPL-3.0-only

//! What a DECLARED `[kernels] overrides` entry must actually BE on disk.
//!
//! `inherited_targets.rs` proves each inheriting `common/` is gb10's directory
//! plus exactly the names its `HARDWARE.toml` declares — it checks the NAMES.
//! Whether a declared name is a tuned source or a link nobody updated is
//! invisible to that check, and a declaration that lies is worse than no
//! declaration: it tells a reader the target was tuned.
//!
//! So this binary checks the CONTENT, in the two shapes the rule allows
//! (maintainer rule, 2026-09-11 — tbraun96):
//!
//! * an OVERRIDE replaces a gb10 namesake. Same entry points, this hardware's
//!   instruction selection, and gb10's own file untouched, because gb10, b200,
//!   strix and strix-hip all compile it.
//! * an ADDITION has a stem gb10 does not have at all, and must bring entry
//!   points gb10 does not declare — otherwise one target compiles two
//!   definitions of one kernel name.
//!
//! Its own binary rather than more tests in `inherited_targets.rs` because that
//! file is at the house 500-LoC cap; the split is by QUESTION, not by size, so
//! each file still reads as one argument.

#[path = "support/inherited.rs"]
mod inherited;
// THE resolver `build.rs` and `kernel_shadow_detector.rs` use. A `__global__
// void ` grep is NOT a substitute: `dense_gemm_m16_bf16.cu` spells its two
// entries `extern "C" __global__` on one line and `void <name>(` on the next,
// and twenty-one `common/*.cu` files declare theirs only through macros. A
// scan that misses them reports "declares nothing", which is how this file
// first went red.
#[path = "../build_shadow.rs"]
#[allow(dead_code)] // only `entry_points` is this binary's question
mod build_shadow;

use build_shadow::entry_points;
use inherited::{gb10_dir, hw_dir, kernel_overrides};

use std::path::PathBuf;

/// The maintainer rule of 2026-09-11, as a property of the checked-in tree:
/// a declared OVERRIDE replaces its gb10 namesake and does not edit it.
///
/// Three things, because the rule has three ways to be broken and the
/// declaration in `HARDWARE.toml` only covers the first:
///  1. the hopper entry is a real file (the override exists at all);
///  2. the gb10 source it overrides is still gb10's — a regular file, still
///     the shared-memory `E4M3_LUT` gather, with no Hopper instruction in it.
///     `w8a16_gemv.cu` is compiled by gb10, b200, strix and strix-hip; editing
///     it to serve Hopper would change all four;
///  3. the OTHER inherited target still links to gb10. An override that leaked
///     into b200 would ship sm_90a-tuned code to a B200 with no receipt.
///
/// The instruction strings are the discriminator because they are what the
/// override is FOR — see `kernels/hopper/common/w8a16_gemv_hopper.cuh`. They
/// are the W8A16 GEMV family's witnesses, which is the whole of the
/// override-in-place set today; declaring an override of some OTHER gb10 file
/// will red here until this test is given that file's witness, which is the
/// review this test exists to force.
#[test]
fn a_hopper_owned_kernel_overrides_gb10_without_editing_it() {
    let hopper = hw_dir("hopper").join("common");
    let gb10 = gb10_dir().join("common");
    // A declared entry is one of two things, and only the first is an
    // OVERRIDE: a file whose stem gb10 also has. A file with a NEW stem —
    // `gdn_decode_hopper.cu` (#927), and the four sources moved out of gb10 —
    // replaces nothing, so the three assertions below have no gb10 side to
    // check and are answered by
    // `a_hopper_owned_addition_brings_entry_points_gb10_does_not` instead.
    let declared = kernel_overrides("hopper");
    let sources: Vec<&str> = declared
        .iter()
        .map(String::as_str)
        .filter(|n| n.ends_with(".cu") && gb10.join(n).exists())
        .collect();
    assert!(
        !sources.is_empty(),
        "kernels/hopper/HARDWARE.toml [kernels] overrides has no gb10-overriding \
         sources to check"
    );

    for name in sources {
        let over = hopper.join(name);
        assert!(
            std::fs::read_link(&over).is_err(),
            "kernels/hopper/common/{name} is still a symlink; it is declared as \
             a Hopper override"
        );
        let over_text = std::fs::read_to_string(&over).unwrap();
        assert!(
            over_text.contains("cvt.rn.f16x2.e4m3x2") || over_text.contains("w8a16_gemv_hopper"),
            "kernels/hopper/common/{name} overrides gb10 without using anything \
             this hardware has; an override that is not tuned is drift"
        );

        let base = gb10.join(name);
        assert!(
            std::fs::read_link(&base).is_err(),
            "kernels/gb10/common/{name} must stay a real file"
        );
        let base_text = std::fs::read_to_string(&base).unwrap();
        assert!(
            base_text.contains("s_lut["),
            "kernels/gb10/common/{name} no longer holds the shared-memory E4M3 \
             LUT gather — the Hopper work edited gb10 instead of overriding it"
        );
        assert!(
            !base_text.contains("cvt.rn.f16x2.e4m3x2"),
            "kernels/gb10/common/{name} gained a Hopper dequant instruction; \
             gb10, b200, strix and strix-hip all compile this file"
        );

        let b200 = hw_dir("b200").join("common").join(name);
        assert!(
            std::fs::read_link(&b200).is_ok(),
            "kernels/b200/common/{name} stopped being a link to gb10; the \
             override leaked to a target with no receipt for it"
        );
    }
}

/// The other half of the rule, for a declared source with a NEW stem.
///
/// `gdn_decode_hopper.cu` (#927) does not replace a gb10 file; it adds entry
/// points beside them, because its gb10 namesake `gated_delta_rule.cu` is
/// shadowed out of the build by the model directory's own copy and a
/// same-stem override in `common/` would never be compiled. The four sources
/// MOVED out of `kernels/gb10/common` — `dense_gemm_m16_bf16.cu`,
/// `fp8_scale_transpose.cu`, `w8a16_gemm_m16.cu`, `w8a16_gemv_ncol.cu` — are
/// the same shape from the other direction: gb10 no longer has them, so a
/// GB10 build does not compile a kernel it has no receipt for.
///
/// That freedom is exactly what needs a guard. A new stem that re-declared an
/// entry gb10 already declares would put two definitions of one kernel name in
/// one target's module set, and a new stem whose bytes are a gb10 file's are a
/// fork wearing a new name. Both are checked here; neither is visible to
/// `mirror_faults`, which only knows the name is declared.
#[test]
fn a_hopper_owned_addition_brings_entry_points_gb10_does_not() {
    let hopper = hw_dir("hopper").join("common");
    let gb10 = gb10_dir().join("common");
    let declared = kernel_overrides("hopper");
    let additions: Vec<&str> = declared
        .iter()
        .map(String::as_str)
        .filter(|n| n.ends_with(".cu") && !gb10.join(n).exists())
        .collect();
    assert!(
        !additions.is_empty(),
        "kernels/hopper/HARDWARE.toml [kernels] overrides has no additions to check"
    );

    // Entry names and file bodies of everything gb10's common/ declares.
    let mut gb10_entries = std::collections::BTreeSet::new();
    let mut gb10_bodies = std::collections::BTreeSet::new();
    for f in std::fs::read_dir(&gb10).expect("gb10 common").flatten() {
        let path = f.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        gb10_entries.extend(entry_points(&path));
        gb10_bodies.insert(std::fs::read_to_string(&path).expect("gb10 source"));
    }

    for name in additions {
        let path = hopper.join(name);
        assert!(
            std::fs::read_link(&path).is_err(),
            "kernels/hopper/common/{name} is declared and is still a symlink"
        );
        let text = std::fs::read_to_string(&path).expect("hopper source");
        assert!(
            !gb10_bodies.contains(&text),
            "kernels/hopper/common/{name} is byte-identical to a gb10 common source — \
             an undeclared fork under a new name, not a tuned addition"
        );
        let entries = entry_points(&path);
        assert!(
            !entries.is_empty(),
            "kernels/hopper/common/{name} declares no entry point, so nothing can \
             dispatch to it"
        );
        for e in &entries {
            assert!(
                !gb10_entries.contains(e),
                "kernels/hopper/common/{name} re-declares `{e}`, which kernels/gb10/common \
                 already defines: one target would compile two definitions of one kernel name"
            );
        }
    }
}

/// The VALUE-SPLIT prefill spine, pinned by name (#928).
///
/// The generic addition rule above proves the file declares entries gb10 does
/// not. This pins the two that matter, because the ONE thing a reader of the
/// serve log has to be able to trust about this lever is that the split it
/// names is the kernel that launched — `ops::gdn_spine_vsplit_entry` maps the
/// resolved split to these strings, `qwen3_ssm::init_kernels` binds the handle
/// with them, and the route line prints them. A rename on the CUDA side that
/// this test did not catch would leave all three resolving to a handle of 0 and
/// the launcher silently on the unsplit parent, which is the PR #296 shape.
///
/// It is also where the 2- and 4-way arms are declared to EXIST: the lever's
/// grammar (`ops::GDN_SPINE_VSPLIT_VALUES`) accepts exactly those two, and a
/// split with no compiled entry behind it is the one failure the lever cannot
/// report at runtime.
#[test]
fn the_value_split_spine_declares_both_of_its_entry_points() {
    let path = hw_dir("hopper")
        .join("common")
        .join("gdn_chunk_delta_h_vsplit_hopper.cu");
    assert!(
        kernel_overrides("hopper")
            .iter()
            .any(|n| n == "gdn_chunk_delta_h_vsplit_hopper.cu"),
        "the value-split spine must be DECLARED in [kernels] overrides — an \
         undeclared regular file in the mirror is indistinguishable from a \
         silent fork of a shared kernel"
    );
    let entries = entry_points(&path);
    for want in [
        "gated_delta_rule_chunk_delta_h_vsplit2_hopper",
        "gated_delta_rule_chunk_delta_h_vsplit4_hopper",
    ] {
        assert!(
            entries.contains(want),
            "{want} is not among the entry points scraped from \
             gdn_chunk_delta_h_vsplit_hopper.cu: {entries:?}"
        );
    }
    // ...and nothing else. A third entry would be a split the lever cannot
    // name, i.e. a kernel compiled into every Hopper image that nothing can
    // dispatch to.
    assert_eq!(entries.len(), 2, "{entries:?}");
}

/// B200 reaches every source IT declares by relative symlink into Hopper's
/// copy, rather than by carrying a second regular file with the same bytes —
/// which is the cross-target duplicate `scripts/check_kernel_shadows.py` RULE2
/// forbids.
///
/// That is a SHARING decision, not an inheritance one: these four are ordinary
/// datacentre MMA with nothing sm_90a about them, so both targets want the one
/// source. The day a B200 receipt calls for something different, the fix is a
/// real file under `kernels/b200/common` and a line in that HARDWARE.toml
/// saying what diverged — which this test would then need updating for, on
/// purpose.
#[test]
fn b200_reaches_its_declared_overrides_by_link_into_hopper() {
    let declared = kernel_overrides("b200");
    assert!(
        !declared.is_empty(),
        "kernels/b200/HARDWARE.toml declares no [kernels] overrides"
    );
    for name in &declared {
        let hopper = hw_dir("hopper").join("common").join(name);
        assert!(
            std::fs::symlink_metadata(&hopper)
                .unwrap_or_else(|e| panic!("{}: {e}", hopper.display()))
                .file_type()
                .is_file(),
            "kernels/hopper/common/{name} must be a REAL FILE — editing a \
             symlink here would edit gb10's kernel, which is what the rule \
             exists to prevent"
        );
        let b200 = hw_dir("b200").join("common").join(name);
        let link = std::fs::read_link(&b200)
            .unwrap_or_else(|e| panic!("kernels/b200/common/{name}: expected a symlink: {e}"));
        assert_eq!(
            link,
            PathBuf::from("../../hopper/common").join(name),
            "kernels/b200/common/{name} must point at Hopper's copy"
        );
    }
}
