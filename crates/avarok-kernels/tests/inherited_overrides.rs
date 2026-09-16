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
//! (maintainer rule, 2026-09-11 — tbraun96: "symlinks are fine provided the
//! pointed-to gb10 file is not edited when iterating on Hopper; Hopper-tuned
//! kernels must be real files under `kernels/hopper/`"):
//!
//! * an OVERRIDE replaces a gb10 namesake. Same entry points, this hardware's
//!   instruction selection, and gb10's own file untouched, because gb10,
//!   b200, strix and strix-hip all compile it.
//! * an ADDITION has a stem gb10 does not have at all, and must bring entry
//!   points gb10 does not declare — otherwise one target compiles two
//!   definitions of one kernel name.
//!
//! ★ NO TEST HERE ASSERTS THAT ANYTHING IS DECLARED. Both lists are EMPTY on
//! this tip: every inheriting target takes all of gb10's kernels, so the cases
//! below iterate nothing and pass vacuously. That is deliberate and not a
//! placeholder. An emptiness assertion would have to be DELETED by the first
//! PR that adds a tuned kernel, which is the worst possible moment to be
//! editing the test that guards it; written this way the declaration a kernel
//! PR adds is graded the moment it lands, with this file untouched. That the
//! rules themselves can FAIL is proved on fixtures in `support/mirror.rs`,
//! which is where a checker that always returns nothing would be caught.
//!
//! Its own binary rather than more tests in `inherited_targets.rs` because
//! that file is near the house 500-LoC cap; the split is by QUESTION, not by
//! size, so each file still reads as one argument.

#[path = "support/inherited.rs"]
mod inherited;
// THE resolver `build.rs` and `kernel_shadow_detector.rs` use. A `__global__
// void ` grep is NOT a substitute: entry points are spelled across two lines
// in some sources and declared only through macros in others, and a scan that
// misses them reports "declares nothing", which would make the addition rule
// below pass on a file that re-declares a gb10 kernel.
#[path = "../build_shadow.rs"]
#[allow(dead_code)] // only `entry_points` is this binary's question
mod build_shadow;

use build_shadow::entry_points;
use inherited::{INHERITED, gb10_dir, hw_dir, kernel_overrides};

use std::collections::BTreeSet;
use std::path::PathBuf;

/// A declared name that gb10 ALSO has is an OVERRIDE: it must be a real file
/// here, and gb10's own copy must be untouched by it.
///
/// Three things, because the rule has three ways to be broken and the
/// declaration in `HARDWARE.toml` only covers the first:
///  1. the entry is a real file (the override exists at all);
///  2. the gb10 source it overrides is still a regular file — gb10, b200,
///     strix and strix-hip may all compile it, so editing it to serve one
///     target would change every one of them;
///  3. its bytes are not gb10's. An "override" that is a copy is drift with a
///     declaration attached, and the declaration is what makes a reviewer
///     believe it was tuned.
#[test]
fn a_declared_override_replaces_its_gb10_namesake_without_editing_it() {
    let gb10 = gb10_dir().join("common");
    for t in INHERITED {
        let common = hw_dir(t.hw).join("common");
        for name in kernel_overrides(t.hw) {
            let base = gb10.join(&name);
            if !base.exists() {
                continue; // an ADDITION; the next test owns it
            }
            let over = common.join(&name);
            assert!(
                std::fs::read_link(&over).is_err(),
                "kernels/{}/common/{name} is still a symlink; it is declared as \
                 an override",
                t.hw
            );
            assert!(
                std::fs::read_link(&base).is_err(),
                "kernels/gb10/common/{name} must stay a real file — an override \
                 that turned the origin into a link would redirect every target \
                 that inherits it"
            );
            let over_text = std::fs::read_to_string(&over).expect("override source");
            let base_text = std::fs::read_to_string(&base).expect("gb10 source");
            assert_ne!(
                over_text, base_text,
                "kernels/{}/common/{name} is byte-identical to gb10's — a copy \
                 with a declaration attached, not a tuned override",
                t.hw
            );
        }
    }
}

/// A declared name gb10 does NOT have is an ADDITION, and its freedom is
/// exactly what needs a guard.
///
/// A new stem that re-declared an entry gb10 already declares would put two
/// definitions of one kernel name in one target's module set, and a new stem
/// whose bytes are a gb10 file's is a fork wearing a new name. Both are
/// checked here; neither is visible to `mirror_faults`, which only knows the
/// name is declared.
#[test]
fn a_declared_addition_brings_entry_points_gb10_does_not() {
    let gb10 = gb10_dir().join("common");
    // Entry names and file bodies of everything gb10's common/ declares.
    let mut gb10_entries = BTreeSet::new();
    let mut gb10_bodies = BTreeSet::new();
    for f in std::fs::read_dir(&gb10).expect("gb10 common").flatten() {
        let path = f.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        gb10_entries.extend(entry_points(&path));
        gb10_bodies.insert(std::fs::read_to_string(&path).expect("gb10 source"));
    }

    for t in INHERITED {
        let common = hw_dir(t.hw).join("common");
        for name in kernel_overrides(t.hw) {
            if gb10.join(&name).exists() || !name.ends_with(".cu") {
                // An OVERRIDE (previous test), or a header, which declares no
                // entry point of its own and is checked only for presence.
                continue;
            }
            let path = common.join(&name);
            let text = std::fs::read_to_string(&path).expect("addition source");
            assert!(
                !gb10_bodies.contains(&text),
                "kernels/{}/common/{name} is byte-identical to a gb10 common \
                 source — an undeclared fork under a new name, not a tuned \
                 addition",
                t.hw
            );
            let entries = entry_points(&path);
            assert!(
                !entries.is_empty(),
                "kernels/{}/common/{name} declares no entry point, so nothing \
                 can dispatch to it",
                t.hw
            );
            for e in &entries {
                assert!(
                    !gb10_entries.contains(e),
                    "kernels/{}/common/{name} re-declares `{e}`, which \
                     kernels/gb10/common already defines: one target would \
                     compile two definitions of one kernel name",
                    t.hw
                );
            }
        }
    }
}

/// A name declared by MORE THAN ONE inheriting target is reached by exactly
/// one real file, with the others linking to it.
///
/// That is a SHARING decision, not an inheritance one: two datacentre parts
/// may well want one tuned source, and a second regular file with the same
/// bytes is the cross-target duplicate `scripts/check_kernel_shadows.py`
/// RULE2 forbids. The day one of them needs something different, the fix is a
/// real file and a line in that target's HARDWARE.toml saying what diverged.
#[test]
fn a_shared_override_has_one_real_file_and_the_rest_link_to_it() {
    let mut owners: std::collections::BTreeMap<String, Vec<&str>> = Default::default();
    for t in INHERITED {
        for name in kernel_overrides(t.hw) {
            owners.entry(name).or_default().push(t.hw);
        }
    }
    for (name, targets) in owners {
        if targets.len() < 2 {
            continue;
        }
        let real: Vec<&str> = targets
            .iter()
            .copied()
            .filter(|hw| {
                std::fs::symlink_metadata(hw_dir(hw).join("common").join(&name))
                    .is_ok_and(|m| m.file_type().is_file())
            })
            .collect();
        assert_eq!(
            real.len(),
            1,
            "{name} is declared by {targets:?}; exactly one may hold the real \
             source (the others link to it), got {real:?}"
        );
        for hw in targets.iter().filter(|hw| **hw != real[0]) {
            let link = std::fs::read_link(hw_dir(hw).join("common").join(&name))
                .unwrap_or_else(|e| panic!("kernels/{hw}/common/{name}: expected a symlink: {e}"));
            assert_eq!(
                link,
                PathBuf::from("../..")
                    .join(real[0])
                    .join("common")
                    .join(&name),
                "kernels/{hw}/common/{name} must point at {}'s copy",
                real[0]
            );
        }
    }
}
