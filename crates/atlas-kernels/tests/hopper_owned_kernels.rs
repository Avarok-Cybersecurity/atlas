// SPDX-License-Identifier: AGPL-3.0-only

//! The maintainer rule of 2026-09-11 — a kernel tuned for one hardware set
//! REPLACES its link in that set's `common/` and leaves the gb10 source
//! untouched, rather than editing a file five other targets compile — as a
//! property of the checked-in tree.
//!
//! Split out of `inherited_targets.rs` when #928's GDN prefill remnant twins
//! took that file past the 500-line cap. It reads [`HOPPER_OWNED_COMMON`] and
//! the tree helpers from the same `support/inherited.rs` every sibling binary
//! reads, so a target or an owned source can never be declared to one of them
//! and not the others.
//!
//! The rule has two shapes and each gets a test. An OVERRIDE is a real file
//! whose stem gb10 also has; an ADDITION is a new stem, which overrides
//! nothing and is therefore free in exactly the way that needs a guard.

#[path = "support/inherited.rs"]
mod inherited;

use inherited::{HOPPER_OWNED_COMMON, gb10_dir, hw_dir};

/// The maintainer rule of 2026-09-11, as a property of the checked-in tree:
/// a Hopper-tuned kernel OVERRIDES its gb10 namesake and does not edit it.
///
/// Three things, because the rule has three ways to be broken and the `owned`
/// list only covers the first:
///  1. the hopper entry is a real file (the override exists at all);
///  2. the gb10 source it overrides is still gb10's — a regular file, still
///     the shared-memory `E4M3_LUT` gather, with no Hopper instruction in it.
///     `w8a16_gemv.cu` is compiled by gb10, b200, strix and strix-hip; editing
///     it to serve Hopper would change all four;
///  3. the OTHER inherited target still links to gb10. An override that leaked
///     into b200 would ship sm_90a-tuned code to a B200 with no receipt.
///
/// The instruction strings are the discriminator because they are what the
/// override is FOR — see `kernels/hopper/common/w8a16_gemv_hopper.cuh`.
#[test]
fn a_hopper_owned_kernel_overrides_gb10_without_editing_it() {
    let hopper = hw_dir("hopper").join("common");
    let gb10 = gb10_dir().join("common");
    // An owned source is one of two things, and only the first is an
    // OVERRIDE: a file whose stem gb10 also has. A file with a NEW stem —
    // `gdn_decode_hopper.cu` (#927) — overrides nothing, so the three
    // assertions below have no gb10 side to check and are answered by
    // `a_hopper_owned_addition_brings_entry_points_gb10_does_not` instead.
    let sources: Vec<&str> = HOPPER_OWNED_COMMON
        .iter()
        .copied()
        .filter(|n| n.ends_with(".cu") && gb10.join(n).exists())
        .collect();
    assert!(
        !sources.is_empty(),
        "the owned list has no gb10-overriding sources to check"
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

/// The other half of the rule, for an owned source with a NEW stem.
///
/// `gdn_decode_hopper.cu` (#927) does not replace a gb10 file; it adds entry
/// points beside them, because its gb10 namesake `gated_delta_rule.cu` is
/// shadowed out of the build by the model directory's own copy and a
/// same-stem override in `common/` would never be compiled.
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
    let additions: Vec<&str> = HOPPER_OWNED_COMMON
        .iter()
        .copied()
        .filter(|n| n.ends_with(".cu") && !gb10.join(n).exists())
        .collect();

    // Entry names and file hashes of everything gb10's common/ declares.
    // `__launch_bounds__` SITS BETWEEN `void` AND THE NAME, and taking the
    // first token blind reported the attribute as the entry point. Every gb10
    // kernel that carries launch bounds then contributed the SAME fake name, so
    // the collision check below fired on `__launch_bounds__` for any addition
    // that used them — and was blind to the real names it exists to protect.
    let entry = |text: &str| -> Vec<String> {
        text.lines()
            .filter_map(|l| l.split_once("__global__ void "))
            .filter_map(|(_, rest)| {
                let rest = rest.trim_start();
                let rest = match rest.strip_prefix("__launch_bounds__") {
                    Some(r) => r.split_once(')').map(|(_, t)| t).unwrap_or(r),
                    None => rest,
                };
                let name = rest.trim().split(['(', ' ']).next()?;
                (!name.is_empty()).then(|| name.to_string())
            })
            .collect()
    };
    let mut gb10_entries = std::collections::BTreeSet::new();
    let mut gb10_bodies = std::collections::BTreeSet::new();
    for f in std::fs::read_dir(&gb10).expect("gb10 common").flatten() {
        let path = f.path();
        if path.extension().and_then(|e| e.to_str()) != Some("cu") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("gb10 source");
        gb10_entries.extend(entry(&text));
        gb10_bodies.insert(text);
    }

    for name in additions {
        let path = hopper.join(name);
        assert!(
            std::fs::read_link(&path).is_err(),
            "kernels/hopper/common/{name} is declared owned and is still a symlink"
        );
        let text = std::fs::read_to_string(&path).expect("hopper source");
        assert!(
            !gb10_bodies.contains(&text),
            "kernels/hopper/common/{name} is byte-identical to a gb10 common source — \
             an undeclared fork under a new name, not a tuned addition"
        );
        let declared = entry(&text);
        assert!(
            !declared.is_empty(),
            "kernels/hopper/common/{name} declares no entry point, so nothing can \
             dispatch to it"
        );
        for e in declared {
            assert!(
                !gb10_entries.contains(&e),
                "kernels/hopper/common/{name} re-declares `{e}`, which kernels/gb10/common \
                 already defines: one target would compile two definitions of one kernel name"
            );
        }
    }
}
