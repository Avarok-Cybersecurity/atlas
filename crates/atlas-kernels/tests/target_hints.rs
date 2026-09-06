// SPDX-License-Identifier: AGPL-3.0-only

//! `atlas_core::arch::target_hint` against the tree it claims to describe.
//!
//! `target_hint` answers "which `ATLAS_TARGET_HW=` should this operator have
//! built" for a device compute capability, and it is a hand-written `match`.
//! It lives in atlas-core because a shipped binary needs it and atlas-core is
//! the leaf every layer can reach — but that means it has no way to read
//! `kernels/`, so nothing made the two agree.
//!
//! `kernels/<hw>/HARDWARE.toml` already states the answer: `compute_capability`
//! is the device CC that hardware set is for. **Nothing in the repo read that
//! key** — `docs/HARDWARE.md` said so explicitly, and
//! `kernels/strix/HARDWARE.toml` records two neighbouring keys being deleted
//! for exactly that reason. This file is its first reader, and it is the whole
//! justification for keeping the key: the tree declares the mapping, and this
//! test holds the `match` to it.
//!
//! ORACLE: the `[hardware]` tables themselves. Every NVIDIA hardware set must
//! declare a `compute_capability` that `target_hint` maps back to that set's
//! own directory name. A new `kernels/<hw>/` whose CC the hint does not know
//! fails here rather than silently telling operators of that GPU that Atlas
//! ships nothing for them.

use std::path::{Path, PathBuf};

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/atlas-kernels is two levels below the workspace root")
        .join("kernels")
}

/// `(directory name, [hardware] table)` for every hardware set in the tree,
/// by `build.rs::resolve_targets`' own rule: a `kernels/<hw>/` with a
/// HARDWARE.toml is a hardware set.
fn hardware_sets() -> Vec<(String, toml::Value)> {
    let mut sets: Vec<(String, toml::Value)> = std::fs::read_dir(kernels_root())
        .expect("kernels/ is in the tree")
        .flatten()
        .filter_map(|entry| {
            let path = entry.path().join("HARDWARE.toml");
            if !path.is_file() {
                return None;
            }
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let toml: toml::Value = toml::from_str(&text)
                .unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()));
            let hw = toml
                .get("hardware")
                .unwrap_or_else(|| panic!("{}: no [hardware] table", path.display()))
                .clone();
            Some((entry.file_name().to_string_lossy().to_string(), hw))
        })
        .collect();
    sets.sort_by(|a, b| a.0.cmp(&b.0));
    sets
}

/// `"12.1"` -> `(12, 1)`. Rejects anything else loudly: a CC that does not
/// parse cannot be compared, and quietly skipping it would turn this test back
/// into the no-op it exists to replace. (`strix`'s `"11.5.1"` is a three-part
/// gfx version, not a CUDA compute capability — it is excluded by vendor
/// before it reaches here.)
fn parse_cc(text: &str, hw: &str) -> (u32, u32) {
    let (major, minor) = text.split_once('.').unwrap_or_else(|| {
        panic!("kernels/{hw}: compute_capability {text:?} is not `major.minor`")
    });
    (
        major
            .parse()
            .unwrap_or_else(|e| panic!("kernels/{hw}: compute_capability major {major:?}: {e}")),
        minor
            .parse()
            .unwrap_or_else(|e| panic!("kernels/{hw}: compute_capability minor {minor:?}: {e}")),
    )
}

/// Every NVIDIA hardware set declares a compute capability, and `target_hint`
/// maps it back to that set's directory name.
///
/// The failure this catches is silent in both directions: a new target whose
/// CC is missing from the `match` leaves every operator of that GPU reading
/// "no shipped target matches compute capability X.Y" from an image that in
/// fact ships one, and a `match` arm naming a directory that was renamed or
/// removed sends them to rebuild something that does not exist.
#[test]
fn every_nvidia_hardware_set_is_reachable_from_its_declared_compute_capability() {
    let mut checked = Vec::new();
    for (dir, hw) in hardware_sets() {
        if hw.get("vendor").and_then(|v| v.as_str()) != Some("nvidia") {
            continue;
        }
        let cc_text = hw
            .get("compute_capability")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                panic!(
                    "kernels/{dir}/HARDWARE.toml is an NVIDIA target with no \
                     compute_capability; atlas_core::arch::target_hint cannot be \
                     held to a value that is not declared"
                )
            });
        let cc = parse_cc(cc_text, &dir);
        assert_eq!(
            atlas_core::arch::target_hint(cc),
            Some(dir.as_str()),
            "kernels/{dir}/HARDWARE.toml declares compute_capability {cc_text:?}, \
             but atlas_core::arch::target_hint({cc:?}) does not name {dir:?} — \
             add the arm in crates/atlas-core/src/arch.rs"
        );
        checked.push(dir);
    }
    assert!(
        checked.len() >= 2,
        "found {checked:?} — fewer NVIDIA hardware sets than the tree has, so the \
         walk is broken rather than the hints"
    );
}
/// The other direction, for the arms this repo can check: a hint must name a
/// directory that exists. A `match` arm pointing at a removed or renamed
/// hardware set is a rebuild instruction that cannot be followed.
#[test]
fn every_target_hint_names_a_hardware_set_that_exists() {
    // The CCs the match arms answer for. Listed rather than enumerated
    // because the function is a `match` over a sparse space, not a table —
    // and a CC listed here that stops being hinted fails the test above.
    for cc in [(9, 0), (10, 0), (12, 1)] {
        let hw = atlas_core::arch::target_hint(cc)
            .unwrap_or_else(|| panic!("target_hint({cc:?}) answered None"));
        assert!(
            kernels_root().join(hw).join("HARDWARE.toml").is_file(),
            "target_hint({cc:?}) says to build ATLAS_TARGET_HW={hw}, but \
             kernels/{hw}/HARDWARE.toml does not exist"
        );
    }
}
