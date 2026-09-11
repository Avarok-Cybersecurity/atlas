// SPDX-License-Identifier: AGPL-3.0-only

//! Is one kernel directory a faithful symlink mirror of another, apart from
//! the files it DECLARES it overrides?
//!
//! Shared by `tests/inherited_targets.rs`, which asks it of every hardware set
//! that inherits `kernels/gb10`'s kernels rather than shipping its own. Lives
//! under `tests/support/` so cargo does not also pick it up as a test target
//! of its own — a subdirectory is not auto-discovered, a `tests/*.rs` is — and
//! its self-tests therefore run exactly once, inside the binary that includes
//! it.
//!
//! # Overrides
//!
//! Maintainer rule, 2026-09-11 (tbraun96): "symlinks are fine provided the
//! pointed-to gb10 file is not edited when iterating on Hopper; Hopper-tuned
//! kernels must be real files under `kernels/hopper/`." So the mirror is no
//! longer a pure bijection: a target may carry sources the oracle does not
//! have, for hardware whose tuning has nothing to do with GB10.
//!
//! They are DECLARED, in `kernels/<hw>/HARDWARE.toml` `[kernels] overrides`,
//! and not merely tolerated. An undeclared regular file in a mirror is a
//! silent fork of a shared kernel — the exact defect this checker exists to
//! catch — and nothing on disk distinguishes the two cases.

use std::collections::BTreeSet;
use std::path::Path;

/// Every problem with one mirrored directory, as human-readable lines.
///
/// A mirror is correct when:
///
/// * every entry the `origin` has, it has, as a RELATIVE symlink that resolves
///   — an entry the origin has and the mirror does not is a kernel that
///   vanishes from this hardware's build, and a link that does not resolve is
///   a compile error deferred to whoever next owns a GPU;
/// * every entry it has that the origin does not is DECLARED in `overrides` —
///   an undeclared one is a fork nobody announced;
/// * every declared override is present and resolves, and does NOT resolve to
///   the origin's file of the same name — an "override" that is a link back to
///   the file it claims to replace is a declaration that lies. (It may be a
///   regular file, which is what the rule asks for on the tuned target, or a
///   relative symlink to ANOTHER target's override, which is how a second
///   architecture shares one tuned source without making a duplicate copy.)
///
/// Returns the empty vec when the mirror is sound. Kept as a function rather
/// than inline assertions so the self-tests below can drive it against
/// deliberately broken trees — a checker that has never failed has never been
/// tested.
pub fn mirror_faults(mirror: &Path, origin: &Path, overrides: &BTreeSet<String>) -> Vec<String> {
    let names = |dir: &Path| -> BTreeSet<String> {
        std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect()
    };
    let (mirrored, originals) = (names(mirror), names(origin));
    let mut faults = Vec::new();
    for missing in originals.difference(&mirrored) {
        faults.push(format!("{missing}: present in origin, absent from mirror"));
    }
    for extra in mirrored.difference(&originals) {
        if !overrides.contains(extra) {
            faults.push(format!(
                "{extra}: present in mirror, absent from origin, and not \
                 declared in HARDWARE.toml [kernels] overrides"
            ));
        }
    }
    for declared in overrides {
        if !mirrored.contains(declared) {
            faults.push(format!(
                "{declared}: declared as an override, absent from the mirror"
            ));
        }
    }
    // Every mirrored entry is checked, including ones the origin no longer
    // has: a link is most likely to dangle precisely when its target was
    // renamed away, and reporting only "absent from origin" would hide that
    // the tree in hand does not compile.
    for name in &mirrored {
        let path = mirror.join(name);
        // THE ORACLE for both arms. `read_link` reports the stored text;
        // `exists()` follows it. A dangling link is invisible to `read_dir`
        // and to git, and shows up first as an nvcc "No such file or
        // directory".
        if overrides.contains(name) {
            if !path.exists() {
                faults.push(format!("{name}: declared override does not resolve"));
            } else if std::fs::canonicalize(&path).ok()
                == std::fs::canonicalize(origin.join(name)).ok()
            {
                faults.push(format!(
                    "{name}: declared as an override but resolves to the origin's \
                     own file — it overrides nothing"
                ));
            }
            if let Ok(link) = std::fs::read_link(&path)
                && link.is_absolute()
            {
                faults.push(format!("{name}: absolute symlink {}", link.display()));
            }
            continue;
        }
        let Ok(link) = std::fs::read_link(&path) else {
            faults.push(format!("{name}: a regular file, not a symlink to origin"));
            continue;
        };
        if link.is_absolute() {
            faults.push(format!("{name}: absolute symlink {}", link.display()));
        }
        if !path.exists() {
            faults.push(format!("{name}: dangling symlink -> {}", link.display()));
        }
    }
    faults.sort();
    faults
}

// ── the oracle, tested ──

fn no_overrides() -> BTreeSet<String> {
    BTreeSet::new()
}

fn tmp_root(line: u32) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("atlas-hopper-mirror-{}-{line}", std::process::id()))
}

/// A dangling symlink must FAIL `mirror_faults`. Without this the whole
/// symlink half of this file is an assertion that has never once been observed
/// to fire, and `assert!(faults.is_empty())` passes just as happily against a
/// checker that always returns nothing.
#[test]
fn the_dangling_symlink_check_can_fail() {
    let root = tmp_root(line!());
    let origin = root.join("origin");
    let mirror = root.join("mirror");
    std::fs::create_dir_all(&origin).unwrap();
    std::fs::create_dir_all(&mirror).unwrap();
    std::fs::write(origin.join("present.cu"), "// kernel\n").unwrap();
    std::fs::write(origin.join("gone.cu"), "// kernel\n").unwrap();
    std::os::unix::fs::symlink("../origin/present.cu", mirror.join("present.cu")).unwrap();
    // The failure this exists to catch: a link whose target was renamed or
    // removed. `read_dir` still lists it and git still stores it.
    std::os::unix::fs::symlink("../origin/gone.cu", mirror.join("gone.cu")).unwrap();
    std::fs::remove_file(origin.join("gone.cu")).unwrap();

    let faults = mirror_faults(&mirror, &origin, &no_overrides());
    let _ = std::fs::remove_dir_all(&root);

    assert_eq!(
        faults,
        vec![
            "gone.cu: dangling symlink -> ../origin/gone.cu".to_string(),
            "gone.cu: present in mirror, absent from origin, and not declared in \
             HARDWARE.toml [kernels] overrides"
                .to_string(),
        ],
        "the mirror check did not report a dangling symlink"
    );
}

/// …and the other two faults it is responsible for: a missing entry and a
/// regular file where a symlink belongs (a silent fork of a shared kernel).
#[test]
fn the_mirror_check_reports_missing_entries_and_regular_files() {
    let root = tmp_root(line!());
    let origin = root.join("origin");
    let mirror = root.join("mirror");
    std::fs::create_dir_all(&origin).unwrap();
    std::fs::create_dir_all(&mirror).unwrap();
    std::fs::write(origin.join("forked.cu"), "// origin\n").unwrap();
    std::fs::write(origin.join("missing.cu"), "// origin\n").unwrap();
    std::fs::write(mirror.join("forked.cu"), "// a copy, not a link\n").unwrap();

    let faults = mirror_faults(&mirror, &origin, &no_overrides());
    let _ = std::fs::remove_dir_all(&root);

    assert_eq!(
        faults,
        vec![
            "forked.cu: a regular file, not a symlink to origin".to_string(),
            "missing.cu: present in origin, absent from mirror".to_string(),
        ]
    );
}

/// A DECLARED override is accepted as a real file, and as a relative symlink
/// to another target's copy of it — the two shapes `kernels/hopper` and
/// `kernels/b200` use.
#[test]
fn a_declared_override_is_accepted_as_a_file_or_a_link_elsewhere() {
    let root = tmp_root(line!());
    let origin = root.join("origin");
    let mirror = root.join("mirror");
    let tuned = root.join("tuned");
    std::fs::create_dir_all(&origin).unwrap();
    std::fs::create_dir_all(&mirror).unwrap();
    std::fs::create_dir_all(&tuned).unwrap();
    std::fs::write(origin.join("shared.cu"), "// origin\n").unwrap();
    std::os::unix::fs::symlink("../origin/shared.cu", mirror.join("shared.cu")).unwrap();
    // A real file the origin does not have — the Hopper shape.
    std::fs::write(mirror.join("tuned_here.cu"), "// hopper-tuned\n").unwrap();
    // A link to another target's tuned file — the B200 shape.
    std::fs::write(tuned.join("tuned_there.cu"), "// hopper-tuned\n").unwrap();
    std::os::unix::fs::symlink("../tuned/tuned_there.cu", mirror.join("tuned_there.cu")).unwrap();

    let declared: BTreeSet<String> = ["tuned_here.cu", "tuned_there.cu"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let faults = mirror_faults(&mirror, &origin, &declared);
    let _ = std::fs::remove_dir_all(&root);
    assert!(faults.is_empty(), "{faults:?}");
}

/// The three ways a declaration can be wrong, each of which would otherwise
/// let a broken tree pass: declared and missing, declared and dangling, and
/// declared but pointing back at the file it claims to replace.
#[test]
fn a_declaration_that_does_not_hold_is_a_fault() {
    let root = tmp_root(line!());
    let origin = root.join("origin");
    let mirror = root.join("mirror");
    std::fs::create_dir_all(&origin).unwrap();
    std::fs::create_dir_all(&mirror).unwrap();
    std::fs::write(origin.join("shared.cu"), "// origin\n").unwrap();
    // Declared, and a link straight back to the origin: it overrides nothing,
    // and a reader of HARDWARE.toml would believe this target was tuned.
    std::os::unix::fs::symlink("../origin/shared.cu", mirror.join("shared.cu")).unwrap();
    // Declared and dangling.
    std::os::unix::fs::symlink("../origin/vanished.cu", mirror.join("vanished.cu")).unwrap();

    let declared: BTreeSet<String> = ["shared.cu", "vanished.cu", "never_written.cu"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let faults = mirror_faults(&mirror, &origin, &declared);
    let _ = std::fs::remove_dir_all(&root);

    assert_eq!(
        faults,
        vec![
            "never_written.cu: declared as an override, absent from the mirror".to_string(),
            "shared.cu: declared as an override but resolves to the origin's own \
             file — it overrides nothing"
                .to_string(),
            "vanished.cu: declared override does not resolve".to_string(),
        ]
    );
}
