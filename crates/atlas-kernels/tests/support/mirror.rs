// SPDX-License-Identifier: AGPL-3.0-only

//! Is one kernel directory a faithful symlink mirror of another?
//!
//! Shared by `tests/inherited_targets.rs`, which asks it of every hardware set
//! that inherits `kernels/gb10`'s kernels rather than shipping its own. Lives
//! under `tests/support/` so cargo does not also pick it up as a test target
//! of its own — a subdirectory is not auto-discovered, a `tests/*.rs` is — and
//! its self-tests therefore run exactly once, inside the binary that includes
//! it.

use std::collections::BTreeSet;
use std::path::Path;

/// Every problem with one mirrored directory, as human-readable lines.
///
/// A mirror is correct when it is a bijection onto `origin` made of relative
/// symlinks that resolve: an entry `origin` has and `mirror` does not is a
/// kernel that vanishes from this hardware's build, an entry `mirror` has and
/// `origin` does not is a fork nobody declared, and a link that does not
/// resolve is a compile error deferred to whoever next owns a GPU.
///
/// `owned` is the DECLARED exception: entry names this hardware set tunes for
/// itself, where a real file REPLACES the link (maintainer rule, 2026-09-11 —
/// a Hopper-tuned kernel overrides its gb10 namesake and must not edit it).
/// Each name is required to be present and required to be a regular file, and
/// is permitted — only there — to have no counterpart in `origin`, which is
/// how a tuned kernel brings its own header. Everything about the exception is
/// checked: a name listed here that is still a symlink is a dead declaration,
/// and a name NOT listed here that has become a regular file is still the
/// silent fork this checker exists to catch.
///
/// Returns the empty vec when the mirror is sound. Kept as a function rather
/// than inline assertions so `the_dangling_symlink_check_can_fail` can drive
/// it against a deliberately broken tree — a checker that has never failed has
/// never been tested.
pub fn mirror_faults(mirror: &Path, origin: &Path, owned: &[&str]) -> Vec<String> {
    let names = |dir: &Path| -> BTreeSet<String> {
        std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect()
    };
    let (mirrored, originals) = (names(mirror), names(origin));
    let owned: BTreeSet<String> = owned.iter().map(|s| (*s).to_string()).collect();
    let mut faults = Vec::new();
    for missing in originals.difference(&mirrored) {
        faults.push(format!("{missing}: present in origin, absent from mirror"));
    }
    for extra in mirrored.difference(&originals) {
        if owned.contains(extra) {
            continue;
        }
        faults.push(format!("{extra}: present in mirror, absent from origin"));
    }
    for declared in owned.difference(&mirrored) {
        faults.push(format!(
            "{declared}: declared hardware-owned, absent from mirror"
        ));
    }
    // Every mirrored entry is link-checked, including ones the origin no
    // longer has: a link is most likely to dangle precisely when its target
    // was renamed away, and reporting only "absent from origin" would hide
    // that the tree in hand does not compile.
    for name in &mirrored {
        let path = mirror.join(name);
        let is_owned = owned.contains(name);
        let Ok(link) = std::fs::read_link(&path) else {
            if !is_owned {
                faults.push(format!("{name}: a regular file, not a symlink to origin"));
            }
            continue;
        };
        if is_owned {
            faults.push(format!(
                "{name}: declared hardware-owned but still a symlink -> {}",
                link.display()
            ));
            continue;
        }
        if link.is_absolute() {
            faults.push(format!("{name}: absolute symlink {}", link.display()));
        }
        // THE ORACLE. `read_link` reports the stored text; `exists()` follows
        // it. A dangling link is invisible to `read_dir` and to git, and
        // shows up first as an nvcc "No such file or directory".
        if !path.exists() {
            faults.push(format!("{name}: dangling symlink -> {}", link.display()));
        }
    }
    faults.sort();
    faults
}

// ── the oracle, tested ──

/// A dangling symlink must FAIL `mirror_faults`. Without this the whole
/// symlink half of this file is an assertion that has never once been observed
/// to fire, and `assert!(faults.is_empty())` passes just as happily against a
/// checker that always returns nothing.
#[test]
fn the_dangling_symlink_check_can_fail() {
    let root = std::env::temp_dir().join(format!(
        "atlas-hopper-mirror-{}-{}",
        std::process::id(),
        line!()
    ));
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

    let faults = mirror_faults(&mirror, &origin, &[]);
    let _ = std::fs::remove_dir_all(&root);

    assert_eq!(
        faults,
        vec![
            "gone.cu: dangling symlink -> ../origin/gone.cu".to_string(),
            "gone.cu: present in mirror, absent from origin".to_string(),
        ],
        "the mirror check did not report a dangling symlink"
    );
}

/// …and the other two faults it is responsible for: a missing entry and a
/// regular file where a symlink belongs (a silent fork of a shared kernel).
#[test]
fn the_mirror_check_reports_missing_entries_and_regular_files() {
    let root = std::env::temp_dir().join(format!(
        "atlas-hopper-mirror-{}-{}",
        std::process::id(),
        line!()
    ));
    let origin = root.join("origin");
    let mirror = root.join("mirror");
    std::fs::create_dir_all(&origin).unwrap();
    std::fs::create_dir_all(&mirror).unwrap();
    std::fs::write(origin.join("forked.cu"), "// origin\n").unwrap();
    std::fs::write(origin.join("missing.cu"), "// origin\n").unwrap();
    std::fs::write(mirror.join("forked.cu"), "// a copy, not a link\n").unwrap();

    let faults = mirror_faults(&mirror, &origin, &[]);
    let _ = std::fs::remove_dir_all(&root);

    assert_eq!(
        faults,
        vec![
            "forked.cu: a regular file, not a symlink to origin".to_string(),
            "missing.cu: present in origin, absent from mirror".to_string(),
        ]
    );
}

/// The `owned` exception, in BOTH directions. A declared hardware-owned entry
/// is allowed to be a regular file and allowed to have no origin counterpart;
/// a declared entry that is still a symlink, one that is missing outright, and
/// an UNDECLARED regular file are all still faults. Without this the allowance
/// would be an `if` nobody has watched take either branch, which is how a
/// checker quietly stops checking.
#[test]
fn the_hardware_owned_exception_is_narrow() {
    let root = std::env::temp_dir().join(format!(
        "atlas-hopper-mirror-{}-{}",
        std::process::id(),
        line!()
    ));
    let origin = root.join("origin");
    let mirror = root.join("mirror");
    std::fs::create_dir_all(&origin).unwrap();
    std::fs::create_dir_all(&mirror).unwrap();
    for name in ["tuned.cu", "linked.cu", "stale.cu"] {
        std::fs::write(origin.join(name), "// origin\n").unwrap();
    }
    // Declared and tuned: a real file, plus a hardware-only header.
    std::fs::write(mirror.join("tuned.cu"), "// hardware-tuned\n").unwrap();
    std::fs::write(mirror.join("tuned.cuh"), "// hardware-only header\n").unwrap();
    // Not declared: still an ordinary mirrored link.
    std::os::unix::fs::symlink("../origin/linked.cu", mirror.join("linked.cu")).unwrap();
    // Declared but never actually forked — a dead declaration.
    std::os::unix::fs::symlink("../origin/stale.cu", mirror.join("stale.cu")).unwrap();
    // Declared and not there at all.
    let owned = ["tuned.cu", "tuned.cuh", "stale.cu", "absent.cu"];

    let clean = mirror_faults(&mirror, &origin, &owned[..2]);
    let faults = mirror_faults(&mirror, &origin, &owned);
    // The same tree with NOTHING declared: every allowance turns back into a fault.
    let undeclared = mirror_faults(&mirror, &origin, &[]);
    let _ = std::fs::remove_dir_all(&root);

    assert!(
        clean.is_empty(),
        "a declared real file and a declared hardware-only header must both pass: {clean:?}"
    );
    assert_eq!(
        faults,
        vec![
            "absent.cu: declared hardware-owned, absent from mirror".to_string(),
            "stale.cu: declared hardware-owned but still a symlink -> ../origin/stale.cu"
                .to_string(),
        ]
    );
    assert_eq!(
        undeclared,
        vec![
            "tuned.cu: a regular file, not a symlink to origin".to_string(),
            "tuned.cuh: a regular file, not a symlink to origin".to_string(),
            "tuned.cuh: present in mirror, absent from origin".to_string(),
        ],
        "an undeclared regular file is still the silent fork this checker catches"
    );
}
