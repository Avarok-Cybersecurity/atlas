// SPDX-License-Identifier: AGPL-3.0-only

//! The denominator for the Apple Metal 1:1 map, resolved by the build's own
//! resolver.
//!
//! A hand count of `__global__ void NAME` in `kernels/gb10` gives 514, and that
//! number is wrong in both directions: it counts macro PLACEHOLDERS
//! (`KERNEL_NAME`, `AVAROK_PREFILL_ENTRY`) as if they were kernels, and it
//! misses every name produced by `#define KERNEL_NAME` + `#include` and by the
//! instantiation macros (`AVAROK_WYN_INSTANTIATE`, `AVAROK_MOE_BATCHM_ENTRY`).
//! `kernel_shadow_detector.rs`'s header records what that silence already cost
//! once: twenty-one `common/*.cu` files resolved to the EMPTY SET, so every
//! shadow comparison against them reported "drops nothing", and the 27B's four
//! missing multi-sequence GDN decode kernels stayed hidden until 2026-07-26.
//!
//! So this test resolves both sides through `build_shadow::entry_points` — the
//! same code `build.rs` uses — and prints the census. The 1:1 manifest is built
//! from ITS output, never from a grep.
//!
//! It asserts no total. A pinned total is a merge conflict on every kernel
//! added, and the number is not the property worth protecting; what matters is
//! that neither side resolved to nothing and that every dispatch site resolves.

// The module is shared with `kernel_shadow_detector.rs`, which uses both of its
// public fns; this test uses only `entry_points`, and the workspace denies
// `dead_code`. Allowing it here is narrower than splitting the resolver, and
// splitting it would put the rule in two places -- which is the defect
// `check_kernel_shadows.py`'s own header warns about.
#[allow(dead_code)]
#[path = "../build_shadow.rs"]
mod build_shadow;

use build_shadow::entry_points;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/avarok-kernels -> repo root")
        .to_path_buf()
}

/// Every `ext` source under `dir`, with symlinks skipped.
///
/// gb10 carries 91 symlinks: following them would attribute one kernel to two
/// paths and inflate the denominator. The manifest is keyed by (source, entry),
/// so a duplicated path is a duplicated row.
fn sources(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(md) = std::fs::symlink_metadata(&p) else {
                continue;
            };
            if md.file_type().is_symlink() {
                continue;
            }
            if md.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|s| s.to_str()) == Some(ext) {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// (entry point -> the source paths that declare it), resolved.
fn census(dir: &Path, ext: &str) -> BTreeMap<String, Vec<String>> {
    let root = repo_root();
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for src in sources(dir, ext) {
        let rel = src
            .strip_prefix(&root)
            .unwrap_or(&src)
            .to_string_lossy()
            .into_owned();
        for name in entry_points(&src) {
            map.entry(name).or_default().push(rel.clone());
        }
    }
    map
}

/// Kernel lookups written as string literals in Rust.
///
/// Matches the two-string forms the backends expose: `.kernel("m", "f")`,
/// `try_kernel(gpu, "m", "f")`, `try_target_kernel(gpu, "m", "f")`.
fn literal_lookups(root: &Path) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    for rs in sources(&root.join("crates"), "rs") {
        let Ok(text) = std::fs::read_to_string(&rs) else {
            continue;
        };
        for marker in ["kernel(", "try_kernel(", "try_target_kernel("] {
            let mut from = 0usize;
            while let Some(i) = text[from..].find(marker) {
                let at = from + i + marker.len();
                from = at;
                // `at` is a char boundary (it came from `find` of an ASCII
                // marker) but `at + 400` need not be: these files are full of
                // box-drawing `─`, and slicing mid-character panics.
                let mut end = (at + 400).min(text.len());
                while end > at && !text.is_char_boundary(end) {
                    end -= 1;
                }
                let tail = &text[at..end];
                let strings: Vec<&str> = tail
                    .split('"')
                    .skip(1)
                    .step_by(2)
                    .take(2)
                    .filter(|s| !s.is_empty() && s.len() < 80)
                    .collect();
                if strings.len() == 2
                    && strings
                        .iter()
                        .all(|s| s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
                {
                    out.insert((strings[0].to_string(), strings[1].to_string()));
                }
            }
        }
    }
    out
}

/// Lookups whose NAME is built at runtime, and therefore invisible to the scan
/// above. Each row claims a family of resolved names.
///
/// This table is the reason the census is a test and not a grep: five
/// production/example sites build a kernel name with `format!`, and a grep for
/// string literals reports them as absent. A family that matches nothing is a
/// dead dispatch path, which is why the empty case is a failure.
const DYNAMIC_FAMILIES: &[(&str, &str)] = &[
    (
        "w4a16_gemv_batch",
        "crates/spark-model/src/layers/w4a16_gemv_tiers.rs",
    ),
    ("gated_delta_rule_wy", "the wyN dispatcher"),
    ("w4a16_gemv_sw_moe_batchm_m", "the MoE batch-M arms"),
];

#[test]
fn census_resolves_both_kernel_trees_and_every_dispatch_site() {
    let root = repo_root();
    let gb10 = census(&root.join("kernels/gb10"), "cu");
    let metal = census(&root.join("kernels/metal"), "metal");

    // ── Refusals. An empty read must never be reported as agreement: the whole
    // point of this file is that a resolver returning nothing once looked
    // exactly like a tree with no drift.
    assert!(
        !gb10.is_empty(),
        "resolved ZERO gb10 entry points. kernels/gb10 is unreadable or the \
         resolver broke. This is not 'no kernels'; it is 'could not look', and \
         a manifest built on it would claim 1:1 coverage of nothing."
    );
    assert!(
        !metal.is_empty(),
        "resolved ZERO Metal entry points from kernels/metal/**/*.metal."
    );

    let dup: Vec<_> = gb10
        .iter()
        .filter(|(_, paths)| paths.len() > 1)
        .map(|(n, paths)| format!("{n} declared in {} files: {paths:?}", paths.len()))
        .collect();

    // ── The dispatch cross-check. A resolved name nothing looks up is a
    // candidate `no_dispatch_site` exception; a lookup that resolves nowhere is
    // a dead dispatch, which is the direction that breaks at runtime.
    let looked_up = literal_lookups(&root);
    let all_names: BTreeSet<&String> = gb10.keys().chain(metal.keys()).collect();
    let unresolved: Vec<_> = looked_up
        .iter()
        .filter(|(_, f)| !all_names.contains(f))
        .map(|(m, f)| format!("{m}::{f}"))
        .collect();

    for (prefix, whose) in DYNAMIC_FAMILIES {
        let n = all_names.iter().filter(|k| k.starts_with(prefix)).count();
        assert!(
            n > 0,
            "the dynamic dispatch family `{prefix}*` ({whose}) matches NO resolved \
             kernel. Either the family is dead or the resolver is not expanding \
             the macros that generate these names — and a name built with \
             format! is invisible to a literal scan, so nothing else would have \
             noticed."
        );
    }

    // Printed, never asserted against a literal: a pinned total is a merge
    // conflict per kernel added, and the manifest consumes this output.
    println!("gb10 entry points resolved : {}", gb10.len());
    println!("metal entry points resolved: {}", metal.len());
    println!(
        "literal (module, kernel) lookups in crates/: {}",
        looked_up.len()
    );
    println!("gb10 names declared in more than one source: {}", dup.len());
    for d in dup.iter().take(20) {
        println!("  multi-source: {d}");
    }
    println!(
        "lookups resolving to no kernel in either tree: {}",
        unresolved.len()
    );
    for u in unresolved.iter().take(40) {
        println!("  unresolved: {u}");
    }
    if let Ok(path) = std::env::var("GITHUB_STEP_SUMMARY") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(path) {
            let _ = writeln!(
                f,
                "### kernel census\n\n| side | resolved |\n|---|---|\n| gb10 | {} |\n| metal | {} |\n| lookups | {} |\n",
                gb10.len(),
                metal.len(),
                looked_up.len()
            );
        }
    }
}
