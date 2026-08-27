// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn rust_sources_under(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut pending = vec![root.to_path_buf()];
    let mut sources = Vec::new();
    while let Some(path) = pending.pop() {
        if path.is_dir() {
            pending.extend(
                std::fs::read_dir(&path)
                    .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
                    .flatten()
                    .map(|entry| entry.path()),
            );
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            sources.push(path);
        }
    }
    sources.sort();
    sources
}

/// ★ The cross-import precondition behind the per-benchmark exclusions.
///
/// TTFT excludes the BFCL driver on the grounds that one cannot affect the
/// other. Every nested Rust module is included so moving an import one level
/// down cannot silently invalidate that rationale.
#[test]
fn benchmark_drivers_do_not_import_each_other() {
    let probe_root =
        std::env::temp_dir().join(format!("atlas-driver-recursion-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&probe_root);
    let nested = probe_root.join("nested/deeper/probe.rs");
    std::fs::create_dir_all(nested.parent().unwrap()).unwrap();
    std::fs::write(&nested, "// nested source\n").unwrap();
    assert_eq!(
        rust_sources_under(&probe_root),
        [nested],
        "the cross-import scan must reach nested modules"
    );

    let root = repo_root().join("crates/atlas-plugin/src/benchmarks");
    let names = [
        "ttft",
        "bfcl",
        "agentic",
        "contamination",
        "ssm_poison",
        "decode_floor",
        "quick_speed",
        "concurrency",
    ];
    for driver in names {
        let directory = root.join(driver);
        let sources = if directory.is_dir() {
            rust_sources_under(&directory)
        } else {
            let main = root.join(format!("{driver}.rs"));
            assert!(main.exists(), "driver file {} is missing", main.display());
            [
                main,
                root.join(format!("{driver}_tests.rs")),
                root.join(format!("{driver}_verdict.rs")),
            ]
            .into_iter()
            .filter(|path| path.exists())
            .collect()
        };
        assert!(!sources.is_empty(), "{driver} has no Rust source to audit");
        for path in sources {
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
            for other in names.iter().filter(|other| **other != driver) {
                let needle = format!("benchmarks::{other}");
                assert!(
                    !source.contains(&needle),
                    "{} imports {other}; per-driver exclusions assume independence",
                    path.display()
                );
            }
        }
    }
}
