// SPDX-License-Identifier: AGPL-3.0-only

//! Proofs for Rust files excluded from benchmark invalidation because their
//! only module edge is guarded by `#[cfg(test)]`.

use std::path::{Path, PathBuf};

use super::coverage::{self, BOUNDARY_FILES, REQUIRED, TEST_ONLY_RUST_MODULES, TestOnlyRustModule};
use super::coverage_tests::{any_gate, scratch_repo};
use super::record_covers;
use super::tests::tempdir;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("repo root is two levels above the crate")
        .to_path_buf()
}

fn rust_sources_beneath(directory: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory).expect("source directory is readable") {
        let path = entry.expect("source entry is readable").path();
        if path.is_dir() {
            rust_sources_beneath(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn quoted_literal(line: &str) -> Option<&str> {
    let start = line.find('"')? + 1;
    let end = line[start..].find('"')? + start;
    Some(&line[start..end])
}

fn assert_guarded(root: &Path, module: &TestOnlyRustModule) {
    let module_path = root.join(module.path);
    let parent_path = root.join(module.parent);
    assert!(module_path.is_file(), "{} is not a file", module.path);
    assert!(parent_path.is_file(), "{} is not a file", module.parent);

    let parent = std::fs::read_to_string(&parent_path).expect("parent module is readable");
    let guarded_declaration = match module.declared_path {
        Some(path) => format!("#[cfg(test)]\n#[path = \"{path}\"]\nmod {};", module.name),
        None => format!("#[cfg(test)]\nmod {};", module.name),
    };
    assert!(
        parent.contains(&guarded_declaration),
        "{} must declare `{}` through its registered #[cfg(test)] edge",
        module.parent,
        module.name
    );
    assert_eq!(
        parent.matches(&format!("mod {};", module.name)).count(),
        1,
        "{} must have exactly one declaration for module `{}`",
        module.parent,
        module.name
    );

    // Search every Rust source for an explicit include or path edge resolving
    // to this file. The one allowed edge is the implicit guarded `mod` above.
    let canonical_module = module_path.canonicalize().expect("test module resolves");
    let mut sources = Vec::new();
    rust_sources_beneath(&root.join("crates"), &mut sources);
    for path in sources {
        if path == module_path || path == parent_path {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("sibling Rust source is readable");
        for line in source
            .lines()
            .filter(|line| line.contains("include!(") || line.contains("#[path"))
        {
            let Some(literal) = quoted_literal(line) else {
                continue;
            };
            let Some(base) = path.parent() else {
                continue;
            };
            let direct = base.join(literal).canonicalize().ok();
            let nested = path
                .file_stem()
                .map(|stem| base.join(stem).join(literal))
                .and_then(|candidate| candidate.canonicalize().ok());
            assert!(
                direct.as_ref() != Some(&canonical_module)
                    && nested.as_ref() != Some(&canonical_module),
                "{} creates a second edge to {} with `{}`",
                path.display(),
                module.path,
                line.trim()
            );
        }
    }
}

#[test]
fn registered_test_modules_are_proven_cfg_test_only() {
    let root = repo_root();
    for module in TEST_ONLY_RUST_MODULES {
        assert_guarded(&root, module);
    }
}

#[test]
fn registered_test_modules_do_not_invalidate_gpu_records() {
    for module in TEST_ONLY_RUST_MODULES {
        assert!(
            coverage::invalidated_by([module.path]).is_empty(),
            "{} is absent from release builds and must not reopen GPU gates",
            module.path
        );
    }
}

#[test]
fn test_looking_neighbours_remain_fail_closed() {
    for path in [
        "crates/atlas-core/src/config/tests.rs.bak",
        "crates/atlas-core/src/config/tests/tests_b.rs",
        "crates/atlas-core/src/config/gguf/not_really_tests.rs",
        "crates/atlas-core/src/config/gguf.rs",
        "crates/atlas-core/src/config.rs",
        "crates/atlas-plugin/src/benchmarks/concurrency_tests.rs.bak",
        "crates/atlas-plugin/src/benchmarks/concurrency_tests/helper.rs",
    ] {
        let invalidated = coverage::invalidated_by([path]);
        assert_eq!(
            invalidated.len(),
            REQUIRED.len(),
            "unregistered path {path} escaped the fail-closed boundary: {invalidated:?}"
        );
    }
}

#[test]
fn a_production_change_cannot_hide_beside_an_exempt_test_change() {
    for module in TEST_ONLY_RUST_MODULES {
        let parent_only = coverage::invalidated_by([module.parent]);
        assert!(
            !parent_only.is_empty(),
            "{} is registered as a production parent but invalidates no gate",
            module.parent
        );
        let with_test = coverage::invalidated_by([module.path, module.parent]);
        assert_eq!(
            with_test, parent_only,
            "{} changed the coverage owed by its production parent {}",
            module.path, module.parent
        );
    }
}

#[test]
fn no_test_only_entry_is_a_gate_boundary_file() {
    for module in TEST_ONLY_RUST_MODULES {
        assert!(
            !BOUNDARY_FILES.contains(&module.path),
            "{} cannot be both exempt and verdict-defining",
            module.path
        );
    }
}

#[test]
fn a_record_survives_a_test_only_commit_but_not_its_production_parent() {
    let directory = tempdir::Dir::new();
    let root = directory.path();
    scratch_repo::init(root);
    let recorded = scratch_repo::head(root);

    scratch_repo::commit(
        root,
        TEST_ONLY_RUST_MODULES[0].path,
        "#[test]\nfn assertion_changed() {}\n",
        "change only tests",
    );
    assert!(
        record_covers(root, &scratch_repo::head(root), &recorded, &any_gate()),
        "a record must keep covering a release-equivalent test-only commit"
    );

    scratch_repo::commit(
        root,
        TEST_ONLY_RUST_MODULES[0].parent,
        "pub fn production_changed() {}\n",
        "change production",
    );
    assert!(
        !record_covers(root, &scratch_repo::head(root), &recorded, &any_gate()),
        "the production parent must invalidate the earlier record"
    );
}
