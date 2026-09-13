// SPDX-License-Identifier: AGPL-3.0-only

//! Every cuBLASLt dispatch site reads the field for ITS OWN projection family.
//!
//! WHY a source scan and not nine runtime tests: the property under test is
//! "no dispatch site reads a family that is not its own", which is a statement
//! about the whole crate, not about any one call. A runtime test can only
//! prove the site it drives; the bug this replaces — ONE boolean arming the
//! dense FFN and an unrelated SSM projection together, 10.3 GiB of off-ledger
//! BF16 weight copies, an H100 prefill dead at layer 36 on 2026-09-11 — is
//! exactly the kind that hides in the site nobody wrote a test for. Scanning
//! the sources catches the NEXT one: a new consumer that reaches for
//! `cublas.ffn` from an attention file fails here the day it is written.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Which directory prefix (relative to `crates/spark-model/src`) each family
/// may be read from. A site outside its family's area is the regression.
const FAMILY_HOMES: &[(&str, &[&str])] = &[
    (
        "ffn",
        &[
            "layers/dense_ffn.rs",
            "layers/dense_ffn_w8a8_prefill.rs",
            "layers/moe/",
        ],
    ),
    ("attn", &["layers/qwen3_attention/"]),
    ("ssm", &["layers/qwen3_ssm/"]),
    // `head` has no consumer yet (see `CublasScope::head`); an empty home list
    // means "must not be read anywhere", which is the assertion that will fire
    // the day someone wires it up without updating this table.
    ("head", &[]),
];

fn src_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable source dir") {
        let path = entry.expect("readable dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// `(relative path, contents)` for every `.rs` file under `spark-model/src`,
/// excluding the test files themselves (a doc comment quoting the old spelling
/// must not fail the scan).
fn sources() -> Vec<(String, String)> {
    let root = src_root();
    let mut paths = Vec::new();
    rust_sources(&root, &mut paths);
    paths.sort();
    paths
        .into_iter()
        .filter(|p| {
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            !name.contains("test")
        })
        .map(|p| {
            let rel = p
                .strip_prefix(&root)
                .expect("path under src")
                .to_string_lossy()
                .replace('\\', "/");
            (rel, std::fs::read_to_string(&p).expect("readable source"))
        })
        .collect()
}

/// The pre-2026-09-11 global spelling must be gone. Left anywhere, it is a
/// site that silently kept arming every family.
#[test]
fn the_global_cublas_gemm_boolean_has_no_readers_left() {
    let offenders: Vec<_> = sources()
        .into_iter()
        .filter(|(_, body)| body.contains("dispatch.cublas_gemm"))
        .map(|(rel, _)| rel)
        .collect();
    assert!(
        offenders.is_empty(),
        "these files still read the un-scoped `dispatch.cublas_gemm`: {offenders:?}"
    );
}

#[test]
fn each_cublas_family_is_read_only_from_its_own_projection_area() {
    let sources = sources();
    let mut seen: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for (family, homes) in FAMILY_HOMES {
        let needle = format!("dispatch.cublas.{family}");
        for (rel, body) in &sources {
            if !body.contains(&needle) {
                continue;
            }
            assert!(
                homes.iter().any(|home| rel.starts_with(home)),
                "{rel} reads `{needle}` but is not a {family} projection site \
                 (allowed: {homes:?}) — a family must not be armed from another's file"
            );
            seen.entry(family).or_default().push(rel.clone());
        }
    }
    // Every family with a home must actually have a consumer, or the lever
    // accepts a spelling that does nothing and says nothing.
    for (family, homes) in FAMILY_HOMES {
        if homes.is_empty() {
            assert!(
                !seen.contains_key(family),
                "`{family}` is documented as having no consumer, but {:?} reads it — \
                 wire it into FAMILY_HOMES and CublasScope's doc comment",
                seen.get(family)
            );
        } else {
            assert!(
                seen.contains_key(family),
                "no dispatch site reads `dispatch.cublas.{family}`"
            );
        }
    }
}
