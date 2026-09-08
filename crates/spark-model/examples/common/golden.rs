// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime loader for the large GLM-5.3 reference goldens.
//!
//! These goldens are DERIVED ARTEFACTS: every one is reproduced byte-for-byte by the
//! `gen_*.py` generator that sits beside it, run against HF transformers 5.16.1 and the
//! published checkpoint. They are megabytes of dense float arrays, so they are not tracked
//! in git -- committing a reproducible multi-megabyte blob costs every clone forever and
//! GitHub will not render it.
//!
//! The small `kda_golden.json` toy fixture IS tracked: it is the deterministic CI gate and
//! it is 36 KB.
//!
//! These microtests need a GPU and, for several of them, the real checkpoint, so they never
//! run in CI regardless. Generate what you need, then run the example.

use std::path::PathBuf;

/// Workspace root, derived from this crate's manifest directory.
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("crates/spark-model sits two levels below the workspace root")
        .to_path_buf()
}

/// Read a golden, or explain exactly how to regenerate it.
///
/// `rel` is relative to the workspace root; `generator` is the script beside it.
pub fn load(rel: &str, generator: &str) -> String {
    let path = workspace_root().join(rel);
    match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => panic!(
            "\n\
             missing reference golden: {rel}\n\
             ({e})\n\n\
             This golden is a derived artefact and is deliberately not tracked in git.\n\
             Regenerate it with the generator beside it:\n\n    \
             python3 {dir}/{generator}\n\n\
             It needs HF transformers 5.16.1 and the GLM-5.3-Flash checkpoint; see that\n\
             script's header for the exact fixture geometry it reproduces.\n",
            rel = rel,
            e = e,
            dir = std::path::Path::new(rel).parent().unwrap().display(),
            generator = generator,
        ),
    }
}
