// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the crate root. Split out of `lib.rs` to keep it under the
//! repo's 500-LoC cap; `lib.rs` re-attaches this file with `#[path]`, the
//! same idiom used by `atlas-closure` and `atlas-plugin::gate`.

use super::*;

#[test]
fn all_ptx_modules_non_empty() {
    for (name, blob) in ptx_modules() {
        assert!(
            !blob.is_empty(),
            "PTX module '{name}' is empty — nvcc compilation may have failed"
        );
        // Blobs are `&[u8]` (uniform across backends). For the NVIDIA
        // build under test the bytes are ASCII PTX, so decode and check
        // the `.version` directive; on a non-text backend this lossily
        // decodes to "" and the assert would (correctly) not apply.
        let ptx = std::str::from_utf8(blob).unwrap_or("");
        assert!(
            ptx.contains(".version"),
            "PTX module '{name}' doesn't contain .version directive"
        );
    }
}

// These tests assert that PTX modules were actually compiled into the
// crate at build time. They require nvcc + a real CUDA toolchain — the
// CI host runs with `ATLAS_SKIP_BUILD=1`, which emits an empty stub
// registry by design (so `cargo check` / `cargo clippy` / `cargo test`
// can run on hosts without a GPU). Mark them `#[ignore]` so default
// `cargo test` is green; they're still exercised on a developer
// machine via `cargo test -p atlas-kernels -- --ignored` after a
// real PTX build.

#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
fn module_count_matches_cu_files() {
    let count = ptx_modules().len();
    assert!(count >= 31, "Expected at least 31 PTX modules, got {count}");
}

#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
fn available_targets_non_empty() {
    let targets = available_targets();
    assert!(!targets.is_empty(), "No kernel targets available");
    assert!(
        targets.iter().any(|t| t.target.quant == "nvfp4"),
        "Expected at least one NVFP4 target"
    );
}

#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
fn all_targets_have_modules() {
    for t in available_targets() {
        assert!(
            t.modules.len() >= 31,
            "Target {} has only {} modules (expected >= 31)",
            t.target,
            t.modules.len()
        );
    }
}

#[test]
#[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
fn ptx_for_model_lookup() {
    let found = ptx_for_model("qwen3-next-80b");
    assert!(
        found.is_some(),
        "ptx_for_model('qwen3-next-80b') should find the default target"
    );
}
