// SPDX-License-Identifier: AGPL-3.0-only

//! A resolved symbol must not disguise the vendor's unsupported-ISA trap.
//! The first real H100 dense-27B request hit quantize_impl.cuh's NO_DEVICE_CODE
//! despite a green symbol audit. Guard the entire optional module using the
//! vendor's capability predicate, before Rust resolves any of its handles.

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels")
}

fn source() -> String {
    std::fs::read_to_string(root().join("gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu")).unwrap()
}

fn exports(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| line.starts_with("extern \"C\" __global__ void "))
        .map(|line| {
            let name = line.split("atlas_nvfp4_").nth(1).unwrap();
            format!("atlas_nvfp4_{}", name.split('(').next().unwrap())
        })
        .collect()
}

#[test]
fn every_mmq_export_is_inside_the_vendor_capability_guard() {
    let text = source();
    let marker = "#if defined(BLACKWELL_MMA_AVAILABLE) // Atlas optional module";
    let (before, inside) = text.split_once(marker).expect(
        "Hopper resolves trap-only MMQ symbols: guard exports before handle-based selection",
    );
    assert!(exports(before).is_empty());
    assert!(
        inside
            .trim_end()
            .ends_with("#endif // Atlas optional module")
    );
    assert_eq!(exports(inside).len(), 13);
    // This is the SAME macro used by both quantize_mmq_nvfp4_worker and MMA.
    // A generic 'Blackwell or newer' check would incorrectly admit B200.
    let vendor =
        std::fs::read_to_string(root().join("gb10/qwen3.6-27b/nvfp4/q4k_vendor/common.cuh"))
            .unwrap();
    assert!(vendor.contains("#define GGML_CUDA_CC_BLACKWELL       1200"));
    assert!(vendor.contains("#define GGML_CUDA_CC_RUBIN           1300"));
    assert!(
        vendor.contains(
            "__CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN"
        )
    );
}

#[test]
fn gb10_dense_targets_keep_every_supported_mmq_export() {
    let functions = exports(&source());
    assert_eq!(functions.len(), 13);
    // GB10 is SM 12.1: BLACKWELL_MMA_AVAILABLE is defined, so every entry
    // point above compiles to real device code and none of them may be
    // declared absent. A target whose arch does not define the predicate
    // records each exclusion under `expected_absent.nvfp4_mmq` instead.
    for model in ["qwen3.6-27b", "qwen3.8-27b"] {
        let path = root().join("gb10").join(model).join("MODEL.toml");
        let doc: toml::Value = toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let absent = doc.get("expected_absent").and_then(|t| t.get("nvfp4_mmq"));
        assert!(absent.is_none(), "GB10 must retain the supported MMQ path");
    }
}
