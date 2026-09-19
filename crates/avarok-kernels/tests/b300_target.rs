// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side B300 bring-up contract. This proves build inputs, not CUDA execution.
//! In particular, the Kimi expert source must be discoverable in every quant
//! directory: a manifest alias alone does not compile the dependency.

#[path = "../build_arch.rs"]
mod build_arch;

use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels")
}

fn toml_at(path: &Path) -> toml::Value {
    toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn b300_build_identity_preserves_arch_specific_preflight() {
    let hw = toml_at(&root().join("b300/HARDWARE.toml"));
    let arch = hw["hardware"]["arch"].as_str().unwrap();
    assert_eq!(arch, "sm_103a");
    assert_eq!(
        build_arch::target_arch_fields(arch),
        ("sm_103".into(), "sm_103a")
    );
    assert_eq!(hw["hardware"]["compute_capability"].as_str(), Some("10.3"));
    assert_eq!(hw["hardware"]["sm_count"].as_integer(), Some(148));
    assert_eq!(hw["hardware"]["memory_gb"].as_integer(), Some(288));
    assert_eq!(avarok_core::arch::target_hint((10, 3)), Some("b300"));
    assert!(avarok_core::arch::ptx_arch_runs_on_device(arch, (10, 3)).is_ok());
    for other in [(9, 0), (10, 0), (12, 1)] {
        assert!(avarok_core::arch::ptx_arch_runs_on_device(arch, other).is_err());
    }
    assert!(
        hw["build"]["extra_nvcc_flags"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_str() == Some("-DAVAROK_NO_WARP_BLOCKSCALE_MMA"))
    );
}

#[test]
fn b300_sources_never_resolve_into_another_hardware_tree() {
    let target = std::fs::canonicalize(root().join("b300")).unwrap();
    fn walk(dir: &Path, target: &Path) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            let resolved = std::fs::canonicalize(&path).unwrap();
            assert!(
                resolved.starts_with(target),
                "cross-hardware input: {}",
                path.display()
            );
            if path.is_dir() {
                walk(&path, target);
            }
        }
    }
    walk(&target, &target);
    for name in ["KERNEL.toml", "mx_block_scale.cuh", "dense_gemm_bf16.cu"] {
        let path = target.join("common").join(name);
        assert!(path.is_file() && !path.is_symlink(), "{}", path.display());
    }
}

#[test]
fn every_kimi_quant_resolves_the_real_e8m0_kernel_dependency() {
    for quant in ["bf16", "mxfp4", "nvfp4"] {
        let dir = root().join("b300/kimi-k3").join(quant);
        let manifest = toml_at(&dir.join("KERNEL.toml"));
        // The compiler discovers .cu files in the quant directory. An
        // unconsumed extra_cu TOML key is not proof of a compiled dependency.
        let resolved = std::fs::canonicalize(dir.join("moe_w4a16_grouped_gemm.cu")).unwrap();
        assert_eq!(
            resolved,
            std::fs::canonicalize(
                root().join("b300/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu")
            )
            .unwrap()
        );
        let source = std::fs::read_to_string(resolved).unwrap();
        assert!(source.contains("moe_w4a16_grouped_gemm_ptrtable_e8m0"));
        assert_eq!(
            manifest["modules"]["moe_w4a16_grouped_gemm"].as_str(),
            Some("moe_w4a16")
        );
    }
    let models: Vec<_> = std::fs::read_dir(root().join("b300"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("MODEL.toml").is_file())
        .collect();
    assert_eq!(
        models.len(),
        1,
        "initial target must not claim other models"
    );
    assert_eq!(models[0].file_name().unwrap(), "kimi-k3");
    let target = toml_at(&models[0].join("MODEL.toml"));
    assert!(
        target["model_types"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["model_type"].as_str() == Some("kimi_k3")
                && row["hidden_size"].as_integer() == Some(7168))
    );
    for quant in ["bf16", "mxfp4", "nvfp4"] {
        for name in ["kda_decode.cu", "mla_decode.cu"] {
            assert!(models[0].join(quant).join(name).is_file());
        }
    }
}
