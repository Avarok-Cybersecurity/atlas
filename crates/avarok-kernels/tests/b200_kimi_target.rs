// SPDX-License-Identifier: AGPL-3.0-only

//! B200 K3 registration contract; execution still requires a B200 GPU.
use std::path::Path;

#[test]
fn b200_kimi_compiles_all_quant_dependencies() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/b200");
    let model = root.join("kimi-k3");
    let manifest: toml::Value =
        toml::from_str(&std::fs::read_to_string(model.join("MODEL.toml")).unwrap()).unwrap();
    assert_eq!(manifest["model"]["name"].as_str(), Some("kimi-k3"));
    for quant in ["bf16", "mxfp4", "nvfp4"] {
        let dir = model.join(quant);
        for stem in ["kda_decode", "mla_decode", "moe_w4a16_grouped_gemm"] {
            assert!(dir.join(format!("{stem}.cu")).is_file(), "{quant}/{stem}");
        }
        let source = std::fs::read_to_string(dir.join("moe_w4a16_grouped_gemm.cu")).unwrap();
        assert!(source.contains("moe_w4a16_grouped_gemm_ptrtable_e8m0"));
        let kernel: toml::Value =
            toml::from_str(&std::fs::read_to_string(dir.join("KERNEL.toml")).unwrap()).unwrap();
        assert_eq!(
            kernel["modules"]["moe_w4a16_grouped_gemm"].as_str(),
            Some("moe_w4a16")
        );
    }
    for stem in ["kda_decode", "mla_decode", "moe_w4a16_grouped_gemm"] {
        let source = model.join("bf16").join(format!("{stem}.cu"));
        assert!(
            !source.is_symlink(),
            "B200 model kernels must be independently editable"
        );
    }
}
