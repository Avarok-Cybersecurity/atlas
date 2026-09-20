// SPDX-License-Identifier: AGPL-3.0-only

//! Data contract for the local `.cu` discovery used by build.rs::find_cu_files.
//! A module override alone cannot select an external source file.
use std::path::Path;

#[test]
fn each_gb10_k3_quant_exposes_the_e8m0_expert_source() {
    let hw = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10");
    let expected = hw.join("deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu");
    for quant in ["bf16", "mxfp4", "nvfp4"] {
        let dir = hw.join("kimi-k3").join(quant);
        let source = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.file_name().unwrap() == "moe_w4a16_grouped_gemm.cu")
            .unwrap_or_else(|| panic!("{quant}: E8M0 source absent from build discovery"));
        assert_eq!(
            source.canonicalize().unwrap(),
            expected.canonicalize().unwrap()
        );
        assert!(
            std::fs::read_to_string(&source)
                .unwrap()
                .contains("moe_w4a16_grouped_gemm_ptrtable_e8m0")
        );
        let manifest: toml::Value =
            toml::from_str(&std::fs::read_to_string(dir.join("KERNEL.toml")).unwrap()).unwrap();
        assert_eq!(
            manifest["modules"]["moe_w4a16_grouped_gemm"].as_str(),
            Some("moe_w4a16")
        );
        assert!(
            manifest
                .get("build")
                .and_then(|build| build.get("extra_cu"))
                .is_none(),
            "unused external-source declaration"
        );
    }
}
