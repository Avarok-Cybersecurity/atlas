// SPDX-License-Identifier: AGPL-3.0-only

//! The first paid H200 checkpoint must resolve to the 3.8 target while reusing
//! 3.6's sources. CPU-only oracles: the real GB10 declarations and resolver.

#[path = "support/inherited.rs"]
mod inherited;

use atlas_kernels::ModelTypeMatch;
use atlas_kernels::resolve::{ResolveCandidate, resolve_target};
use inherited::{gb10_dir, hardware_toml, hw_dir};

fn declaration(hw: &str, model: &str) -> toml::Value {
    let path = hw_dir(hw).join(model).join("MODEL.toml");
    toml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

#[test]
fn hopper_dense_27b_preserves_checkpoint_behavior_and_the_source_redirect() {
    for model in ["qwen3.6-27b", "qwen3.8-27b"] {
        let gb10 = declaration("gb10", model);
        let hopper = declaration("hopper", model);
        for table in ["model", "model_types", "sampling", "behavior", "dflash"] {
            assert_eq!(hopper.get(table), gb10.get(table), "{model}: {table}");
        }
    }
    let alias = declaration("hopper", "qwen3.8-27b");
    assert_eq!(
        alias["model"]["kernel_source"].as_str(),
        Some("qwen3.6-27b")
    );
    assert_eq!(
        hardware_toml("hopper")["hardware"]["arch"].as_str(),
        Some("sm_90a")
    );
}

#[test]
fn the_h200_fp8_checkpoint_selects_38_behavior_without_an_order_default() {
    let models = [
        declaration("hopper", "qwen3.6-27b"),
        declaration("hopper", "qwen3.8-27b"),
    ];
    let matches: Vec<Vec<ModelTypeMatch>> = models
        .iter()
        .map(|m| {
            m["model_types"]
                .as_array()
                .unwrap()
                .iter()
                .map(|mt| ModelTypeMatch {
                    model_type: Box::leak(
                        mt["model_type"]
                            .as_str()
                            .unwrap()
                            .to_owned()
                            .into_boxed_str(),
                    ),
                    hidden_size: mt
                        .get("hidden_size")
                        .and_then(|s| s.as_integer())
                        .map(|s| s as usize),
                })
                .collect()
        })
        .collect();
    let needles: Vec<Vec<&str>> = models
        .iter()
        .map(|m| {
            m["model"]["match_names"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap())
                .collect()
        })
        .collect();
    let candidates: Vec<_> = models
        .iter()
        .enumerate()
        .map(|(i, m)| ResolveCandidate {
            name: m["model"]["name"].as_str().unwrap(),
            type_matches: &matches[i],
            match_names: &needles[i],
        })
        .collect();
    // Known-bad input first: the identical numeric architecture must not
    // silently choose the alphabetically earlier 3.6 sampling policy.
    assert!(resolve_target(&candidates, "qwen3_5", 5120, &["/model"]).is_err());
    let index = resolve_target(&candidates, "qwen3_5", 5120, &["Qwen/Qwen3.8-27B-FP8"])
        .unwrap()
        .expect("first paid checkpoint must resolve");
    assert_eq!(candidates[index].name, "qwen3.8-27b");
}

#[test]
fn dense_warp_fp4_is_explicitly_absent_only_on_hopper() {
    let source =
        std::fs::read_to_string(gb10_dir().join("qwen3.6-27b/nvfp4/w4a4_gemm.cu")).unwrap();
    let (_, guarded) = source
        .split_once("#ifndef ATLAS_NO_WARP_BLOCKSCALE_MMA")
        .unwrap();
    assert!(guarded.contains("void w4a4_gemm("));
    assert!(
        guarded
            .trim_end()
            .ends_with("#endif  // ATLAS_NO_WARP_BLOCKSCALE_MMA")
    );
    for model in ["qwen3.6-27b", "qwen3.8-27b"] {
        let hopper = declaration("hopper", model);
        let reason = hopper["expected_absent"]["w4a4"]["w4a4_gemm"]
            .as_str()
            .unwrap();
        assert!(reason.contains("sm_90a") && reason.contains("mma with block scale"));
        let gb10 = declaration("gb10", model);
        assert!(gb10["expected_absent"]["w4a4"].get("w4a4_gemm").is_none());
    }
}
