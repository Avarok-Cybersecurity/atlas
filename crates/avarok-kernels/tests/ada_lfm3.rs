// SPDX-License-Identifier: AGPL-3.0-only

//! The Ada (sm_89) LFM3-1.7B target (port sprint lane L37): its declaration
//! must carry the checkpoint facts Liquid publishes (layer split, sampling
//! presets, parser), compile an EXISTING Ada kernel set through
//! `kernel_source` without adding a single symlink (rule S1 of
//! `scripts/check_cross_hardware.py`), say what the weights' license is, and
//! resolve its checkpoint without moving any Holo or Qwen target. CPU-only
//! oracles: the real declarations on disk and the real resolver, in the
//! pattern of `tests/ada_27b.rs`.

#[path = "support/inherited.rs"]
mod inherited;

use avarok_kernels::ModelTypeMatch;
use avarok_kernels::resolve::{ResolveCandidate, resolve_target};
use inherited::{hw_dir, kernels_root};

const MODEL: &str = "lfm3-1.7b";
/// The existing Ada directory whose kernel set this target compiles.
const SOURCE: &str = "holo-3.1-0.8b";

fn declaration(hw: &str, model: &str) -> toml::Value {
    let path = hw_dir(hw).join(model).join("MODEL.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()))
}

fn manifest_text() -> String {
    std::fs::read_to_string(hw_dir("ada").join(MODEL).join("MODEL.toml")).unwrap()
}

/// Model directories under `kernels/<hw>`, by the rule `build.rs` uses to
/// expand `AVAROK_TARGET_MODEL=*`: a subdirectory that carries a MODEL.toml.
fn model_dirs(hw: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(hw_dir(hw))
        .unwrap_or_else(|e| panic!("kernels/{hw}: {e}"))
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            e.path().join("MODEL.toml").exists().then_some(name)
        })
        .collect();
    names.sort();
    names
}

#[test]
fn the_declaration_carries_the_published_checkpoint_facts() {
    let m = declaration("ada", MODEL);
    let model = &m["model"];
    assert_eq!(model["name"].as_str(), Some(MODEL));
    assert_eq!(model["hf_id"].as_str(), Some("LiquidAI/LFM3-1.7B"));
    assert_eq!(model["kernel_source"].as_str(), Some(SOURCE));
    // 24 layers: 13 conv + 6 global + 5 sliding.
    let n = |k: &str| {
        model[k]
            .as_integer()
            .unwrap_or_else(|| panic!("[model] {k}"))
    };
    assert_eq!(n("layers_total"), 24);
    assert_eq!(
        n("layers_linear_attention") + n("layers_full_attention") + n("layers_sliding_attention"),
        24
    );
    assert_eq!(
        (
            n("layers_linear_attention"),
            n("layers_full_attention"),
            n("layers_sliding_attention")
        ),
        (13, 6, 5)
    );
    assert_eq!(
        (n("hidden_dim"), n("head_dim"), n("q_heads"), n("kv_heads")),
        (2048, 128, 16, 4)
    );
    assert_eq!(
        (
            n("intermediate_size"),
            n("sliding_window"),
            n("conv_kernel")
        ),
        (7680, 256, 7)
    );
    assert_eq!(n("vocab_size"), 128_000);
    assert_eq!(n("max_position_embeddings"), 262_144);
}

/// Liquid's card: temperature 0.1, top_k 50, repetition_penalty 1.1 for chat,
/// tools, agents and coding; there is no top_p. All four categories carry it.
#[test]
fn every_sampling_category_is_liquids_general_preset() {
    let m = declaration("ada", MODEL);
    for cat in ["thinking_text", "thinking_coding", "non_thinking", "tools"] {
        let s = &m["sampling"][cat];
        assert_eq!(s["temperature"].as_float(), Some(0.1), "{cat}");
        assert_eq!(s["top_k"].as_integer(), Some(50), "{cat}");
        assert_eq!(s["repetition_penalty"].as_float(), Some(1.1), "{cat}");
        assert_eq!(s["top_p"].as_float(), Some(1.0), "{cat}");
    }
}

#[test]
fn behavior_selects_the_lfm3_parser_and_default_thinking() {
    let m = declaration("ada", MODEL);
    let b = &m["behavior"];
    assert_eq!(b["tool_call_parser"].as_str(), Some("lfm3"));
    assert_eq!(b["thinking_default"].as_bool(), Some(true));
    assert_eq!(
        b["default_num_drafts"].as_integer(),
        Some(0),
        "no MTP head, no drafter"
    );
    // Grammar-constrained tool decoding stays off until lfm3_qual.py has A/B'd it.
    assert_eq!(b["disable_tool_grammar"].as_bool(), Some(true));
}

/// The weights' license is the user's to read before serving: the manifest
/// names it, with the threshold, in prose and in a key.
#[test]
fn the_manifest_states_the_weights_license_and_its_threshold() {
    let text = manifest_text();
    let head: String = text.lines().take(30).collect::<Vec<_>>().join("\n");
    // Comment lines wrap, so join them before looking for a phrase.
    let hay = head.replace("\n# ", " ");
    for needle in [
        "LFM Open License v1.0",
        "USD 10,000,000",
        "not licensed",
        "contains no LFM3 weights",
        "docs/porting/lfm3.md",
    ] {
        assert!(
            hay.contains(needle),
            "the head of MODEL.toml lacks {needle:?}:\n{head}"
        );
    }
    let m = declaration("ada", MODEL);
    let license = m["model"]["license"].as_str().expect("[model] license");
    assert!(
        license.contains("LFM Open License v1.0") && license.contains("USD 10M"),
        "{license}"
    );
    assert!(license.contains("not bundled"), "{license}");
}

/// The redirect names an Ada target that owns its sources (no chain), and
/// that target's quant tree is the one the wildcard build iterates.
#[test]
fn the_redirect_names_an_existing_ada_kernel_set() {
    let source = declaration("ada", SOURCE);
    assert!(
        source["model"].get("kernel_source").is_none(),
        "{SOURCE} itself redirects: chains are refused by build.rs"
    );
    let tree = hw_dir("ada").join(SOURCE).join("nvfp4");
    assert!(tree.is_dir(), "{} missing", tree.display());
    // Nothing LFM3 needs is FP4: the curated Ada set has no W4A4 or NVFP4-MMQ source.
    for stem in ["w4a4_gemm.cu", "nvfp4_mmq.cu"] {
        assert!(
            !tree.join(stem).exists(),
            "kernels/ada/{SOURCE}/nvfp4/{stem}"
        );
    }
}

/// Rule S1: no new cross-hardware symlink, and no quant directory of its own
/// (that would fork the redirect target). The directory holds MODEL.toml only.
#[test]
fn the_directory_is_one_real_file() {
    let dir = hw_dir("ada").join(MODEL);
    let mut entries: Vec<String> = Vec::new();
    for e in std::fs::read_dir(&dir).unwrap().flatten() {
        let path = e.path();
        assert!(
            std::fs::symlink_metadata(&path)
                .map(|m| !m.file_type().is_symlink())
                .unwrap_or(false),
            "{} is a symlink (rule S1)",
            path.display()
        );
        entries.push(e.file_name().to_string_lossy().to_string());
    }
    assert_eq!(
        entries,
        vec!["MODEL.toml".to_string()],
        "kernels/ada/{MODEL}"
    );
}

/// Only Ada ships LFM3: every other target's tree is byte-for-byte what it was.
#[test]
fn no_other_hardware_declares_the_target() {
    for hw in ["gb10", "hopper", "b200", "b300"] {
        let p = kernels_root().join(hw).join(MODEL);
        assert!(
            !p.exists(),
            "kernels/{hw}/{MODEL} must not exist in this sprint"
        );
    }
}

/// The kernel set is holo-3.1-0.8b's, so the lookups that may fail are its
/// lookups: the carried tables must equal the source target's until the GPU
/// pass harvests this directory.
#[test]
fn expected_absent_is_the_source_targets_until_harvested() {
    let ada = declaration("ada", MODEL);
    let source = declaration("ada", SOURCE);
    let names = |v: &toml::Value| -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (table, entries) in v["expected_absent"].as_table().unwrap() {
            for kernel in entries.as_table().unwrap().keys() {
                out.push((table.clone(), kernel.clone()));
            }
        }
        out.sort();
        out
    };
    assert_eq!(names(&ada), names(&source));
    let head: String = manifest_text()
        .lines()
        .take(6)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(head.contains("--check-kernels"), "{head}");
    assert!(head.contains("NOT yet harvested"), "{head}");
}

/// Every Ada target as `resolve.rs` sees it.
fn ada_candidates() -> (Vec<String>, Vec<Vec<ModelTypeMatch>>) {
    let names = model_dirs("ada");
    let mut matches = Vec::new();
    for name in &names {
        let m = declaration("ada", name);
        matches.push(
            m.get("model_types")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
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
                .unwrap_or_default(),
        );
    }
    (names, matches)
}

#[test]
fn the_lfm3_checkpoint_resolves_here_and_no_other_target_moves() {
    let (names, matches) = ada_candidates();
    assert!(names.iter().any(|n| n == MODEL), "{names:?}");
    let no_needles: Vec<&str> = Vec::new();
    let candidates: Vec<ResolveCandidate> = names
        .iter()
        .enumerate()
        .map(|(i, n)| ResolveCandidate {
            name: n.as_str(),
            type_matches: &matches[i],
            match_names: &no_needles,
        })
        .collect();
    let pick = |model_type: &str, hidden: usize, refs: &[&str]| -> Option<&str> {
        resolve_target(&candidates, model_type, hidden, refs)
            .unwrap()
            .map(|i| candidates[i].name)
    };
    assert_eq!(pick("lfm3", 2048, &["LiquidAI/LFM3-1.7B"]), Some(MODEL));
    assert_eq!(pick("lfm3", 2048, &["/model"]), Some(MODEL));
    // The VL and d1 checkpoints are a different model_type with a vision tower.
    assert_eq!(pick("lfm3_vl", 2048, &["/model"]), None);
    // A different LFM3 width is not claimed by this target.
    assert_eq!(pick("lfm3", 1024, &["/model"]), None);
    // The Holo targets resolve exactly as before the LFM3 target existed.
    assert_eq!(pick("qwen3_5", 2560, &["/model"]), Some("holo-3.1-4b"));
    assert_eq!(pick("qwen3_5", 1024, &["/model"]), Some("holo-3.1-0.8b"));
}
