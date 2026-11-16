// SPDX-License-Identifier: AGPL-3.0-only

//! The Ada (sm_89) Qwen3.8-27B target: it must keep the
//! checkpoint behavior gb10 declares for 3.8, compile an EXISTING Ada kernel
//! set through `kernel_source` without adding a single symlink (rule S1 of
//! `scripts/check_cross_hardware.py`), and resolve its checkpoint without
//! moving either Holo target. CPU-only oracles: the real declarations on disk
//! and the real resolver, in the pattern of `tests/hopper_27b.rs`.

#[path = "support/inherited.rs"]
mod inherited;

use avarok_kernels::ModelTypeMatch;
use avarok_kernels::resolve::{ResolveCandidate, resolve_target};
use inherited::{gb10_dir, hardware_toml, hw_dir};

const MODEL: &str = "qwen3.8-27b";
/// The existing Ada directory whose kernel set this target compiles.
const SOURCE: &str = "holo-3.1-4b";

fn declaration(hw: &str, model: &str) -> toml::Value {
    let path = hw_dir(hw).join(model).join("MODEL.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("bad TOML in {}: {e}", path.display()))
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
fn the_ada_27b_keeps_gb10s_checkpoint_behavior() {
    let gb10 = declaration("gb10", MODEL);
    let ada = declaration("ada", MODEL);
    for table in ["model_types", "sampling", "behavior", "dflash"] {
        assert_eq!(ada.get(table), gb10.get(table), "{MODEL}: [{table}]");
    }
    // [model]: every architecture fact is gb10's; only the redirect differs.
    let (ga, aa) = (
        gb10["model"].as_table().unwrap(),
        ada["model"].as_table().unwrap(),
    );
    let mut keys: Vec<&String> = ga.keys().chain(aa.keys()).collect();
    keys.sort();
    keys.dedup();
    for key in keys {
        if key == "kernel_source" {
            continue;
        }
        assert_eq!(aa.get(key), ga.get(key), "{MODEL}: [model] {key}");
    }
    assert_eq!(gb10["model"]["kernel_source"].as_str(), Some("qwen3.6-27b"));
    assert_eq!(ada["model"]["kernel_source"].as_str(), Some(SOURCE));
    assert_eq!(
        hardware_toml("ada")["hardware"]["arch"].as_str(),
        Some("sm_89")
    );
}

/// The redirect names an Ada target that owns its sources (no chain), and that
/// target's quant tree is the one the build will compile for the 27B.
#[test]
fn the_redirect_names_an_existing_ada_kernel_set() {
    let source = declaration("ada", SOURCE);
    assert!(
        source["model"].get("kernel_source").is_none(),
        "{SOURCE} itself redirects: chains are refused by build.rs"
    );
    let tree = hw_dir("ada").join(SOURCE).join("nvfp4");
    assert!(tree.is_dir(), "{} missing", tree.display());
    // The 27B needs nothing gb10's qwen3.6-27b tree carries that sm_89 cannot
    // assemble: the curated Ada set has no W4A4 or NVFP4-MMQ source.
    for stem in ["w4a4_gemm.cu", "nvfp4_mmq.cu"] {
        assert!(
            !tree.join(stem).exists(),
            "kernels/ada/{SOURCE}/nvfp4/{stem}: an FP4-MMA source in the Ada set"
        );
    }
}

/// Rule S1: no new cross-hardware symlink, and no quant directory of its own
/// (that would fork the redirect target). The directory holds MODEL.toml only.
#[test]
fn the_27b_directory_is_one_real_file() {
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
    assert!(!gb10_dir().join(MODEL).join("nvfp4").exists());
}

/// The kernel set is holo-3.1-4b's, so the lookups that may fail are its
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
    let head: String = std::fs::read_to_string(hw_dir("ada").join(MODEL).join("MODEL.toml"))
        .unwrap()
        .lines()
        .take(6)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(head.contains("--check-kernels"), "{head}");
    assert!(head.contains("NOT yet harvested"), "{head}");
}

/// Every Ada target as `resolve.rs` sees it.
fn ada_candidates() -> (Vec<String>, Vec<Vec<ModelTypeMatch>>, Vec<Vec<String>>) {
    let names = model_dirs("ada");
    let mut matches = Vec::new();
    let mut needles = Vec::new();
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
        needles.push(
            m["model"]
                .get("match_names")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().map(|s| s.as_str().unwrap().to_string()).collect())
                .unwrap_or_default(),
        );
    }
    (names, matches, needles)
}

#[test]
fn the_27b_checkpoint_resolves_here_and_the_holo_targets_do_not_move() {
    let (names, matches, needles) = ada_candidates();
    let needle_refs: Vec<Vec<&str>> = needles
        .iter()
        .map(|n| n.iter().map(String::as_str).collect())
        .collect();
    let candidates: Vec<ResolveCandidate> = names
        .iter()
        .enumerate()
        .map(|(i, n)| ResolveCandidate {
            name: n.as_str(),
            type_matches: &matches[i],
            match_names: &needle_refs[i],
        })
        .collect();
    let pick = |hidden: usize, refs: &[&str]| -> Option<&str> {
        resolve_target(&candidates, "qwen3_5", hidden, refs)
            .unwrap()
            .map(|i| candidates[i].name)
    };
    // The GGUF the plan downloads, and an anonymous directory: one declarer of
    // (qwen3_5, 5120) on Ada, so no needle is needed to pick it.
    assert_eq!(
        pick(5120, &["/home/u/models/Qwen3.8-27B-UD-IQ4_XS.gguf"]),
        Some(MODEL)
    );
    assert_eq!(pick(5120, &["/model"]), Some(MODEL));
    // The two Holo targets resolve exactly as before the 27B existed.
    assert_eq!(pick(2560, &["/model"]), Some("holo-3.1-4b"));
    assert_eq!(pick(1024, &["/model"]), Some("holo-3.1-0.8b"));
}

/// The 27B's pool sizing on a 16 GB card is keyed on this row (the consumer
/// verify-slot fit in `ssm_reserve::consumer_fit`); gb10 and the datacenter
/// targets must keep it off so their sizing stays byte-identical.
#[test]
fn only_ada_declares_the_consumer_memory_fit() {
    let fit = |hw: &str| {
        hardware_toml(hw)
            .get("defaults")
            .and_then(|d| d.get("memory_fit_consumer"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    };
    assert!(fit("ada"));
    for hw in ["gb10", "hopper", "b200", "b300"] {
        assert!(!fit(hw), "{hw}");
    }
}
