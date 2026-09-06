// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn repo_root() -> PathBuf {
    // crates/atlas-plugin -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atlas-taxon-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Build a minimal fake tree so the unit tests do not depend on the real
/// kernel layout, which changes as models are added.
fn fixture(name: &str) -> PathBuf {
    let root = tmp(name);
    let hw = root.join("kernels/gb10");
    std::fs::create_dir_all(hw.join("common")).unwrap();
    std::fs::create_dir_all(hw.join("modelA/nvfp4")).unwrap();
    std::fs::create_dir_all(hw.join("modelB/nvfp4")).unwrap();
    std::fs::write(
        hw.join("HARDWARE.toml"),
        "[hardware]\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(hw.join("modelA/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(hw.join("modelB/MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(hw.join("common/shared.cu"), "__global__ void s() {}\n").unwrap();
    std::fs::write(hw.join("common/other.cu"), "__global__ void o() {}\n").unwrap();
    std::fs::write(hw.join("common/helper.cuh"), "#define H 1\n").unwrap();
    std::fs::write(
        hw.join("modelA/nvfp4/shared.cu"),
        "__global__ void s2() {}\n",
    )
    .unwrap();
    root
}

// ---------------------------------------------------------------------------
// Agreement with the real tree
// ---------------------------------------------------------------------------

/// ★ The load-bearing invariant: every target the walk finds must resolve
/// sources. `sources()` returning `None` means the gate falls back to
/// "affected" — correct but expensive — and returning an EMPTY set would be a
/// fail-open, since every empty set hashes alike.
#[test]
fn every_real_target_resolves_a_nonempty_source_set() {
    let root = repo_root();
    let targets = walk(&root);
    assert_eq!(
        targets.iter().map(ToString::to_string).collect::<Vec<_>>(),
        [
            // The B200 P0 five, sorting ahead of gb10's. Same symlinked
            // sources as hopper's, and the same reason they are not redundant.
            "b200/deepseek-v4-flash/nvfp4",
            "b200/nemotron-3-nano-30b-a3b/nvfp4",
            "b200/nemotron-super-120b-a12b/nvfp4",
            "b200/qwen3-next-80b-a3b/nvfp4",
            "b200/qwen3.6-35b-a3b/nvfp4",
            "gb10/deepseek-v4-flash/nvfp4",
            "gb10/gemma-4-26b-a4b/nvfp4",
            "gb10/gemma-4-31b/nvfp4",
            "gb10/holo-3.1-0.8b/nvfp4",
            "gb10/holo-3.1-35b-a3b/nvfp4",
            "gb10/holo-3.1-4b/nvfp4",
            "gb10/laguna-s-2.1/nvfp4",
            "gb10/longcat-flash-lite/nvfp4",
            "gb10/minimax-m2-229b/nvfp4",
            "gb10/mistral-small-4/nvfp4",
            "gb10/nemotron-3-nano-30b-a3b/nvfp4",
            "gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4",
            "gb10/nemotron-super-120b-a12b/nvfp4",
            "gb10/nllb-200-3.3b/bf16",
            "gb10/ornith-1.0-9b/nvfp4",
            "gb10/qwen3-next-80b-a3b/nvfp4",
            "gb10/qwen3-vl-30b-a3b/nvfp4",
            "gb10/qwen3.5-122b-a10b/nvfp4",
            "gb10/qwen3.5-27b/nvfp4",
            "gb10/qwen3.5-35b-a3b/nvfp4",
            "gb10/qwen3.5-397b-a17b/nvfp4",
            "gb10/qwen3.6-27b/nvfp4",
            "gb10/qwen3.6-35b-a3b/nvfp4",
            "gb10/qwen3.8-27b/nvfp4",
            "gb10/qwen3.8-flash-next/nvfp4",
            "gb10/step3p7-flash/nvfp4",
            // The seven Hopper targets. Their sources are symlinks into gb10's, and
            // `sources()` follows them — so these targets resolve non-empty
            // source sets that OVERLAP gb10's, which is correct: a change to a
            // shared kernel affects both hardware sets and must not be
            // skippable for either.
            "hopper/deepseek-v4-flash/nvfp4",
            "hopper/nemotron-3-nano-30b-a3b/nvfp4",
            "hopper/nemotron-super-120b-a12b/nvfp4",
            "hopper/qwen3-next-80b-a3b/nvfp4",
            "hopper/qwen3.6-27b/nvfp4",
            "hopper/qwen3.6-35b-a3b/nvfp4",
            "hopper/qwen3.8-27b/nvfp4",
            "metal/nllb-200-3.3b/bf16",
            "metal/qwen3-5-4b-vlm-mlx-int8/mlx_int8",
            "strix/qwen3.6-27b/nvfp4",
            "strix/qwen3.6-35b-a3b/nvfp4",
            "strix-hip/qwen3.6-27b/nvfp4",
            "strix-hip/qwen3.6-35b-a3b/nvfp4",
        ]
    );
    for t in &targets {
        let srcs = sources(&root, t)
            .unwrap_or_else(|| panic!("{t}: sources() returned None — vendor table stale?"));
        assert!(!srcs.is_empty(), "{t}: empty source set");
    }
}

/// Every hardware dir in the tree must have a known vendor. A new backend
/// lands as `None` and its targets go to the expensive path silently; this
/// test makes that visible at the moment it is added.
#[test]
fn every_hardware_vendor_is_known_to_the_source_extension_table() {
    let root = repo_root();
    for t in walk(&root) {
        let v = vendor(&root, &t.hardware)
            .unwrap_or_else(|| panic!("{}: HARDWARE.toml has no vendor", t.hardware));
        assert!(
            source_ext(&v).is_some(),
            "hardware {} declares vendor {v:?}, which source_ext() does not know — \
             teach it the extension or its targets can never be skipped",
            t.hardware
        );
    }
}

// ---------------------------------------------------------------------------
// Shadowing
// ---------------------------------------------------------------------------

#[test]
fn a_model_file_shadows_the_common_file_with_the_same_stem() {
    let root = fixture("shadow");
    let t = Target {
        hardware: "gb10".into(),
        model: "modelA".into(),
        quant: "nvfp4".into(),
    };
    let srcs = sources(&root, &t).unwrap();
    assert_eq!(
        srcs,
        [
            root.join("kernels/gb10/common/other.cu"),
            root.join("kernels/gb10/modelA/nvfp4/shared.cu"),
        ]
    );
}

/// Headers are not sources — they enter the hash only by being included. If
/// `sources()` returned them, an unused header would invalidate a target that
/// never sees it.
#[test]
fn headers_are_not_in_the_source_set() {
    let root = fixture("headers");
    let t = Target {
        hardware: "gb10".into(),
        model: "modelB".into(),
        quant: "nvfp4".into(),
    };
    assert_eq!(
        sources(&root, &t).unwrap(),
        [
            root.join("kernels/gb10/common/other.cu"),
            root.join("kernels/gb10/common/shared.cu"),
        ]
    );
}

#[test]
fn a_redirect_uses_the_source_owners_kernels_but_the_targets_model_config() {
    let root = fixture("redirect");
    let hw = root.join("kernels/gb10");
    std::fs::create_dir_all(hw.join("modelC")).unwrap();
    std::fs::write(
        hw.join("modelC/MODEL.toml"),
        "[model]\nkernel_source = \"modelA\"\n",
    )
    .unwrap();
    std::fs::write(hw.join("common/KERNEL.toml"), "[build]\n").unwrap();
    std::fs::write(hw.join("modelA/nvfp4/KERNEL.toml"), "[build]\n").unwrap();

    let redirected = Target {
        hardware: "gb10".into(),
        model: "modelC".into(),
        quant: "nvfp4".into(),
    };
    assert!(walk(&root).contains(&redirected));
    assert_eq!(
        sources(&root, &redirected).unwrap(),
        [
            hw.join("common/other.cu"),
            hw.join("modelA/nvfp4/shared.cu"),
        ]
    );
    assert_eq!(
        configs(&root, &redirected),
        [
            hw.join("HARDWARE.toml"),
            hw.join("common/KERNEL.toml"),
            hw.join("modelC/MODEL.toml"),
            hw.join("modelA/nvfp4/KERNEL.toml"),
        ]
    );
}

#[test]
fn a_source_owner_change_affects_its_redirected_consumers() {
    let root = fixture("redirect-affected");
    let hw = root.join("kernels/gb10");
    std::fs::create_dir_all(hw.join("modelC")).unwrap();
    std::fs::write(
        hw.join("modelC/MODEL.toml"),
        "[model]\nkernel_source = \"modelA\"\n",
    )
    .unwrap();

    assert_eq!(
        affected(&root, &["kernels/gb10/modelA/nvfp4/shared.cu".to_string()]),
        [
            Target {
                hardware: "gb10".into(),
                model: "modelA".into(),
                quant: "nvfp4".into(),
            },
            Target {
                hardware: "gb10".into(),
                model: "modelC".into(),
                quant: "nvfp4".into(),
            },
        ]
        .into_iter()
        .collect()
    );
}

/// ★ Fail-closed: no resolvable sources is `None`, never `Some(vec![])`.
#[test]
fn an_unresolvable_target_is_none_not_empty() {
    let root = fixture("empty");
    let t = Target {
        hardware: "gb10".into(),
        model: "modelA".into(),
        quant: "does-not-exist".into(),
    };
    std::fs::remove_dir_all(root.join("kernels/gb10/common")).unwrap();
    assert!(
        sources(&root, &t).is_none(),
        "an empty resolution must be None — every empty set hashes alike"
    );
}

#[test]
fn an_unknown_vendor_resolves_to_none() {
    let root = fixture("vendor");
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\nvendor = \"quantum-abacus\"\n",
    )
    .unwrap();
    let t = Target {
        hardware: "gb10".into(),
        model: "modelA".into(),
        quant: "nvfp4".into(),
    };
    assert!(sources(&root, &t).is_none());
}

// ---------------------------------------------------------------------------
// Path -> node
// ---------------------------------------------------------------------------

#[test]
fn paths_map_to_the_right_nodes() {
    assert_eq!(hardware_of("kernels/gb10/common/x.cu"), Some("gb10"));
    assert_eq!(hardware_of("kernels/strix/qwen/nvfp4/x.cu"), Some("strix"));
    assert_eq!(hardware_of("crates/atlas-plugin/src/lib.rs"), None);

    assert_eq!(
        model_of("kernels/gb10/qwen3.6-27b/nvfp4/x.cu"),
        Some(("gb10", "qwen3.6-27b"))
    );
    assert_eq!(
        model_of("kernels/gb10/common/x.cu"),
        None,
        "a shared kernel belongs to no single model"
    );
    assert_eq!(
        model_of("kernels/gb10/HARDWARE.toml"),
        None,
        "a hardware-level file is not under a model"
    );
    assert_eq!(
        model_of("kernels/gb10/qwen3.6-27b/MODEL.toml"),
        Some(("gb10", "qwen3.6-27b"))
    );
}

/// A directory whose name merely starts with `kernels` is not the kernel tree.
#[test]
fn lookalike_paths_are_not_kernel_paths() {
    assert_eq!(hardware_of("kernels-old/gb10/common/x.cu"), None);
    assert_eq!(hardware_of("docs/kernels/gb10/x.cu"), None);
}

// ---------------------------------------------------------------------------
// Affected sets
// ---------------------------------------------------------------------------

#[test]
fn a_common_change_affects_every_target_on_that_hardware() {
    let root = fixture("affected-common");
    let hit = affected(&root, &["kernels/gb10/common/shared.cu".to_string()]);
    assert_eq!(
        hit,
        [
            Target {
                hardware: "gb10".into(),
                model: "modelA".into(),
                quant: "nvfp4".into(),
            },
            Target {
                hardware: "gb10".into(),
                model: "modelB".into(),
                quant: "nvfp4".into(),
            },
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn a_model_change_affects_only_that_model() {
    let root = fixture("affected-model");
    let hit = affected(&root, &["kernels/gb10/modelA/nvfp4/shared.cu".to_string()]);
    assert_eq!(
        hit,
        [Target {
            hardware: "gb10".into(),
            model: "modelA".into(),
            quant: "nvfp4".into(),
        }]
        .into_iter()
        .collect()
    );
}

#[test]
fn a_non_kernel_change_affects_no_target_here() {
    let root = fixture("affected-host");
    assert!(
        affected(
            &root,
            &["crates/atlas-plugin/src/gate/check.rs".to_string()]
        )
        .is_empty(),
        "host code is handled by the path boundary, not the taxonomy"
    );
}

// ---------------------------------------------------------------------------
// Span checks
// ---------------------------------------------------------------------------

#[test]
fn spans_are_reported_per_node_level() {
    let changed = vec![
        "kernels/gb10/qwen3.6-27b/nvfp4/a.cu".to_string(),
        "kernels/strix/qwen3.6-27b/nvfp4/a.cu".to_string(),
    ];
    assert_eq!(
        hardware_span(&changed),
        ["gb10", "strix"].map(str::to_string).into()
    );
    assert_eq!(
        model_span(&changed),
        [
            ("gb10".to_string(), "qwen3.6-27b".to_string()),
            ("strix".to_string(), "qwen3.6-27b".to_string()),
        ]
        .into()
    );

    let one_hw = vec![
        "kernels/gb10/modelA/nvfp4/a.cu".to_string(),
        "kernels/gb10/modelB/nvfp4/a.cu".to_string(),
    ];
    assert_eq!(hardware_span(&one_hw), ["gb10".to_string()].into());
    assert_eq!(
        model_span(&one_hw),
        [
            ("gb10".to_string(), "modelA".to_string()),
            ("gb10".to_string(), "modelB".to_string()),
        ]
        .into()
    );
}
