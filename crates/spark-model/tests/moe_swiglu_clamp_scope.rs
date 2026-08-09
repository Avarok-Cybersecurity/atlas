// SPDX-License-Identifier: AGPL-3.0-only

//! Keeps the SwiGLU clamp inside the models that ask for it.
//!
//! `swiglu_limit` is a per-checkpoint config value: DeepSeek-V4-Flash declares
//! 10.0, Step-3.7-Flash declares a per-LAYER array, GPT-OSS (not ours) declares
//! 7.0, and every Qwen, Gemma, Nemotron, Mistral, MiniMax and Holo checkpoint on
//! the fleet declares nothing at all. Their reference implementations clamp
//! nothing either — `Qwen3_5MLP.forward` is a bare `act_fn(gate) * up`.
//!
//! It nonetheless spent from #186 to wave 55 hardcoded in
//! `kernels/gb10/common/moe_silu_mul.cu`, which is not a DeepSeek kernel: it is
//! the SiLU activation for every dense model's decode and K-verify FFN, every
//! MoE model's grouped prefill, and the MTP and DFlash draft heads. Instrumented
//! on Qwen3.6-27B it bound over 100,000 times across a 20-sample BFCL draw, with
//! `up` reaching -21.78 against a limit of 10 — so it was reshaping activations
//! on twenty checkpoints, not sitting dormant as a safety net.
//!
//! A constant in `common/` reaches the whole fleet, and nothing about the name
//! `SWIGLU_LIMIT` says so at the point of editing. This test says so instead.

use std::path::{Path, PathBuf};

/// Model kernel directories permitted to define a SwiGLU clamp, because their
/// checkpoint's `config.json` declares `swiglu_limit` / `swiglu_limits`. Adding
/// a name here should mean you have read that checkpoint's config, not that you
/// wanted the test to pass.
const DECLARES_A_SWIGLU_LIMIT: &[&str] = &["deepseek-v4-flash", "step3p7-flash"];

/// Kernels whose clamp is known-inconsistent and deliberately left alone. See
/// the comment block at the clamp in `moe_shared_expert_fused.cu`: resolving it
/// moves numbers for DeepSeek-V4, which has no checkpoint on any box, and for
/// the MoE families behind a separate accuracy gate. Recorded, not fixed.
const KNOWN_INCONSISTENT: &[&str] = &["moe_shared_expert_fused.cu"];

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/spark-model is two levels below the workspace root")
        .join("kernels")
}

fn files_defining_a_clamp(root: &Path) -> Vec<PathBuf> {
    let mut hits = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // Symlinked backends (strix, strix-hip) point into gb10; following
            // them would report the same file three times.
            if path.is_dir() && !path.is_symlink() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "cu" || e == "cuh")
                && !path.is_symlink()
                && std::fs::read_to_string(&path).is_ok_and(|t| t.contains("SWIGLU_LIMIT ="))
            {
                hits.push(path);
            }
        }
    }
    hits.sort();
    hits
}

/// The clamp may live in a model directory whose checkpoint declares a limit,
/// or in a kernel explicitly recorded as known-inconsistent. Anywhere else —
/// `common/` above all — it silently reaches models that never asked for it.
#[test]
fn the_swiglu_clamp_stays_in_models_that_declare_one() {
    let root = kernels_root();
    let mut stray = Vec::new();
    for path in files_defining_a_clamp(&root) {
        let rel = path.strip_prefix(&root).unwrap_or(path.as_path());
        let owned_by_a_declaring_model = rel
            .components()
            .any(|c| DECLARES_A_SWIGLU_LIMIT.contains(&c.as_os_str().to_string_lossy().as_ref()));
        let recorded = path
            .file_name()
            .is_some_and(|n| KNOWN_INCONSISTENT.contains(&n.to_string_lossy().as_ref()));
        if !owned_by_a_declaring_model && !recorded {
            stray.push(rel.display().to_string());
        }
    }
    assert!(
        stray.is_empty(),
        "SWIGLU_LIMIT is a per-checkpoint config value, and these files apply it \
         to every model that compiles them: {stray:?}. Put it in the shadow \
         directory of the model whose config.json declares it, or add that model \
         to DECLARES_A_SWIGLU_LIMIT once you have checked the config."
    );
}

/// The one that actually regressed. Called out separately so the failure names
/// the file rather than making you read a list.
#[test]
fn the_shared_silu_activation_has_no_clamp() {
    let path = kernels_root().join("gb10/common/moe_silu_mul.cu");
    let text = std::fs::read_to_string(&path).expect("gb10/common/moe_silu_mul.cu");
    assert!(
        !text.contains("SWIGLU_LIMIT ="),
        "moe_silu_mul is the SiLU activation for every dense decode/K-verify FFN, \
         every MoE grouped prefill, and the MTP and DFlash draft heads. A clamp \
         here reaches all of them; only DeepSeek-V4 and Step-3.7 declare a limit, \
         and they shadow this file."
    );
}

/// The 27B's NVFP4-MMQ kernels own its PREFILL activation while `moe_silu_mul`
/// owns its decode and K-verify. They hand-duplicate each other's math, so a
/// clamp returning to one and not the other splits the model's own numerics —
/// which is the failure the wave-55 change exists to remove.
#[test]
fn the_27b_prefill_and_decode_activations_agree_on_clamping() {
    let root = kernels_root();
    let mmq = std::fs::read_to_string(root.join("gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu"))
        .expect("gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu");
    let shared = std::fs::read_to_string(root.join("gb10/common/moe_silu_mul.cu"))
        .expect("gb10/common/moe_silu_mul.cu");
    assert_eq!(
        mmq.contains("SWIGLU_LIMIT ="),
        shared.contains("SWIGLU_LIMIT ="),
        "Qwen3.6-27B's prefill activation (nvfp4_mmq.cu) and its decode/K-verify \
         activation (moe_silu_mul.cu) must clamp or not clamp together; one of \
         them changed without the other"
    );
}
