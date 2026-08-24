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
const KNOWN_INCONSISTENT: &[&str] = &["gb10/common/moe_shared_expert_fused.cu"];

fn kernels_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/spark-model is two levels below the workspace root")
        .join("kernels")
}

fn cuda_code_mentions_swiglu_limit(text: &str) -> bool {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        LineComment,
        BlockComment,
        Quoted(char),
    }

    let mut state = State::Code;
    let mut token = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        match state {
            State::Code if ch == '/' && chars.peek() == Some(&'/') => {
                chars.next();
                token.clear();
                state = State::LineComment;
            }
            State::Code if ch == '/' && chars.peek() == Some(&'*') => {
                chars.next();
                token.clear();
                state = State::BlockComment;
            }
            State::Code if ch == '"' || ch == '\'' => {
                token.clear();
                state = State::Quoted(ch);
            }
            State::Code if ch.is_ascii_alphanumeric() || ch == '_' => token.push(ch),
            State::Code => {
                if token == "SWIGLU_LIMIT" {
                    return true;
                }
                token.clear();
            }
            State::LineComment if ch == '\n' => state = State::Code,
            State::BlockComment if ch == '*' && chars.peek() == Some(&'/') => {
                chars.next();
                state = State::Code;
            }
            State::Quoted(quote) if ch == '\\' => {
                chars.next();
            }
            State::Quoted(quote) if ch == quote => state = State::Code,
            _ => {}
        }
    }
    token == "SWIGLU_LIMIT"
}

fn is_known_inconsistent(path: &Path) -> bool {
    KNOWN_INCONSISTENT
        .iter()
        .any(|known| path == Path::new(known))
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
                && std::fs::read_to_string(&path).is_ok_and(|t| cuda_code_mentions_swiglu_limit(&t))
            {
                hits.push(path);
            }
        }
    }
    hits.sort();
    hits
}

#[test]
fn clamp_scanner_distinguishes_code_from_comments_and_spacing() {
    assert!(cuda_code_mentions_swiglu_limit(
        "const float SWIGLU_LIMIT=10.0f;"
    ));
    assert!(cuda_code_mentions_swiglu_limit(
        "constexpr float\n SWIGLU_LIMIT\t = 7.0f;"
    ));
    assert!(!cuda_code_mentions_swiglu_limit(
        "// const float SWIGLU_LIMIT = 10.0f;\nfloat x = 1.0f;"
    ));
    assert!(!cuda_code_mentions_swiglu_limit(
        "/* SWIGLU_LIMIT = 10 */ const char* name = \"SWIGLU_LIMIT\";"
    ));
    assert!(is_known_inconsistent(Path::new(
        "gb10/common/moe_shared_expert_fused.cu"
    )));
    assert!(!is_known_inconsistent(Path::new(
        "gb10/another-model/moe_shared_expert_fused.cu"
    )));
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
        let recorded = is_known_inconsistent(rel);
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
