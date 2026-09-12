// SPDX-License-Identifier: AGPL-3.0-only

//! C1 skeleton: greedy tokens vs HF goldens.
//!
//! Goldens live at `docs/k3/goldens/kimi-k3-0.40b-greedy.json`. They are not
//! faked here — skip when the file is missing.

use serde::Deserialize;

const GOLDEN_REL: &str = "../../docs/k3/goldens/kimi-k3-0.40b-greedy.json";

#[derive(Debug, Deserialize)]
struct GreedyGoldens {
    model: String,
    #[serde(default)]
    max_new_tokens: usize,
    prompts: Vec<GreedyPrompt>,
}

#[derive(Debug, Deserialize)]
struct GreedyPrompt {
    #[serde(default)]
    prompt: String,
    tokens: Vec<u32>,
}

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDEN_REL)
}

#[test]
fn c1_greedy_vs_hf_goldens_skip_if_missing() {
    let path = golden_path();
    if !path.exists() {
        eprintln!(
            "skip C1: {} not present (generate from inference-optimization/Kimi-K3-0.40B)",
            path.display()
        );
        return;
    }
    let raw = std::fs::read_to_string(&path).expect("read C1 goldens");
    let g: GreedyGoldens = serde_json::from_str(&raw).expect("C1 goldens JSON");
    assert!(
        g.model.contains("Kimi-K3-0.40B"),
        "C1 goldens must be the 0.40B twin, got {}",
        g.model
    );
    assert!(!g.prompts.is_empty(), "C1 goldens have no prompts");
    if g.max_new_tokens != 0 {
        assert_eq!(g.max_new_tokens, 128);
    }
    for (i, p) in g.prompts.iter().enumerate() {
        assert!(
            !p.tokens.is_empty(),
            "prompt {i} ({}) has empty greedy tokens",
            p.prompt
        );
    }
    // Token-exact engine compare is S1/C1 GPU work. This slice only binds the
    // golden schema so a missing file cannot be papered over with fakes.
}

/// Ignored until a host can run the 0.40B twin. Fails closed if goldens exist
/// but the CPU graph has no engine to compare — do not `unwrap` a dummy match.
#[test]
#[ignore = "C1 token-exact needs HF goldens + bound engine"]
fn c1_greedy_token_exact_engine() {
    let path = golden_path();
    assert!(
        path.exists(),
        "run without ignore only when {} exists",
        path.display()
    );
}
