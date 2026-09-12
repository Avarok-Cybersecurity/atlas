// SPDX-License-Identifier: AGPL-3.0-only

//! C1: greedy tokens vs HF goldens + RST known-bad mutants.

use super::cpu_weights::{Ablation, K3CpuModel};
use super::greedy::greedy_decode;
use serde::Deserialize;

const GOLDEN_REL: &str = "../../docs/k3/goldens/kimi-k3-0.40b-greedy.json";

#[derive(Debug, Deserialize)]
struct GreedyGoldens {
    #[serde(default)]
    model: String,
    #[serde(default)]
    max_new_tokens: usize,
    #[serde(default)]
    prompts: Vec<GreedyPrompt>,
    #[serde(default)]
    rows: Vec<GreedyRow>,
}

#[derive(Debug, Deserialize)]
struct GreedyPrompt {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    tokens: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct GreedyRow {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    token_ids: Vec<u32>,
}

struct PromptRow {
    prompt: String,
    tokens: Vec<u32>,
}

fn golden_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(GOLDEN_REL)
}

fn load_goldens() -> Option<(String, usize, Vec<PromptRow>)> {
    let path = golden_path();
    if !path.exists() {
        eprintln!(
            "skip C1: {} not present (generate from inference-optimization/Kimi-K3-0.40B)",
            path.display()
        );
        return None;
    }
    let raw = std::fs::read_to_string(&path).expect("read C1 goldens");
    let g: GreedyGoldens = serde_json::from_str(&raw).expect("C1 goldens JSON");
    if !g.model.is_empty() {
        assert!(
            g.model.contains("Kimi-K3-0.40B"),
            "C1 goldens must be the 0.40B twin, got {}",
            g.model
        );
    }
    let max_new = if g.max_new_tokens == 0 {
        128
    } else {
        g.max_new_tokens
    };
    assert_eq!(max_new, 128, "C1 goldens max_new_tokens");
    let mut rows: Vec<PromptRow> = g
        .prompts
        .into_iter()
        .map(|p| PromptRow {
            prompt: p.prompt,
            tokens: p.tokens,
        })
        .collect();
    rows.extend(g.rows.into_iter().map(|r| PromptRow {
        prompt: r.prompt,
        tokens: r.token_ids,
    }));
    assert!(!rows.is_empty(), "C1 goldens have no prompts");
    assert_eq!(rows.len(), 8, "C1 expects 8 prompts, got {}", rows.len());
    for (i, p) in rows.iter().enumerate() {
        assert!(
            p.tokens.len() > max_new,
            "prompt {i} ({}) needs prompt+{max_new} token ids",
            p.prompt
        );
    }
    Some((g.model, max_new, rows))
}

#[test]
fn c1_greedy_vs_hf_goldens_skip_if_missing() {
    let Some((_model, max_new, rows)) = load_goldens() else {
        return;
    };
    let Some(engine) = twin_engine() else {
        eprintln!("skip C1 engine: no K3_TWIN weights (synthetic cannot match HF ids)");
        return;
    };
    for (i, row) in rows.iter().enumerate() {
        let split = row.tokens.len() - max_new;
        let prompt = &row.tokens[..split];
        let got = greedy_decode(&engine, prompt, max_new, Ablation::default());
        assert_eq!(
            got, row.tokens,
            "C1 prompt {i} ({}) token mismatch",
            row.prompt
        );
    }
}

#[test]
fn rst_attnres_mix0_or_force_expert0_diverges() {
    let model = K3CpuModel::synthetic_tiny();
    let prompt = [1u32, 2, 3];
    let clean = greedy_decode(&model, &prompt, 16, Ablation::default());
    let mix0 = greedy_decode(
        &model,
        &prompt,
        16,
        Ablation {
            attnres_mix: 0.0,
            force_expert: None,
        },
    );
    let exp0 = greedy_decode(
        &model,
        &prompt,
        16,
        Ablation {
            attnres_mix: 1.0,
            force_expert: Some(0),
        },
    );
    assert_ne!(
        mix0, clean,
        "RST known-bad: AttnRes mix=0 must change greedy tokens"
    );
    assert_ne!(
        exp0, clean,
        "RST known-bad: force expert 0 must change greedy tokens"
    );
}

#[test]
fn rst_mutant_fails_golden_compare() {
    let Some((_model, max_new, rows)) = load_goldens() else {
        return;
    };
    let Some(engine) = twin_engine() else {
        eprintln!("skip C1 mutant-vs-golden: no twin weights");
        return;
    };
    let row = &rows[0];
    let split = row.tokens.len() - max_new;
    let prompt = &row.tokens[..split];
    let mix0 = greedy_decode(
        &engine,
        prompt,
        max_new,
        Ablation {
            attnres_mix: 0.0,
            force_expert: None,
        },
    );
    assert_ne!(
        mix0, row.tokens,
        "planted AttnRes mix=0 must fail the golden compare"
    );
}

fn twin_engine() -> Option<K3CpuModel> {
    // Real 0.40B safetensors live on spark1 (`K3_TWIN`). Mac unit tests use
    // synthetic weights; HF token-exact compare is skipped without them.
    match std::env::var("K3_TWIN") {
        Ok(p) if std::path::Path::new(&p).exists() => {
            eprintln!("K3_TWIN={p} present but BF16 safetensors→K3CpuModel not this slice");
            None
        }
        _ => None,
    }
}
