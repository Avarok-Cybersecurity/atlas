// SPDX-License-Identifier: AGPL-3.0-only

//! C1: greedy tokens vs HF goldens + RST known-bad mutants.

use super::cache::HybridCache;
use super::cpu_forward::{forward_token, logits};
use super::cpu_weights::{Ablation, K3CpuModel};
use super::greedy::greedy_decode;
use super::ops::argmax;
use serde::Deserialize;

/// Golden prompt 0 first generated id. 387 is the *second* generated token
/// (the first mismatch in the C1 FAIL log).
const HF_PROMPT0_FIRST: u32 = 1459;
const HF_PROMPT0_SECOND: u32 = 387;
/// Twin tokenizer `eos_token_id`. Greedy stops here. Tokens after the first
/// EOS in the 128-cap goldens are not an oracle (HF then loops `[EOS]`).
const TWIN_EOS: u32 = 163585;

fn until_first_eos(ids: &[u32]) -> &[u32] {
    match ids.iter().position(|&t| t == TWIN_EOS) {
        Some(i) => &ids[..=i],
        None => ids,
    }
}

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
    let mut fails = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let split = row.tokens.len() - max_new;
        let prompt = &row.tokens[..split];
        let want = until_first_eos(&row.tokens);
        if want.last() != Some(&TWIN_EOS) {
            fails.push(format!("prompt {i}: golden never hits EOS"));
            continue;
        }
        let n_new = want.len() - split;
        let got = greedy_decode(&engine, prompt, n_new, Ablation::default());
        let got = until_first_eos(&got);
        if got == want {
            eprintln!("C1 prompt {i} OK through EOS ({} new)", n_new);
            continue;
        }
        let fork = got
            .iter()
            .zip(want.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(got.len().min(want.len()));
        fails.push(format!(
            "prompt {i} ({:?}) fork@{fork} ours={:?} hf={:?}",
            row.prompt,
            got.get(fork),
            want.get(fork)
        ));
    }
    assert!(
        fails.is_empty(),
        "C1 until-EOS failures ({}/8):\n{}",
        fails.len(),
        fails.join("\n")
    );
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
            ..Ablation::default()
        },
    );
    let exp0 = greedy_decode(
        &model,
        &prompt,
        16,
        Ablation {
            force_expert: Some(0),
            ..Ablation::default()
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
            ..Ablation::default()
        },
    );
    assert_ne!(
        mix0, row.tokens,
        "planted AttnRes mix=0 must fail the golden compare"
    );
}

fn topk_logits(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx.into_iter().map(|i| (i as u32, logits[i])).collect()
}

/// One prefill of prompt 0; dump last-position top-8 vs HF first generated id.
#[test]
fn c1_prompt0_first_token_top8() {
    let Some((_model, max_new, rows)) = load_goldens() else {
        return;
    };
    let Some(engine) = twin_engine() else {
        eprintln!("skip C1 first-token dump: no K3_TWIN");
        return;
    };
    let row = &rows[0];
    let split = row.tokens.len() - max_new;
    let prompt = &row.tokens[..split];
    assert_eq!(prompt.last().copied(), Some(11));
    assert_eq!(row.tokens[split], HF_PROMPT0_FIRST);
    assert_eq!(row.tokens[split + 1], HF_PROMPT0_SECOND);
    let mut cache = HybridCache::from_graph(&engine.graph, &engine.kda);
    let mut h = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        h = forward_token(&engine, tok, pos, &mut cache, Ablation::default());
    }
    let lg = logits(&engine, &h);
    let top = topk_logits(&lg, 8);
    let pred = argmax(&lg);
    eprintln!(
        "C1 prompt0 last-pos argmax={pred} (HF first {HF_PROMPT0_FIRST}, second {HF_PROMPT0_SECOND}); top-8={top:?}"
    );
    assert_eq!(
        pred, HF_PROMPT0_FIRST,
        "first generated token {pred} != HF {HF_PROMPT0_FIRST}; top-8={top:?}"
    );
}

#[test]
fn c1_prompt0_first_eight_generated() {
    let Some((_model, max_new, rows)) = load_goldens() else {
        return;
    };
    let Some(engine) = twin_engine() else {
        eprintln!("skip C1 first-8: no K3_TWIN");
        return;
    };
    let row = &rows[0];
    let split = row.tokens.len() - max_new;
    let prompt = &row.tokens[..split];
    let got = greedy_decode(&engine, prompt, 8, Ablation::default());
    let want = &row.tokens[..split + 8];
    eprintln!("C1 prompt0 first-8 got={:?} want={want:?}", &got[split..]);
    assert_eq!(
        got, want,
        "C1 prompt 0 first 8 generated tokens (not claiming full 128)"
    );
}

fn twin_engine() -> Option<K3CpuModel> {
    // 0.40B BF16 safetensors on spark1 (`K3_TWIN`). Missing path skips;
    // a present dir that fails to load fails the test.
    match std::env::var("K3_TWIN") {
        Ok(p) => {
            let path = std::path::Path::new(&p);
            if !path.exists() {
                eprintln!("skip C1 engine: K3_TWIN={p} does not exist");
                return None;
            }
            Some(
                K3CpuModel::from_pretrained(path)
                    .unwrap_or_else(|e| panic!("K3_TWIN={p} load failed: {e:#}")),
            )
        }
        _ => None,
    }
}
