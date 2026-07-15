// SPDX-License-Identifier: AGPL-3.0-only

//! End-to-end validation against HuggingFace `M2M100ForConditionalGeneration`.
//!
//! Requires a local safetensors NLLB checkpoint; set `NLLB_MODEL_DIR` to run
//! (e.g. a clone of `MonumentalSystems/nllb-200-3.3B`). Skips silently when
//! unset so CPU-only CI without the weights stays green.
//!
//! Ground truth captured from `transformers` 4.x on
//! `facebook/nllb-200-3.3B` (fp32), input "Hello, world. How are you today?"
//! eng_Latn → fra_Latn, greedy:
//!   input_ids = [256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259,
//!                30435, 248130, 2]
//!   forced_bos (fra_Latn) = 256057
//!   generated = [256057, 17994, 141190, 248079, 25358, 123732, 248105,
//!                30213, 248079, 1724, 25601, 385, 2]
//!   encoder last_hidden_state.sum() = -14.769035

use std::path::{Path, PathBuf};

use safetensors::tensor::{Dtype, TensorView};
use spark_nllb::NllbModel;

fn model_dir() -> Option<PathBuf> {
    std::env::var("NLLB_MODEL_DIR").ok().map(PathBuf::from)
}

/// Write a synthetic PEFT adapter (rank `r`, alpha `2r`) targeting encoder
/// `self_attn.q_proj` + `v_proj` over the first `layers` layers. `b_scale == 0`
/// yields a no-op adapter (B all zero); a non-zero scale perturbs the output.
fn write_synthetic_adapter(dir: &Path, d_model: usize, layers: usize, r: usize, b_scale: f32) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("adapter_config.json"),
        format!(
            r#"{{"r":{r},"lora_alpha":{},"target_modules":["q_proj","v_proj"]}}"#,
            2 * r
        ),
    )
    .unwrap();
    // A gets deterministic non-zero values; B is scaled (0 → no-op).
    let a: Vec<f32> = (0..r * d_model)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
        .collect();
    let b: Vec<f32> = (0..d_model * r)
        .map(|i| ((i % 5) as f32 - 2.0) * b_scale)
        .collect();
    let a_bytes: Vec<u8> = a.iter().flat_map(|f| f.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tensors = Vec::new();
    for l in 0..layers {
        for proj in ["q_proj", "v_proj"] {
            let m = format!("model.encoder.layers.{l}.self_attn.{proj}");
            tensors.push((
                format!("base_model.model.{m}.lora_A.weight"),
                TensorView::new(Dtype::F32, vec![r, d_model], &a_bytes).unwrap(),
            ));
            tensors.push((
                format!("base_model.model.{m}.lora_B.weight"),
                TensorView::new(Dtype::F32, vec![d_model, r], &b_bytes).unwrap(),
            ));
        }
    }
    let st = safetensors::serialize(tensors, None).unwrap();
    std::fs::write(dir.join("adapter_model.safetensors"), st).unwrap();
}

const INPUT_IDS: &[u32] = &[
    256047, 94124, 248079, 15697, 248075, 13374, 2442, 1259, 30435, 248130, 2,
];
const FORCED_BOS: u32 = 256057;
const EXPECTED_GEN: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 123732, 248105, 30213, 248079, 1724, 25601, 385, 2,
];
// beam=5, length_penalty=1.0, early_stopping=false (NLLB defaults):
//   "Bonjour, comment vous portez-vous aujourd'hui ?"
const EXPECTED_BEAM5: &[u32] = &[
    256057, 17994, 141190, 248079, 25358, 4255, 956, 34821, 248105, 30213, 102506, 248116, 15510,
    385, 2,
];

#[test]
fn nllb_greedy_matches_reference() {
    let Some(dir) = model_dir() else {
        eprintln!("NLLB_MODEL_DIR not set — skipping");
        return;
    };
    let model = NllbModel::load_dir(&dir).expect("load model");

    // Encoder numerics: sum of the encoder hidden state.
    let enc = model.encode(INPUT_IDS);
    let sum: f32 = enc.iter().sum();
    assert!(
        (sum - (-14.769035)).abs() < 0.05,
        "encoder sum {sum} != reference -14.769035"
    );

    // Greedy generation exact-token match.
    let out = model.generate(INPUT_IDS, FORCED_BOS, 64);
    assert_eq!(out, EXPECTED_GEN, "greedy ids diverged from reference");
}

#[test]
fn nllb_beam5_matches_reference() {
    let Some(dir) = model_dir() else {
        eprintln!("NLLB_MODEL_DIR not set — skipping");
        return;
    };
    let model = NllbModel::load_dir(&dir).expect("load model");
    // NLLB defaults: num_beams=5, length_penalty=1.0, early_stopping=false.
    let out = model.generate_beam(INPUT_IDS, FORCED_BOS, 5, 64, 1.0, false);
    assert_eq!(out, EXPECTED_BEAM5, "beam=5 ids diverged from reference");
}

/// A zero-B PEFT adapter must be a byte-exact no-op end-to-end: the LoRA delta
/// is `scale·(x·Aᵀ)·Bᵀ`, so `B == 0` adds nothing, and the adapted model must
/// reproduce the base encoder output and greedy tokens exactly.
#[test]
fn nllb_lora_zero_b_is_noop() {
    let Some(dir) = model_dir() else {
        eprintln!("NLLB_MODEL_DIR not set — skipping");
        return;
    };
    let base = NllbModel::load_dir(&dir).expect("load base");
    let enc_base = base.encode(INPUT_IDS);
    let gen_base = base.generate(INPUT_IDS, FORCED_BOS, 16);

    let ad = std::env::temp_dir().join(format!("nllb_lora_noop_{}", std::process::id()));
    write_synthetic_adapter(&ad, base.cfg.d_model, base.cfg.encoder_layers, 8, 0.0);
    let m = NllbModel::load_dir_with_lora(&dir, &ad).expect("load with lora");
    assert!(m.lora_modules() > 0, "adapter attached no modules");

    let enc = m.encode(INPUT_IDS);
    assert_eq!(enc.len(), enc_base.len());
    for (a, b) in enc.iter().zip(enc_base.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "zero-B adapter perturbed encoder: {a} vs {b}"
        );
    }
    let gen_lora = m.generate(INPUT_IDS, FORCED_BOS, 16);
    assert_eq!(gen_lora, gen_base, "zero-B adapter changed greedy tokens");
    std::fs::remove_dir_all(&ad).ok();
}

/// A non-zero adapter must actually change the encoder output (the delta is
/// live on every adapted projection), confirming the routing is wired, not
/// silently dropped.
#[test]
fn nllb_lora_nonzero_changes_output() {
    let Some(dir) = model_dir() else {
        eprintln!("NLLB_MODEL_DIR not set — skipping");
        return;
    };
    let base = NllbModel::load_dir(&dir).expect("load base");
    let enc_base = base.encode(INPUT_IDS);

    let ad = std::env::temp_dir().join(format!("nllb_lora_live_{}", std::process::id()));
    write_synthetic_adapter(&ad, base.cfg.d_model, base.cfg.encoder_layers, 8, 0.02);
    let m = NllbModel::load_dir_with_lora(&dir, &ad).expect("load with lora");
    let enc = m.encode(INPUT_IDS);
    let max_diff = enc
        .iter()
        .zip(enc_base.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "non-zero adapter left encoder unchanged (max_diff={max_diff})"
    );
    std::fs::remove_dir_all(&ad).ok();
}
