// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the NLLB PEFT LoRA path. As a child module of `lora`, these
//! may construct `LoraSet`/`LoraPair` with private fields directly.

use std::collections::HashMap;

use safetensors::tensor::{Dtype, TensorView};

use super::*;

/// Reference `scale · (x·Aᵀ)·Bᵀ` computed with plain loops.
fn reference_delta(
    x: &[f32],
    rows: usize,
    a: &[f32],
    r: usize,
    in_dim: usize,
    b: &[f32],
    out_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; rows * out_dim];
    for i in 0..rows {
        // xa[k] = Σ_j x[i,j] · A[k,j]
        let mut xa = vec![0f32; r];
        for (k, xak) in xa.iter_mut().enumerate() {
            for j in 0..in_dim {
                *xak += x[i * in_dim + j] * a[k * in_dim + j];
            }
        }
        for o in 0..out_dim {
            let mut acc = 0f32;
            for k in 0..r {
                acc += xa[k] * b[o * r + k];
            }
            out[i * out_dim + o] = acc * scale;
        }
    }
    out
}

fn set_with(
    module: &str,
    a: Vec<f32>,
    b: Vec<f32>,
    r: usize,
    in_dim: usize,
    out_dim: usize,
    scale: f32,
) -> LoraSet {
    let mut pairs = HashMap::new();
    pairs.insert(
        module.to_string(),
        LoraPair {
            a,
            b,
            r,
            in_dim,
            out_dim,
        },
    );
    LoraSet { scale, pairs }
}

#[test]
fn delta_matches_reference() {
    let (r, in_dim, out_dim, rows) = (2usize, 3usize, 2usize, 2usize);
    let a = vec![0.1, -0.2, 0.3, 0.4, 0.5, -0.6]; // [r, in]
    let b = vec![1.0, -1.0, 0.5, 2.0]; // [out, r]
    let x = vec![1.0, 2.0, 3.0, -1.0, 0.0, 0.5]; // [rows, in]
    let scale = 2.0f32;
    let set = set_with("m", a.clone(), b.clone(), r, in_dim, out_dim, scale);

    let got = set.delta("m", &x, rows).expect("adapted");
    let want = reference_delta(&x, rows, &a, r, in_dim, &b, out_dim, scale);
    assert_eq!(got.len(), rows * out_dim);
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-6, "got {g} want {w}");
    }
}

#[test]
fn delta_none_for_unadapted_module() {
    let set = set_with("q_proj", vec![0.0; 6], vec![0.0; 4], 2, 3, 2, 1.0);
    assert!(set.delta("k_proj", &[0.0; 3], 1).is_none());
}

#[test]
fn zero_b_gives_zero_delta() {
    // Freshly-initialised LoRA (B == 0) is a no-op — the base-identity guarantee.
    let set = set_with("m", vec![0.7; 6], vec![0.0; 4], 2, 3, 2, 3.5);
    let d = set.delta("m", &[1.0, 2.0, 3.0], 1).unwrap();
    assert!(d.iter().all(|v| v.abs() < 1e-9));
}

#[test]
fn strip_lora_key_recovers_module_path() {
    let full = "base_model.model.model.encoder.layers.0.self_attn.q_proj.lora_A.weight";
    assert_eq!(
        strip_lora_key(full, ".lora_A.weight").as_deref(),
        Some("model.encoder.layers.0.self_attn.q_proj")
    );
    // Already-unwrapped key (no base_model prefix) still works.
    let bare = "model.decoder.layers.3.fc2.lora_B.weight";
    assert_eq!(
        strip_lora_key(bare, ".lora_B.weight").as_deref(),
        Some("model.decoder.layers.3.fc2")
    );
    // Wrong suffix → None.
    assert!(strip_lora_key(full, ".lora_B.weight").is_none());
}

fn f32_view(shape: Vec<usize>, data: &[f32]) -> (Vec<usize>, Vec<u8>) {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    (shape, bytes)
}

#[test]
fn load_dir_roundtrip() {
    // Build a tiny PEFT adapter on disk (config + safetensors) and load it,
    // exercising strip_lora_key + the safetensors parse + pairing end-to-end.
    let (r, in_dim, out_dim) = (2usize, 3usize, 2usize);
    let a = vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
    let b = vec![1.0f32, 2.0, 3.0, 4.0];
    let dir = std::env::temp_dir().join(format!("nllb_lora_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("adapter_config.json"),
        r#"{"r":2,"lora_alpha":4.0,"target_modules":["q_proj"]}"#,
    )
    .unwrap();

    let module = "model.encoder.layers.0.self_attn.q_proj";
    let (a_shape, a_bytes) = f32_view(vec![r, in_dim], &a);
    let (b_shape, b_bytes) = f32_view(vec![out_dim, r], &b);
    let a_view = TensorView::new(Dtype::F32, a_shape, &a_bytes).unwrap();
    let b_view = TensorView::new(Dtype::F32, b_shape, &b_bytes).unwrap();
    let tensors = vec![
        (format!("base_model.model.{module}.lora_A.weight"), a_view),
        (format!("base_model.model.{module}.lora_B.weight"), b_view),
    ];
    let st = safetensors::serialize(tensors, None).unwrap();
    std::fs::write(dir.join("adapter_model.safetensors"), st).unwrap();

    let set = LoraSet::load_dir(&dir).unwrap();
    assert_eq!(set.adapted_modules(), 1);
    // scale = alpha/r = 4/2 = 2.
    let x = vec![1.0f32, 2.0, 3.0];
    let got = set.delta(module, &x, 1).unwrap();
    let want = reference_delta(&x, 1, &a, r, in_dim, &b, out_dim, 2.0);
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-6, "got {g} want {w}");
    }
    assert!(
        set.delta("model.encoder.layers.0.self_attn.k_proj", &x, 1)
            .is_none()
    );
    std::fs::remove_dir_all(&dir).ok();
}
