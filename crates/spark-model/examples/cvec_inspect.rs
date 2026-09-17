// SPDX-License-Identifier: AGPL-3.0-only

//! Parse a control-vector GGUF through the real loader and print what the
//! model would install. CPU-only — no GPU, no model — so it is the quickest
//! way to check a vector file against a given geometry before a serve boots.
//!
//!   cargo run -p spark-model --example cvec_inspect -- \
//!       <file.gguf> [hidden] [n_layer] [layer_start] [layer_end] [scale]
//!
//! Defaults are Qwen3.8-Flash-Next: hidden 2560, 48 layers, layers 4..=44,
//! scale 1.0 — the shipped configuration of the published refusal projection.

use std::path::PathBuf;

use spark_model::control_vector::{ControlVectorSpec, CvecMode, build_table};

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: cvec_inspect <file.gguf> [hidden] [n_layer] [start] [end] [scale]");
        std::process::exit(2);
    }
    let arg = |i: usize, d: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let hidden = arg(2, 2560);
    let n_layer = arg(3, 48);
    let spec = ControlVectorSpec {
        path: PathBuf::from(&a[1]),
        scale: a.get(6).and_then(|s| s.parse().ok()).unwrap_or(1.0),
        layer_start: arg(4, 4),
        layer_end: arg(5, 44),
        mode: CvecMode::Project,
    };

    let bytes = std::fs::read(&spec.path)?;
    println!(
        "{} ({} bytes)\n  geometry: hidden={hidden} n_layer={n_layer} \
         layers {}..={} scale={}",
        spec.path.display(),
        bytes.len(),
        spec.layer_start,
        spec.layer_end,
        spec.scale
    );

    let (table, scales) = build_table(&bytes, &spec, hidden, n_layer)?;
    let active: Vec<usize> = scales
        .iter()
        .enumerate()
        .filter(|(_, s)| **s != 0.0)
        .map(|(i, _)| i)
        .collect();
    println!(
        "  ACCEPTED: {} active layers ({}..={}), {} zeroed",
        active.len(),
        active.first().copied().unwrap_or(0),
        active.last().copied().unwrap_or(0),
        n_layer - active.len()
    );

    let (mut smin, mut smax) = (f32::MAX, f32::MIN);
    for &il in &active {
        smin = smin.min(scales[il]);
        smax = smax.max(scales[il]);
        let row = &table[il * hidden..(il + 1) * hidden];
        let n = row.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "layer {il} stored norm {n}");
    }
    println!("  per-layer scale: min={smin} max={smax}  (stored rows all unit norm)");

    // Cross-layer coherence: one feature carried across depth, or per-layer
    // noise? Near-orthogonal rows would mean the latter.
    let dot = |x: &[f32], y: &[f32]| -> f64 {
        x.iter().zip(y).map(|(p, q)| (*p as f64) * (*q as f64)).sum()
    };
    let mut adj = Vec::new();
    for w in active.windows(2) {
        adj.push(dot(
            &table[w[0] * hidden..(w[0] + 1) * hidden],
            &table[w[1] * hidden..(w[1] + 1) * hidden],
        ));
    }
    if !adj.is_empty() {
        println!(
            "  adjacent-layer cosine: mean={:.4} min={:.4}",
            adj.iter().sum::<f64>() / adj.len() as f64,
            adj.iter().copied().fold(f64::MAX, f64::min)
        );
    }
    Ok(())
}
