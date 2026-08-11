// SPDX-License-Identifier: AGPL-3.0-only

//! Writing a fixture baseline into a scratch repo.
//!
//! Split from `tests.rs` for the 500-LoC cap. It grew its own file when the
//! thresholds moved into `kernels/<hw>/<model>/BENCH.toml`: a fixture can no
//! longer drop one JSON file wherever it likes, it has to build the small piece
//! of kernel tree that locates the numbers.

use std::path::Path;

use super::record::GateBaseline;

/// Write a fixture baseline where the gate now reads it: as `BENCH.toml`
/// under a synthetic `kernels/<hw>/<model-dir>/`.
///
/// The thresholds used to be one JSON file per gate, which a test could drop
/// anywhere. They are now assembled from every model's `BENCH.toml`, so a
/// fixture has to build the small piece of kernel tree that locates them —
/// which is the point of the move: a benchmark's numbers cannot exist apart
/// from a model.
pub(super) fn write_baseline(root: &Path, benchmark_id: &str, baseline: &GateBaseline) {
    for (hw, entry) in &baseline.hardware {
        for (checkpoint, model) in &entry.models {
            // One synthetic model dir per checkpoint. `checkpoint` is a
            // HuggingFace name and cannot be a directory, so it is sanitised —
            // the directory name is arbitrary, only the declared checkpoint is
            // load-bearing.
            let dir_name = checkpoint.replace('/', "_");
            let dir = root.join("kernels").join(hw).join(&dir_name);
            std::fs::create_dir_all(dir.join("nvfp4")).unwrap();
            std::fs::write(
                root.join("kernels").join(hw).join("HARDWARE.toml"),
                "[hardware]\nvendor = \"nvidia\"\n",
            )
            .unwrap();
            std::fs::write(dir.join("MODEL.toml"), "[behavior]\n").unwrap();

            let mut toml = String::new();
            let existing = dir.join("BENCH.toml");
            if existing.exists() {
                toml.push_str(&std::fs::read_to_string(&existing).unwrap());
                toml.push('\n');
            }
            toml.push_str("[[benchmarks]]\nquant = \"nvfp4\"\n");
            toml.push_str(&format!("checkpoint = {}\n", json_str(checkpoint)));
            toml.push_str(&format!("gate = {}\n", json_str(benchmark_id)));
            if let Some(recipe) = &model.recipe {
                toml.push_str(&format!("recipe = {}\n", json_str(recipe)));
            }
            if checkpoint == &entry.default {
                toml.push_str("default = true\n");
            }
            toml.push_str("status = \"measured\"\n");
            toml.push_str(&format!("note = {}\n", json_str(&model.note)));
            for (name, bound) in &model.metrics {
                toml.push_str(&format!("\n[benchmarks.metrics.{name}]\n"));
                if let Some(v) = bound.min {
                    toml.push_str(&format!("min = {v}\n"));
                }
                if let Some(v) = bound.max {
                    toml.push_str(&format!("max = {v}\n"));
                }
                if let Some(v) = bound.noise {
                    toml.push_str(&format!("noise = {v}\n"));
                }
            }
            std::fs::write(existing, toml).unwrap();
        }
    }
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}
