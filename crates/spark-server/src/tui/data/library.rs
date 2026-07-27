// SPDX-License-Identifier: AGPL-3.0-only

//! Library tab data: enumerate locally cached HF models and read their
//! metadata from config.json — zero GPU, zero weight loading. Reuses the
//! resolver's cache-root precedence and weights predicate so the list agrees
//! with what `spark serve <model>` would actually find.

use std::path::{Path, PathBuf};

/// One locally cached model.
#[derive(Clone, Debug)]
pub struct LibraryEntry {
    /// HF id, un-mangled (`org/name`).
    pub id: String,
    pub snapshot_dir: PathBuf,
    pub size_bytes: u64,
    pub has_weights: bool,
    /// From config.json (None when parse fails — still listed).
    pub model_type: String,
    pub quant: String,
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub experts: usize,
    pub context: usize,
    /// A compiled kernel target matches this model's (model_type, hidden).
    pub optimized: bool,
}

/// Directory size (recursive, follows no symlinks). Snapshot dirs hardlink
/// into blobs/, so `len()` of the resolved files is the honest number.
fn dir_size(dir: &Path) -> u64 {
    let mut total = 0u64;
    let Ok(rd) = std::fs::read_dir(dir) else {
        return 0;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            total += dir_size(&p);
        } else if let Ok(md) = std::fs::metadata(&p) {
            total += md.len();
        }
    }
    total
}

/// Scan the HF cache. `cache_dir` is the `--cache-dir` override, if any.
pub fn scan(cache_dir: Option<&Path>) -> Vec<LibraryEntry> {
    let Ok(root) = crate::model_resolver::resolve_cache_root(cache_dir) else {
        return Vec::new();
    };
    let Ok(rd) = std::fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(mangled) = name.strip_prefix("models--") else {
            continue;
        };
        let id = mangled.replace("--", "/");
        let snapshots = e.path().join("snapshots");
        let Some(snap) =
            crate::model_resolver::find_snapshot_with_weights(&snapshots).or_else(|| {
                // No weights: still list the newest snapshot if one exists.
                std::fs::read_dir(&snapshots)
                    .ok()?
                    .flatten()
                    .map(|s| s.path())
                    .find(|p| p.is_dir())
            })
        else {
            continue;
        };
        let has_weights = crate::model_resolver::snapshot_has_weights(&snap);
        let mut entry = LibraryEntry {
            id,
            size_bytes: dir_size(&e.path().join("blobs")),
            snapshot_dir: snap.clone(),
            has_weights,
            model_type: "?".into(),
            quant: "-".into(),
            layers: 0,
            hidden: 0,
            heads: 0,
            experts: 0,
            context: 0,
            optimized: false,
        };
        if let Ok(json) = std::fs::read_to_string(snap.join("config.json"))
            && let Ok(cfg) = atlas_core::config::parse_config(&json)
        {
            entry.model_type = cfg.model_type.clone();
            entry.layers = cfg.num_hidden_layers;
            entry.hidden = cfg.hidden_size;
            entry.heads = cfg.num_attention_heads;
            entry.experts = cfg.num_experts;
            entry.context = cfg.max_position_embeddings;
            if let Some(q) = &cfg.quantization_config {
                entry.quant = if q.quant_algo.is_empty() {
                    q.quant_method.clone()
                } else {
                    q.quant_algo.to_lowercase()
                };
            }
            entry.optimized =
                atlas_kernels::ptx_for_config(&cfg.model_type, cfg.hidden_size).is_some();
        }
        out.push(entry);
    }
    out.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));
    out
}

/// Human size, GiB with one decimal above 1 GiB.
pub fn human_size(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let g = bytes as f64 / GIB;
    if g >= 1.0 {
        format!("{g:.1} GB")
    } else {
        format!("{} MB", bytes / (1024 * 1024))
    }
}
