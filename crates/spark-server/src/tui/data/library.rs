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
        // "Ready" has to mean "the loader would accept this", not "some file
        // that looks like weights is present". Two ways it can differ, both
        // reachable the moment downloads exist:
        //   * `snapshot_has_weights` counts `model.safetensors.index.json`,
        //     which is small and lands FIRST — so a download that has barely
        //     started reads as complete;
        //   * a download that never finished has no `refs/main`, which is what
        //     the resolver actually keys on.
        let has_shard = std::fs::read_dir(&snap)
            .map(|rd| {
                rd.filter_map(|e| e.ok()).any(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.ends_with(".safetensors"))
                })
            })
            .unwrap_or(false);
        let published = e.path().join("refs/main").exists();
        let has_weights = has_shard && published;
        let mut entry = LibraryEntry {
            id,
            // `blobs/` is where huggingface-cli puts the real bytes, with
            // snapshots/ as symlinks into it — so `len()` of a snapshot entry
            // would measure the link, not the model. Atlas's own downloader
            // writes the files directly into snapshots/ and has no blobs/ at
            // all, which reported every model it fetched as 0 MB. Measure
            // whichever layout this model actually uses.
            size_bytes: match dir_size(&e.path().join("blobs")) {
                0 => dir_size(&snap),
                n => n,
            },
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

/// Human size. Delegates so the Library card and the download line that
/// replaces it cannot disagree about how big the same file is.
pub fn human_size(bytes: u64) -> String {
    crate::tui::format::bytes(bytes)
}

/// Run [`scan`] on its own thread, delivering the result over a channel.
///
/// The scan recursively `read_dir`s every `models--*/blobs`, which on a cache
/// holding a few dozen multi-gigabyte checkpoints is tens of milliseconds to
/// seconds with a cold page cache — and it was running on the render thread.
/// It also needs to be re-runnable now, because a finished download must
/// appear without restarting the dashboard.
///
/// Same shape as `recipe::fetch::refresh_in_background`, and the same rule:
/// the render thread only ever `try_recv`s.
pub fn scan_in_background(
    cache_dir: Option<&Path>,
) -> std::sync::mpsc::Receiver<Vec<LibraryEntry>> {
    let owned = cache_dir.map(|p| p.to_path_buf());
    // An empty list is what `scan` itself returns for an unreadable cache, so
    // it is also the honest answer when the scanner cannot start.
    crate::tui::worker::spawn(
        "atlas-libscan",
        move || scan(owned.as_deref()),
        |_| Vec::new(),
    )
}
