// SPDX-License-Identifier: AGPL-3.0-only
//
// Kernel-source file discovery helpers for build.rs. Included via
// `#[path = "build_files.rs"] mod build_files;`. Pure filesystem walking —
// no dependency on build.rs types.

use std::collections::HashMap;
use std::path::PathBuf;

/// List subdirectory names (not files) in a directory, sorted.
pub(super) fn list_subdirs(dir: &std::path::Path) -> Vec<String> {
    let mut dirs: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            if entry.file_type().ok()?.is_dir() {
                Some(entry.file_name().to_string_lossy().to_string())
            } else {
                None
            }
        })
        .collect();
    dirs.sort();
    dirs
}

/// Collect kernel-source files with shadowing: common dir provides the
/// base set, model-specific dir can override individual files by matching
/// filename. `source_ext` is the per-vendor extension (e.g. "cu" for
/// NVIDIA, "metal" for Apple).
pub(super) fn collect_cu_files(
    common_dir: Option<&std::path::Path>,
    model_dir: &std::path::Path,
    source_ext: &str,
) -> Vec<PathBuf> {
    let mut files: HashMap<String, PathBuf> = HashMap::new();

    // Base layer: common kernels
    if let Some(common) = common_dir {
        for f in find_cu_files(common, source_ext) {
            let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
            files.insert(stem, f);
        }
    }

    // Override layer: model-specific kernel files shadow common ones
    for f in find_cu_files(model_dir, source_ext) {
        let stem = f.file_stem().unwrap().to_str().unwrap().to_string();
        files.insert(stem, f);
    }

    let mut result: Vec<PathBuf> = files.into_values().collect();
    result.sort();
    result
}

/// Find all kernel-source files (extension `source_ext`) in a directory.
/// Returns empty vec if dir doesn't exist.
pub(super) fn find_cu_files(kernel_dir: &std::path::Path, source_ext: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(kernel_dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|e| e.to_str()) == Some(source_ext) {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}
