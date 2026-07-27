// SPDX-License-Identifier: AGPL-3.0-only

//! Resolves a model specifier (HuggingFace ID or local path) to a
//! validated directory containing `config.json` and weight files.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Resolve a model specifier to a local directory path.
///
/// Resolution order:
/// 1. If the specifier is an existing directory with `config.json`, use it directly.
/// 2. Otherwise, treat it as a HuggingFace model ID and look up the local cache.
pub fn resolve_model_dir(model: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let as_path = Path::new(model);
    if as_path.is_dir() && as_path.join("config.json").exists() {
        tracing::info!("Model path: {} (local directory)", as_path.display());
        return Ok(as_path.to_path_buf());
    }

    resolve_from_hf_cache(model, cache_dir)
}

/// Look up a HuggingFace model ID in the local hub cache.
///
/// Cache layout: `{cache_root}/models--{org}--{name}/snapshots/{hash}/`
/// The active snapshot hash is read from `refs/main`.
fn resolve_from_hf_cache(model_id: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let cache_root = resolve_cache_root(cache_dir)?;

    // "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4" → "models--nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4"
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_cache = cache_root.join(&dir_name);

    if !model_cache.is_dir() {
        bail!(
            "Model '{}' not found in HF cache at {}.\n\
             Download it first:\n  huggingface-cli download {}",
            model_id,
            cache_root.display(),
            model_id,
        );
    }

    let ref_path = model_cache.join("refs/main");
    let snapshot_hash = std::fs::read_to_string(&ref_path)
        .with_context(|| {
            format!(
                "No default revision for '{}'. Expected refs/main at {}.\n\
                 The model may not have been fully downloaded.",
                model_id,
                ref_path.display(),
            )
        })?
        .trim()
        .to_string();

    let snapshot_dir = model_cache.join("snapshots").join(&snapshot_hash);
    if !snapshot_dir.is_dir() {
        bail!(
            "Snapshot directory not found: {}\n\
             refs/main points to hash '{}' but that snapshot doesn't exist.",
            snapshot_dir.display(),
            snapshot_hash,
        );
    }

    if !snapshot_dir.join("config.json").exists() && !snapshot_dir.join("params.json").exists() {
        bail!(
            "Model directory exists but is missing config.json or params.json: {}\n\
             The download may be incomplete.",
            snapshot_dir.display(),
        );
    }

    // The snapshot pointed to by refs/main may be a metadata-only revision
    // (config + tokenizer present, but no safetensors) — observed in the wild
    // for nvidia/Gemma-4-31B-IT-NVFP4 where revision 05fa17… ships only the
    // chat template + config and the actual weights live in 1365cf… . If the
    // refs-pointed snapshot has no weights, fall back to scanning all
    // snapshots/* and pick the most-recent one that does, instead of bailing
    // and forcing the user to pass --model-from-path. The fallback is the
    // primary mechanism that lets the dual-DGX sweep orchestrator (which uses
    // HF model IDs, not paths) survive transient HF-side cache fragmentation.
    if snapshot_has_weights(&snapshot_dir) {
        tracing::info!(
            "Model: {} (resolved to {})",
            model_id,
            snapshot_dir.display(),
        );
        return Ok(snapshot_dir);
    }

    tracing::warn!(
        "Snapshot '{}' for {} has no weight files (metadata-only); \
         scanning {} for a sibling snapshot with weights.",
        snapshot_hash,
        model_id,
        model_cache.join("snapshots").display(),
    );

    if let Some(fallback) = find_snapshot_with_weights(&model_cache.join("snapshots")) {
        tracing::info!(
            "Model: {} (resolved to {} — fell back from refs/main snapshot {} which had no weights)",
            model_id,
            fallback.display(),
            snapshot_hash,
        );
        return Ok(fallback);
    }

    bail!(
        "Snapshot '{}' for {} has no weight files (no model.safetensors / \
         consolidated.safetensors / *.safetensors found in {}). Sibling \
         snapshots in {} also lack weights — refresh the cache:\n  \
         huggingface-cli download {} --revision main",
        snapshot_hash,
        model_id,
        snapshot_dir.display(),
        model_cache.join("snapshots").display(),
        model_id,
    );
}

/// Resolve a LoRA adapter specifier (local path or HF id) to a directory
/// containing `adapter_config.json`. Mirrors `resolve_model_dir`, but PEFT
/// adapter repos ship `adapter_config.json` + `adapter_model.safetensors`
/// and no `config.json`, so the marker checks differ.
pub fn resolve_adapter_dir(spec: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    let as_path = Path::new(spec);
    if as_path.is_dir() {
        if as_path.join("adapter_config.json").exists() {
            tracing::info!("Adapter path: {} (local directory)", as_path.display());
            return validate_adapter_dir(as_path.to_path_buf(), spec);
        }
        bail!(
            "Adapter directory {} has no adapter_config.json — not a PEFT adapter",
            as_path.display(),
        );
    }

    let cache_root = resolve_cache_root(cache_dir)?;
    let dir_name = format!("models--{}", spec.replace('/', "--"));
    let model_cache = cache_root.join(&dir_name);
    if !model_cache.is_dir() {
        bail!(
            "Adapter '{}' not found in HF cache at {}.\n\
             Download it first:\n  huggingface-cli download {}",
            spec,
            cache_root.display(),
            spec,
        );
    }

    let ref_path = model_cache.join("refs/main");
    let snapshot_hash = std::fs::read_to_string(&ref_path)
        .with_context(|| {
            format!(
                "No default revision for adapter '{}'. Expected refs/main at {}.\n\
                 The adapter may not have been fully downloaded.",
                spec,
                ref_path.display(),
            )
        })?
        .trim()
        .to_string();

    let snapshot_dir = model_cache.join("snapshots").join(&snapshot_hash);
    if !snapshot_dir.is_dir() {
        bail!(
            "Snapshot directory not found: {}\n\
             refs/main points to hash '{}' but that snapshot doesn't exist.",
            snapshot_dir.display(),
            snapshot_hash,
        );
    }

    if !snapshot_dir.join("adapter_config.json").exists() {
        bail!(
            "'{}' resolved to {} but it has no adapter_config.json — not a PEFT adapter repo",
            spec,
            snapshot_dir.display(),
        );
    }

    tracing::info!("Adapter: {} (resolved to {})", spec, snapshot_dir.display());
    validate_adapter_dir(snapshot_dir, spec)
}

/// Validate that a resolved adapter directory ships safetensors weights.
/// `adapter_model.bin` (torch pickle) is rejected by name so the failure
/// doesn't surface as a confusing missing-weights error two layers deeper.
fn validate_adapter_dir(dir: PathBuf, spec: &str) -> Result<PathBuf> {
    if dir.join("adapter_model.safetensors").exists() {
        return Ok(dir);
    }
    if dir.join("adapter_model.bin").exists() {
        bail!(
            "Adapter '{}' ships adapter_model.bin (torch pickle) — unsupported. \
             Re-export with save_pretrained(..., safe_serialization=True).",
            spec,
        );
    }
    bail!(
        "Adapter '{}' has no adapter_model.safetensors in {}",
        spec,
        dir.display(),
    );
}

/// True when the directory contains at least one weight file Atlas's
/// safetensors loader can pick up. Mirrors the heuristic in
/// `spark-runtime::weights::SafetensorsLoader::load`.
fn snapshot_has_weights(dir: &Path) -> bool {
    let direct = [
        "model.safetensors",
        "model.safetensors.index.json",
        "consolidated.safetensors",
        "consolidated.safetensors.index.json",
    ];
    if direct.iter().any(|n| dir.join(n).exists()) {
        return true;
    }
    // Sharded safetensors (model-00001-of-00004.safetensors, …).
    let Ok(read) = std::fs::read_dir(dir) else {
        return false;
    };
    read.filter_map(|e| e.ok()).any(|e| {
        e.file_name()
            .to_str()
            .is_some_and(|n| n.ends_with(".safetensors"))
    })
}

/// Pick the most-recently-modified snapshot under `snapshots/` that actually
/// contains weights. Returns `None` if none of the siblings have weights.
fn find_snapshot_with_weights(snapshots_root: &Path) -> Option<PathBuf> {
    let entries: Vec<_> = std::fs::read_dir(snapshots_root)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter_map(|e| {
            let p = e.path();
            // Skip the bad snapshot we already know about by checking weights.
            if !snapshot_has_weights(&p) {
                return None;
            }
            // Sort key: most recent mtime wins.
            let mtime = e.metadata().ok()?.modified().ok()?;
            Some((mtime, p))
        })
        .collect();
    entries.into_iter().max_by_key(|(t, _)| *t).map(|(_, p)| p)
}

/// Determine the HF hub cache root directory.
///
/// Precedence (matches official HuggingFace behavior):
/// 1. `cache_dir` argument (from `--cache-dir`)
/// 2. `$HF_HUB_CACHE` env var
/// 3. `$HF_HOME/hub` env var
/// 4. `~/.cache/huggingface/hub`
fn resolve_cache_root(cache_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = cache_dir {
        return Ok(dir.to_path_buf());
    }

    if let Ok(hub_cache) = std::env::var("HF_HUB_CACHE") {
        return Ok(PathBuf::from(hub_cache));
    }

    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(hf_home).join("hub"));
    }

    let home =
        std::env::var("HOME").context("Cannot determine home directory: $HOME is not set")?;
    Ok(PathBuf::from(home).join(".cache/huggingface/hub"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Create a mock HF cache structure for testing.
    fn setup_mock_cache(tmp: &Path, org: &str, name: &str, hash: &str) -> PathBuf {
        let model_id = format!("{org}/{name}");
        let dir_name = format!("models--{}", model_id.replace('/', "--"));
        let model_cache = tmp.join(&dir_name);

        let snapshot_dir = model_cache.join("snapshots").join(hash);
        fs::create_dir_all(&snapshot_dir).unwrap();
        fs::create_dir_all(model_cache.join("refs")).unwrap();
        fs::write(model_cache.join("refs/main"), hash).unwrap();
        fs::write(snapshot_dir.join("config.json"), "{}").unwrap();
        // Default mock includes weights; tests that need a metadata-only
        // snapshot must remove them explicitly via `setup_mock_cache_no_weights`.
        fs::write(snapshot_dir.join("model.safetensors"), b"weights").unwrap();

        snapshot_dir
    }

    /// Mock HF cache where the snapshot has only metadata (config + tokenizer)
    /// but no weight files — mirrors the real-world failure where refs/main
    /// pointed to a partial sync revision of nvidia/Gemma-4-31B-IT-NVFP4.
    fn setup_mock_cache_no_weights(tmp: &Path, org: &str, name: &str, hash: &str) -> PathBuf {
        let model_id = format!("{org}/{name}");
        let dir_name = format!("models--{}", model_id.replace('/', "--"));
        let model_cache = tmp.join(&dir_name);
        let snapshot_dir = model_cache.join("snapshots").join(hash);
        fs::create_dir_all(&snapshot_dir).unwrap();
        fs::create_dir_all(model_cache.join("refs")).unwrap();
        fs::write(model_cache.join("refs/main"), hash).unwrap();
        fs::write(snapshot_dir.join("config.json"), "{}").unwrap();
        fs::write(snapshot_dir.join("tokenizer.json"), "{}").unwrap();
        snapshot_dir
    }

    #[test]
    fn resolve_local_directory() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("config.json"), "{}").unwrap();

        let result = resolve_model_dir(tmp.path().to_str().unwrap(), None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), tmp.path());
    }

    #[test]
    fn resolve_hf_model_id() {
        let tmp = tempfile::tempdir().unwrap();
        let expected = setup_mock_cache(
            tmp.path(),
            "nvidia",
            "Qwen3-Next-80B-A3B-Instruct-NVFP4",
            "abc123",
        );

        let result =
            resolve_model_dir("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4", Some(tmp.path()));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn resolve_missing_model_gives_download_hint() {
        let tmp = tempfile::tempdir().unwrap();
        let result = resolve_model_dir("nonexistent/model", Some(tmp.path()));
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found in HF cache"));
        assert!(err.contains("huggingface-cli download"));
    }

    #[test]
    fn resolve_missing_config_json() {
        let tmp = tempfile::tempdir().unwrap();
        let dir_name = "models--org--model";
        let snapshot_dir = tmp.path().join(dir_name).join("snapshots/abc");
        fs::create_dir_all(&snapshot_dir).unwrap();
        fs::create_dir_all(tmp.path().join(dir_name).join("refs")).unwrap();
        fs::write(tmp.path().join(dir_name).join("refs/main"), "abc").unwrap();

        let result = resolve_model_dir("org/model", Some(tmp.path()));
        let err = result.unwrap_err().to_string();
        assert!(err.contains("missing config.json"));
    }

    #[test]
    fn cache_dir_arg_takes_precedence() {
        let custom = tempfile::tempdir().unwrap();
        let _expected = setup_mock_cache(custom.path(), "org", "model", "hash1");

        let result = resolve_model_dir("org/model", Some(custom.path()));
        assert!(result.is_ok());
        assert!(result.unwrap().starts_with(custom.path()));
    }

    #[test]
    fn falls_back_to_sibling_snapshot_with_weights() {
        // refs/main points at a metadata-only snapshot; a sibling snapshot
        // has the actual weights. Resolver must return the sibling instead
        // of bailing out (the dual-DGX sweep can't pass --model-from-path).
        let tmp = tempfile::tempdir().unwrap();
        let bad =
            setup_mock_cache_no_weights(tmp.path(), "nvidia", "Gemma-4-31B-IT-NVFP4", "05fa17");
        // Pre-existing sibling snapshot with safetensors.
        let model_cache = bad.parent().unwrap().parent().unwrap();
        let good_hash = "1365cf";
        let good = model_cache.join("snapshots").join(good_hash);
        fs::create_dir_all(&good).unwrap();
        fs::write(good.join("config.json"), "{}").unwrap();
        fs::write(good.join("model-00001-of-00004.safetensors"), b"shard").unwrap();
        // Touch the good dir so its mtime is newer than the bad one.
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(
            good.join("model-00001-of-00004.safetensors"),
            b"shard-newer",
        )
        .unwrap();

        let result = resolve_model_dir("nvidia/Gemma-4-31B-IT-NVFP4", Some(tmp.path()));
        assert!(result.is_ok(), "expected fallback to succeed: {:?}", result);
        assert_eq!(result.unwrap(), good);
    }

    #[test]
    fn bails_when_all_snapshots_lack_weights() {
        // Resolver should still surface a clear error if the entire cache
        // entry is metadata-only with no usable sibling.
        let tmp = tempfile::tempdir().unwrap();
        setup_mock_cache_no_weights(tmp.path(), "org", "model", "h1");
        let result = resolve_model_dir("org/model", Some(tmp.path()));
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("no weight files") || err.contains("metadata-only"),
            "expected weight-files error, got: {err}"
        );
        assert!(err.contains("huggingface-cli download"));
    }
}
