// SPDX-License-Identifier: AGPL-3.0-only

//! Resolves a model specifier (HuggingFace ID or local path) to a
//! validated directory containing `config.json` and weight files.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// The command that fetches `id` into the HF cache this resolver reads.
///
/// One function rather than the same line written at each dead end, because
/// this advice went stale: `huggingface-cli` was renamed `hf` and REMOVED in
/// huggingface_hub 1.0, so the message every first-time user hits told them to
/// run a binary that is not installed. Spelled once, it can only be wrong once.
///
/// The old name is still named — a box pinned to a huggingface_hub older than
/// 1.0 has only `huggingface-cli`, and a reader who typed `hf` and got "command
/// not found" needs to be told what to type instead, not left guessing.
fn download_hint(id: &str) -> String {
    download_hint_at(id, None)
}

/// [`download_hint`] for a specific revision: `hf download` needs
/// `--revision` to fetch a non-default branch, and a hint that omits it sends
/// the user to download the wrong checkpoint.
fn download_hint_at(id: &str, revision: Option<&str>) -> String {
    let rev = revision
        .map(|r| format!(" --revision {r}"))
        .unwrap_or_default();
    format!(
        "  hf download {id}{rev}\n\
         (`hf` is the huggingface_hub CLI; before 1.0 it was `huggingface-cli`. \
         The dashboard's Library downloads into the same cache.)"
    )
}

/// Resolve a model specifier to a local directory path.
///
/// Resolution order:
/// 1. If the specifier is an existing directory with `config.json`, use it directly.
/// 2. Otherwise, treat it as a HuggingFace model ID and look up the local cache
///    at its default revision (`refs/main`).
pub fn resolve_model_dir(model: &str, cache_dir: Option<&Path>) -> Result<PathBuf> {
    resolve_model_dir_at(model, None, cache_dir)
}

/// [`resolve_model_dir`] at a specific HF revision (branch, tag or commit).
///
/// A serve preset pins the branch its checkpoint lives on — turboderp's EXL3
/// repos publish each bit-width as a branch, and the default branch is not the
/// validated one. `None` is the default revision, byte-for-byte the old
/// behaviour. A local directory still wins over the cache lookup: a revision
/// is meaningless for a path and is ignored for one.
pub fn resolve_model_dir_at(
    model: &str,
    revision: Option<&str>,
    cache_dir: Option<&Path>,
) -> Result<PathBuf> {
    let as_path = Path::new(model);
    // Accept a local dir that has a config.json OR a bare .gguf (served via the
    // GGUF loader + config-from-metadata path).
    if as_path.is_dir()
        && (as_path.join("config.json").exists()
            || spark_runtime::weights::find_gguf(as_path).is_some())
    {
        tracing::info!("Model path: {} (local directory)", as_path.display());
        return Ok(as_path.to_path_buf());
    }

    resolve_from_hf_cache(model, revision, cache_dir)
}

/// Look up a HuggingFace model ID in the local hub cache.
///
/// Cache layout: `{cache_root}/models--{org}--{name}/snapshots/{hash}/`
/// The active snapshot hash is read from `refs/main`, or from
/// `refs/<revision>` when a revision is given (`hf download --revision` writes
/// that ref). A revision that is itself a snapshot hash — a commit — is
/// accepted directly when no ref of that name exists.
fn resolve_from_hf_cache(
    model_id: &str,
    revision: Option<&str>,
    cache_dir: Option<&Path>,
) -> Result<PathBuf> {
    let cache_root = resolve_cache_root(cache_dir)?;

    // "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4" → "models--nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4"
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_cache = cache_root.join(&dir_name);

    if !model_cache.is_dir() {
        bail!(
            "Model '{}' not found in HF cache at {}.\n\
             Download it first:\n{}",
            model_id,
            cache_root.display(),
            download_hint_at(model_id, revision),
        );
    }

    let ref_name = revision.unwrap_or("main");
    let ref_path = model_cache.join("refs").join(ref_name);
    let snapshot_hash = match std::fs::read_to_string(&ref_path) {
        Ok(h) => h.trim().to_string(),
        // No ref by that name: a revision may be a commit hash naming the
        // snapshot directly. Only for an explicit revision — `refs/main`
        // missing means an incomplete download, not a commit.
        Err(_) if revision.is_some() && model_cache.join("snapshots").join(ref_name).is_dir() => {
            ref_name.to_string()
        }
        Err(e) => {
            return Err(anyhow::Error::new(e).context(match revision {
                None => format!(
                    "No default revision for '{}'. Expected refs/main at {}.\n\
                     The model may not have been fully downloaded.",
                    model_id,
                    ref_path.display(),
                ),
                Some(rev) => format!(
                    "Revision '{rev}' of '{}' is not in the HF cache (expected {}, and no \
                     snapshot directory named '{rev}'). Available refs: {}.\n\
                     Download that revision:\n{}",
                    model_id,
                    ref_path.display(),
                    list_refs(&model_cache),
                    download_hint_at(model_id, revision),
                ),
            }));
        }
    };

    let snapshot_dir = model_cache.join("snapshots").join(&snapshot_hash);
    if !snapshot_dir.is_dir() {
        bail!(
            "Snapshot directory not found: {}\n\
             refs/{} points to hash '{}' but that snapshot doesn't exist.",
            snapshot_dir.display(),
            ref_name,
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

    // NEVER for an explicit revision: the sibling snapshots of a multi-branch
    // repo are OTHER checkpoints (turboderp's 2.05 / 3.05 / 4.05 bpw branches
    // share one cache entry), so "the most recent sibling with weights" would
    // silently serve a different quant than the one that was pinned.
    if revision.is_some() {
        bail!(
            "Revision '{}' of {} resolved to snapshot {} which has no weight files. \
             Not falling back to a sibling snapshot: siblings of a pinned revision are \
             other branches, not this checkpoint. Re-download it:\n{}",
            ref_name,
            model_id,
            snapshot_dir.display(),
            download_hint_at(model_id, revision),
        );
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
         snapshots in {} also lack weights — refresh the cache:\n{}",
        snapshot_hash,
        model_id,
        snapshot_dir.display(),
        model_cache.join("snapshots").display(),
        download_hint(model_id),
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
             Download it first:\n{}",
            spec,
            cache_root.display(),
            download_hint(spec),
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
pub(crate) fn snapshot_has_weights(dir: &Path) -> bool {
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
            .is_some_and(|n| n.ends_with(".safetensors") || n.ends_with(".gguf"))
    })
}

/// Pick the most-recently-modified snapshot under `snapshots/` that actually
/// contains weights. Returns `None` if none of the siblings have weights.
pub(crate) fn find_snapshot_with_weights(snapshots_root: &Path) -> Option<PathBuf> {
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

/// The ref names under `refs/` for one cached repo, comma-joined for an error
/// message — so a wrong-revision error shows the operator the branches that
/// ARE downloaded (`4.05bpw_h6_ng6`, …) instead of leaving them to `ls`.
fn list_refs(model_cache: &Path) -> String {
    let mut names: Vec<String> = std::fs::read_dir(model_cache.join("refs"))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    if names.is_empty() {
        "(none)".to_string()
    } else {
        names.join(", ")
    }
}

/// Determine the HF hub cache root directory.
///
/// Precedence (matches official HuggingFace behavior):
/// 1. `cache_dir` argument (from `--cache-dir`)
/// 2. `$HF_HUB_CACHE` env var
/// 3. `$HF_HOME/hub` env var
/// 4. `~/.cache/huggingface/hub`
pub(crate) fn resolve_cache_root(cache_dir: Option<&Path>) -> Result<PathBuf> {
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
#[path = "model_resolver_tests.rs"]
mod tests;
