// SPDX-License-Identifier: AGPL-3.0-only

//! Which files to fetch for a model, and how many bytes that is.
//!
//! Pure: no network, no filesystem. The whole point is that "why is this
//! download three times the size of the model" is decided in one testable
//! function rather than discovered on a metered connection.

/// One file as the Hub describes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemoteFile {
    pub name: String,
    /// Bytes, when the listing declared them. `None` is common: the plain
    /// model endpoint omits sizes, so they come from the tree endpoint.
    pub size: Option<u64>,
}

/// Metadata the loader looks for by exact name.
///
/// Deliberately a fixed list rather than "every .json": a repo can carry
/// evaluation results, quantisation manifests and training configs that the
/// loader never opens.
const METADATA: &[&str] = &[
    "config.json",
    "params.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
];

/// Weight formats Atlas cannot load, and directories that duplicate the model.
///
/// `original/` in particular is why an unfiltered mirror costs double: Llama
/// and Gemma repos ship the reference checkpoint there alongside the
/// safetensors the loader actually reads.
fn is_excluded(name: &str) -> bool {
    const SKIP_DIRS: &[&str] = &["original/", "onnx/", "openvino/", "coreml/", "tflite/"];
    const SKIP_EXT: &[&str] = &[".bin", ".pth", ".pt", ".msgpack", ".h5", ".onnx", ".tflite"];
    SKIP_DIRS.iter().any(|d| name.starts_with(d)) || SKIP_EXT.iter().any(|e| name.ends_with(e))
}

fn is_safetensors(name: &str) -> bool {
    name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")
}

fn is_weight(name: &str) -> bool {
    is_safetensors(name) || is_gguf_weight(name)
}

/// Backbone GGUF (not an mmproj sidecar). Split shards keep the `.gguf` suffix.
fn is_gguf_weight(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.ends_with(".gguf") && !lower.contains("mmproj")
}

/// One quant from a GGUF-only Hub repo. Unsloth-style trees ship every
/// bitwidth at once (terabytes). Prefer Unsloth `UD-Q4_K_M`, then `Q4_K_M`,
/// then a lone file. Return `None` rather than downloading the whole tree.
fn pick_gguf(files: &[RemoteFile]) -> Option<RemoteFile> {
    let cands: Vec<&RemoteFile> = files
        .iter()
        .filter(|f| is_contained(&f.name) && is_gguf_weight(&f.name))
        .collect();
    if cands.is_empty() {
        return None;
    }
    for needle in ["UD-Q4_K_M", "Q4_K_M"] {
        let mut hits: Vec<&RemoteFile> = cands
            .iter()
            .copied()
            .filter(|f| f.name.contains(needle))
            .collect();
        if hits.len() == 1 {
            return Some(hits.remove(0).clone());
        }
        if hits.len() > 1 {
            hits.sort_by_key(|f| (f.size.unwrap_or(u64::MAX), f.name.clone()));
            return Some(hits[0].clone());
        }
    }
    if cands.len() == 1 {
        return Some(cands[0].clone());
    }
    None
}

/// Does this name stay inside the snapshot directory it is joined onto?
///
/// The name is the Hub's `rfilename`, so it is chosen by whoever published
/// the repo, and `select`'s output is joined straight onto the cache path by
/// the downloader. `Path::join` resolves `..` against the parent and treats
/// an absolute component as a *replacement* for everything to its left, so
/// an unvalidated name is an arbitrary-file-write primitive on the machine
/// doing the pull — enough to overwrite another model's shards.
///
/// Weights legitimately sit in subdirectories, so `/` itself has to be
/// allowed; it is the traversal and the rooting that are rejected.
fn is_contained(name: &str) -> bool {
    // `\` is a separator on Windows and no Hub repo needs it; a `C:` prefix
    // is absolute on the platforms that parse it.
    if name.is_empty() || name.starts_with('/') || name.contains('\\') {
        return false;
    }
    let b = name.as_bytes();
    if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
        return false;
    }
    // An empty component is `//`, which some joiners re-root on.
    name.split('/')
        .all(|c| !c.is_empty() && c != "." && c != "..")
}

/// Is this file worth downloading?
pub fn wanted(name: &str) -> bool {
    if !is_contained(name) || is_excluded(name) {
        return false;
    }
    // Only top-level metadata: a `subfolder/config.json` belongs to a
    // component the loader resolves separately, if at all.
    // `.gguf` is NOT wanted here — a 20-quant Hub repo would otherwise
    // download terabytes. `select` adds at most one via `pick_gguf`.
    is_safetensors(name) || (!name.contains('/') && METADATA.contains(&name))
}

/// The files to fetch, in the order to fetch them.
///
/// Small files first, then weights ascending. Two reasons: a wrong-model abort
/// costs under a second instead of a shard, and `config.json` — which decides
/// whether the model is loadable at all — lands before gigabytes do.
pub fn select(files: &[RemoteFile]) -> Vec<RemoteFile> {
    let mut out: Vec<RemoteFile> = files.iter().filter(|f| wanted(&f.name)).cloned().collect();
    if !out.iter().any(|f| is_safetensors(&f.name))
        && let Some(gguf) = pick_gguf(files)
    {
        out.push(gguf);
    }
    out.sort_by(|a, b| {
        let key = |f: &RemoteFile| (is_weight(&f.name), f.size.unwrap_or(0), f.name.clone());
        key(a).cmp(&key(b))
    });
    out
}

/// Does this plan contain anything Atlas could actually load?
///
/// A repo publishing only GGUF is a real and common case. We admit exactly
/// one preferred quant (see [`pick_gguf`]); an unmatched 20-quant listing
/// still yields no weights so the download cannot "succeed" on a tokenizer.
pub fn has_weights(plan: &[RemoteFile]) -> bool {
    plan.iter()
        .any(|f| f.name.ends_with(".safetensors") || is_gguf_weight(&f.name))
}

/// Total bytes of a plan, counting only files whose size is known.
pub fn total_bytes(plan: &[RemoteFile]) -> u64 {
    plan.iter().filter_map(|f| f.size).sum()
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
